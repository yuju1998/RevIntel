"""The lead conversion model's input/output contract, shared by training and serving.

- Input ("wire form"): the active features in spec order, as JSON-friendly types:
  strings for categoricals, int64 for counts, float64 (NaN = missing) for numerics.
- Output: `predict_proba` probabilities, shape (rows, 2) = [P(not converted), P(converted)].

Tracked training logs models with this signature; online scoring checks a loaded model
against it and validates every prediction. MLflow is imported lazily so importing this
module stays cheap.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml.features.feature_spec import ACTIVE_FEATURES, FEATURES_BY_NAME

if TYPE_CHECKING:
    from mlflow.models import ModelSignature

WIRE_TYPES = {"categorical": "string", "count": "long", "numeric": "double"}
# The MLflow pyfunc inference function: class probabilities, never hard-class predict().
PYFUNC_PREDICT_FN = "predict_proba"
OUTPUT_NAME = "probabilities"
OUTPUT_SHAPE = (-1, 2)
POSITIVE_CLASS = 1
_ROW_SUM_TOLERANCE = 1e-6


class ModelContractError(ValueError):
    """A model or its output does not match the lead conversion contract."""


def model_signature() -> ModelSignature:
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema, TensorSpec

    inputs = Schema(
        [ColSpec(WIRE_TYPES[FEATURES_BY_NAME[name].kind], name) for name in ACTIVE_FEATURES]
    )
    outputs = Schema([TensorSpec(np.dtype("float64"), OUTPUT_SHAPE, OUTPUT_NAME)])
    return ModelSignature(inputs=inputs, outputs=outputs)


def check_signature(signature: ModelSignature | None) -> None:
    """Fail unless `signature` is exactly the contract's input and probability output."""
    if signature is None:
        raise ModelContractError("model has no signature")
    expected = model_signature()
    if signature.inputs is None or signature.inputs.to_dict() != expected.inputs.to_dict():
        raise ModelContractError("model input signature does not match the active features")
    if signature.outputs is None or signature.outputs.to_dict() != expected.outputs.to_dict():
        raise ModelContractError("model output signature is not a (rows, 2) probability array")


def check_inference_function(flavors: dict[str, Any]) -> None:
    """Fail unless the MLflow pyfunc flavor serves `predict_proba`."""
    predict_fn = flavors.get("python_function", {}).get("predict_fn")
    if predict_fn != PYFUNC_PREDICT_FN:
        raise ModelContractError(
            f"model pyfunc inference function is {predict_fn!r}, expected {PYFUNC_PREDICT_FN!r}"
        )


def to_wire_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Active features as callers send them: strings, int64 counts, float64 (NaN = missing)."""
    columns = {}
    for name in ACTIVE_FEATURES:
        kind = FEATURES_BY_NAME[name].kind
        column = features[name].astype(object).where(features[name].notna(), None)
        if kind == "categorical":
            columns[name] = column.map(lambda v: None if v is None else str(v))
        elif kind == "count":
            columns[name] = column.astype("int64")
        else:
            columns[name] = column.astype("float64")
    return pd.DataFrame(columns).reset_index(drop=True)


def wire_record(wire: pd.DataFrame, row: int = 0) -> dict[str, Any]:
    """One wire-form row as JSON-safe Python values (NaN -> None)."""
    record = {}
    for name in ACTIVE_FEATURES:
        value = wire[name].iloc[row]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            record[name] = None
        elif isinstance(value, np.generic):
            record[name] = value.item()
        else:
            record[name] = value
    return record


def positive_class_probabilities(output: Any, rows: int) -> np.ndarray:
    """Validate `predict_proba` output and return P(converted) for each row."""
    try:
        array = np.asarray(output, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"model output is not numeric: {type(output).__name__}") from exc
    if array.shape != (rows, 2):
        raise ModelContractError(f"model output has shape {array.shape}, expected ({rows}, 2)")
    if not np.isfinite(array).all():
        raise ModelContractError("model output contains non-finite values")
    if ((array < 0) | (array > 1)).any():
        raise ModelContractError("model output contains values outside [0, 1]")
    if (np.abs(array.sum(axis=1) - 1) > _ROW_SUM_TOLERANCE).any():
        raise ModelContractError("model output rows do not sum to 1")
    return array[:, POSITIVE_CLASS]
