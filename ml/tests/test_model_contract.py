"""The model contract shared by tracked training and online scoring."""

import numpy as np
import pandas as pd
import pytest
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from ml.features.feature_spec import ACTIVE_FEATURES
from ml.model_contract import (
    ModelContractError,
    check_inference_function,
    check_signature,
    model_signature,
    positive_class_probabilities,
    to_wire_frame,
    wire_record,
)
from ml.tests.conftest import FeatureInputs


def test_logged_signature_satisfies_the_contract() -> None:
    check_signature(model_signature())
    check_inference_function({"python_function": {"predict_fn": "predict_proba"}})


def test_hard_class_outputs_and_functions_are_rejected() -> None:
    hard_class = ModelSignature(
        inputs=model_signature().inputs, outputs=Schema([ColSpec("boolean", "converted")])
    )
    with pytest.raises(ModelContractError, match="probability"):
        check_signature(hard_class)
    with pytest.raises(ModelContractError, match="predict_proba"):
        check_inference_function({"python_function": {"predict_fn": "predict"}})


def test_positive_class_is_column_one() -> None:
    probabilities = positive_class_probabilities(np.array([[0.9, 0.1], [0.25, 0.75]]), rows=2)

    np.testing.assert_allclose(probabilities, [0.1, 0.75])


def test_wire_record_is_json_safe(feature_inputs: FeatureInputs) -> None:
    rows = feature_inputs.features.head(1).copy()
    rows["days_since_last_activity"] = np.nan
    wire = to_wire_frame(rows)

    record = wire_record(wire)

    assert list(record) == list(ACTIVE_FEATURES)
    assert record["days_since_last_activity"] is None
    assert all(type(v) in (str, int, float, type(None)) for v in record.values())
    pd.testing.assert_frame_equal(to_wire_frame(pd.DataFrame([record])), wire, check_dtype=False)
