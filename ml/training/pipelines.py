"""Model pipelines. Preprocessing lives inside each saved sklearn Pipeline.

Shared by both models: the active feature contract and categorical semantics
(missing -> "__missing__", one-hot, unknown categories ignored). Numeric preprocessing is
model-specific on purpose:
- Logistic Regression: float conversion -> median imputation + missing indicator -> scaling
- XGBoost: float conversion -> native NaN handling (missingness preserved), no scaling
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.features.feature_spec import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MISSING_CATEGORY = "__missing__"
SEED = 42

LOGISTIC_REGRESSION_PARAMS: dict[str, Any] = {"solver": "lbfgs", "C": 1.0, "max_iter": 1000}
XGBOOST_PARAMS: dict[str, Any] = {
    "tree_method": "hist",
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "random_state": SEED,
}
XGBOOST_DEPTHS = (3, 5)
XGBOOST_MAX_ESTIMATORS = 1000
XGBOOST_EARLY_STOPPING_ROUNDS = 50


def to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric features as float64 with NaN for missing (Decimal, Int64, None, pd.NA)."""
    X = pd.DataFrame(X)
    return X.astype(object).where(X.notna(), np.nan).astype("float64")


def to_category(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical features as plain strings with NaN for missing."""
    return (
        pd.DataFrame(X).astype(object).map(lambda value: str(value) if pd.notna(value) else np.nan)
    )


def build_preprocessor(*, impute_and_scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("to_float", FunctionTransformer(to_float, feature_names_out="one-to-one"))
    ]
    if impute_and_scale:
        numeric_steps += [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    categorical = Pipeline(
        [
            ("to_category", FunctionTransformer(to_category, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(impute_and_scale=True)),
            ("model", LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
        ]
    )


@dataclass(frozen=True)
class XGBoostFit:
    pipeline: Pipeline
    max_depth: int
    best_iteration: int
    validation_average_precision_by_depth: dict[int, float]


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    *,
    depths: tuple[int, ...] = XGBOOST_DEPTHS,
    max_estimators: int = XGBOOST_MAX_ESTIMATORS,
    early_stopping_rounds: int = XGBOOST_EARLY_STOPPING_ROUNDS,
) -> XGBoostFit:
    """Early stopping and max_depth are chosen on validation (average precision);
    the test split is never seen here."""
    preprocessor = build_preprocessor(impute_and_scale=False).fit(X_train)
    train_matrix = preprocessor.transform(X_train)
    validation_matrix = preprocessor.transform(X_validation)

    best: tuple[float, int, XGBClassifier] | None = None
    scores: dict[int, float] = {}
    for depth in depths:
        model = XGBClassifier(
            **XGBOOST_PARAMS,
            max_depth=depth,
            n_estimators=max_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            train_matrix, y_train, eval_set=[(validation_matrix, y_validation)], verbose=False
        )
        score = float(
            average_precision_score(y_validation, model.predict_proba(validation_matrix)[:, 1])
        )
        scores[depth] = round(score, 6)
        if best is None or score > best[0]:  # ties keep the shallower model
            best = (score, depth, model)
    assert best is not None
    _, depth, model = best
    return XGBoostFit(
        pipeline=Pipeline([("preprocess", preprocessor), ("model", model)]),
        max_depth=depth,
        best_iteration=int(model.best_iteration),
        validation_average_precision_by_depth=scores,
    )
