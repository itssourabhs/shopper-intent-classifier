"""The five classifiers, their shared preprocessing, and their search grids.

Each saved model is a full scikit-learn Pipeline (preprocessing + classifier),
so the app can feed it raw CSV columns and never has to repeat a
preprocessing step by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shopintent.schema import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 5984  # last four digits of my BITS ID, fixed for reproducibility


@dataclass(frozen=True)
class ModelSpec:
    """One classifier: display name, file name, estimator and CV grid."""

    name: str
    slug: str
    estimator: ClassifierMixin
    param_grid: dict[str, list[Any]]
    # Applied only to the final refit, after the grid search has picked the
    # hyperparameters. Used to give the SVM probabilities (see below).
    wrap: Optional[Callable[[ClassifierMixin], ClassifierMixin]] = None


def calibrate(estimator: ClassifierMixin) -> ClassifierMixin:
    """Give an SVM a predict_proba so AUC can be computed.

    An SVM outputs a signed distance from its boundary, not a probability.
    Platt scaling fits a small logistic curve that maps that distance to
    P(purchase), using 5 internal folds so the curve is not fitted on the same
    rows the SVM was. (This replaces SVC(probability=True), deprecated in
    scikit-learn 1.9.)
    """
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=5, ensemble=False)


def build_preprocessor() -> ColumnTransformer:
    """log1p + standardise the numbers, one-hot the IDs, pass Weekend through.

    log1p tames the long right tails (ProductRelated_Duration runs to 63,974 s
    while its median is ~600 s). Without it, KNN distances and the SVM kernel
    are dominated by a handful of marathon sessions.
    """
    numeric = Pipeline(
        steps=[
            # one-to-one: column names survive, so coefficients can be read by name
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("bin", "passthrough", list(BINARY_FEATURES)),
        ],
        # Dense output: GaussianNB cannot take a sparse matrix.
        sparse_threshold=0.0,
    )


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Wrap a classifier with the shared preprocessing."""
    return Pipeline(steps=[("prep", build_preprocessor()), ("clf", estimator)])


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="Logistic Regression",
        slug="logistic_regression",
        estimator=LogisticRegression(max_iter=3000),
        param_grid={
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    ModelSpec(
        name="Decision Tree",
        slug="decision_tree",
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [1, 10, 30],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    ModelSpec(
        name="KNN",
        slug="knn",
        estimator=KNeighborsClassifier(),
        param_grid={
            "clf__n_neighbors": [5, 15, 25, 41],
            "clf__weights": ["uniform", "distance"],
        },
    ),
    ModelSpec(
        name="Naive Bayes",
        slug="naive_bayes",
        estimator=GaussianNB(),
        param_grid={"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
    ),
    ModelSpec(
        name="Support Vector Machine",
        slug="svm",
        estimator=SVC(kernel="rbf"),
        param_grid={
            "clf__C": [0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
        wrap=calibrate,
    ),
)

MODEL_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODEL_SPECS)
SLUG_BY_NAME: dict[str, str] = {spec.name: spec.slug for spec in MODEL_SPECS}
