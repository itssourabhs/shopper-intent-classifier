"""The six assignment metrics, computed one way for training and for the app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from shopintent.schema import CLASS_NAMES

METRIC_NAMES: tuple[str, ...] = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
THRESHOLD = 0.5


@dataclass(frozen=True)
class Evaluation:
    """Everything the app and the README need about one model on one file."""

    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    confusion: np.ndarray  # [[TN, FP], [FN, TP]]


def purchase_probability(model, X: pd.DataFrame) -> np.ndarray:
    """P(Revenue = 1) for every row. All five pipelines expose predict_proba."""
    return model.predict_proba(X)[:, 1]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Accuracy, AUC, Precision, Recall, F1 and MCC for the positive class.

    AUC needs both classes present; on a one-class file it is reported as NaN
    rather than crashing the app.
    """
    both_classes = np.unique(y_true).size == 2
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = THRESHOLD) -> Evaluation:
    """Score an already-fitted model. Nothing here refits anything."""
    y_true = np.asarray(y, dtype=int)
    y_score = purchase_probability(model, X)
    y_pred = (y_score >= threshold).astype(int)
    return Evaluation(
        metrics=compute_metrics(y_true, y_pred, y_score),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def report_frame(result: Evaluation) -> pd.DataFrame:
    """sklearn's classification report as a tidy table (rows = classes)."""
    report = classification_report(
        result.y_true,
        result.y_pred,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).T
    frame["support"] = frame["support"].astype(int)
    return frame.rename(columns=str.title)
