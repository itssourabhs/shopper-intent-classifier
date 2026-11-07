"""Checks that the metrics are right, the saved models match the README, and
the app starts cleanly. Run with:  pytest -q
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from shopintent.evaluation import METRIC_NAMES, compute_metrics, evaluate
from shopintent.models import MODEL_NAMES, SLUG_BY_NAME
from shopintent.schema import FEATURES, missing_columns, prepare_features, prepare_target, to_binary

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_metrics_match_hand_calculation() -> None:
    # TP=2, FN=1, FP=1, TN=2  ->  acc 4/6, P = R = F1 = 2/3,
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = 3/9
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.4, 0.6, 0.2, 0.1])
    m = compute_metrics(y_true, y_pred, y_score)
    assert m["Accuracy"] == pytest.approx(4 / 6)
    assert m["Precision"] == pytest.approx(2 / 3)
    assert m["Recall"] == pytest.approx(2 / 3)
    assert m["F1"] == pytest.approx(2 / 3)
    assert m["MCC"] == pytest.approx(1 / 3)
    # 9 buyer/non-buyer pairs; only (0.4 vs 0.6) is ranked wrongly -> 8/9
    assert m["AUC"] == pytest.approx(8 / 9)


@pytest.mark.unit
def test_auc_is_nan_when_only_one_class_present() -> None:
    m = compute_metrics(np.array([0, 0]), np.array([0, 1]), np.array([0.1, 0.7]))
    assert math.isnan(m["AUC"])


@pytest.mark.unit
@pytest.mark.parametrize("raw", [[True, False], ["TRUE", "FALSE"], ["True", "False"], [1, 0], ["yes", "no"]])
def test_to_binary_accepts_common_spellings(raw: list) -> None:
    assert to_binary(pd.Series(raw, name="x")).tolist() == [1, 0]


@pytest.mark.unit
def test_to_binary_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="maybe"):
        to_binary(pd.Series(["TRUE", "maybe"], name="Revenue"))


@pytest.mark.unit
def test_missing_columns_are_reported_in_schema_order() -> None:
    df = pd.DataFrame(columns=[c for c in FEATURES if c not in ("Month", "Weekend")])
    assert missing_columns(df) == ["Month", "Weekend", "Revenue"]
    assert missing_columns(df, include_target=False) == ["Month", "Weekend"]


@pytest.mark.integration
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_saved_models_reproduce_readme_metrics(name: str) -> None:
    """The app and the README must report the same numbers for test_data.csv."""
    test_df = pd.read_csv(ROOT / "test_data.csv")
    expected = pd.read_csv(ROOT / "model" / "test_metrics.csv", index_col=0).loc[name]
    model = joblib.load(ROOT / "model" / f"{SLUG_BY_NAME[name]}.joblib")
    got = evaluate(model, prepare_features(test_df), prepare_target(test_df)).metrics
    for metric in METRIC_NAMES:
        assert got[metric] == pytest.approx(expected[metric], abs=5e-5), metric


@pytest.mark.integration
def test_app_runs_and_switches_models() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert [m.label for m in at.metric[:6]] == list(METRIC_NAMES)
    assert list(at.selectbox[0].options) == list(MODEL_NAMES)

    at.selectbox[0].select("KNN").run()
    assert not at.exception
    assert at.subheader[0].value == "KNN"
