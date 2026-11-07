"""Train, tune and save the five classifiers, then score them on held-out data.

Run from the project folder:

    python train.py

Outputs
    test_data.csv                 the 20% hold-out, raw columns + Revenue
    model/<slug>.joblib           one fitted Pipeline per classifier
    model/test_metrics.csv        the README comparison table
    model/training_summary.json   chosen hyperparameters, CV score, timings
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from shopintent.evaluation import METRIC_NAMES, evaluate
from shopintent.models import MODEL_SPECS, RANDOM_STATE, ModelSpec, build_pipeline
from shopintent.schema import FEATURES, TARGET, prepare_features, prepare_target

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
TEST_SIZE = 0.20
CV_FOLDS = 5

log = logging.getLogger("train")


@dataclass(frozen=True)
class TrainedModel:
    """A fitted pipeline plus how it was chosen."""

    spec: ModelSpec
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_mcc: float
    seconds: float


def load_dataset(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the UCI file and drop exact duplicate sessions.

    125 rows are byte-for-byte repeats. Left in, a copy can land in train and
    its twin in test, which flatters every score a little.
    """
    raw = pd.read_csv(path)
    deduped = raw.drop_duplicates().reset_index(drop=True)
    log.info("Loaded %d rows, %d after removing duplicates", len(raw), len(deduped))
    return deduped


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split so both halves keep the ~15.6% purchase rate."""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tune_and_fit(spec: ModelSpec, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
    """Grid-search on the training data only, then refit the winner.

    Selection metric is MCC: with 84% non-buyers, accuracy rewards a model
    that never predicts a purchase, while MCC punishes it.
    """
    start = time.perf_counter()
    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        scoring=make_scorer(matthews_corrcoef),
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=False,
    )
    search.fit(X, y)

    best_params = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
    estimator = clone(spec.estimator).set_params(**best_params)
    if spec.wrap is not None:
        estimator = spec.wrap(estimator)
    final = build_pipeline(estimator).fit(X, y)
    seconds = time.perf_counter() - start

    log.info(
        "%-24s CV MCC %.4f  %s  (%.1fs)",
        spec.name,
        search.best_score_,
        best_params,
        seconds,
    )
    return TrainedModel(
        spec=spec,
        pipeline=final,
        best_params=best_params,
        cv_mcc=float(search.best_score_),
        seconds=seconds,
    )


def train_all(train_df: pd.DataFrame) -> list[TrainedModel]:
    """Tune and fit all five classifiers on the same training rows."""
    X, y = prepare_features(train_df), prepare_target(train_df)
    return [tune_and_fit(spec, X, y) for spec in MODEL_SPECS]


def score_all(models: list[TrainedModel], test_df: pd.DataFrame) -> pd.DataFrame:
    """The README comparison table: one row per model, six metric columns."""
    X, y = prepare_features(test_df), prepare_target(test_df)
    rows = {m.spec.name: evaluate(m.pipeline, X, y).metrics for m in models}
    table = pd.DataFrame.from_dict(rows, orient="index")[list(METRIC_NAMES)]
    table.index.name = "ML Model Name"
    return table


def save_artifacts(
    models: list[TrainedModel],
    table: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path = ROOT,
) -> None:
    """Write the models, the hold-out CSV and the metric table to disk."""
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    for m in models:
        joblib.dump(m.pipeline, model_dir / f"{m.spec.slug}.joblib", compress=3)

    test_df[list(FEATURES) + [TARGET]].to_csv(out_dir / "test_data.csv", index=False)
    table.round(4).to_csv(model_dir / "test_metrics.csv")

    summary = {
        "dataset": "UCI Online Shoppers Purchasing Intention (id 468)",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "selection_metric": "MCC",
        "sklearn_version": sklearn.__version__,
        "models": {
            m.spec.name: {
                "file": f"model/{m.spec.slug}.joblib",
                "best_params": m.best_params,
                "cv_mcc": round(m.cv_mcc, 4),
                "train_seconds": round(m.seconds, 1),
            }
            for m in models
        },
    }
    (model_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Saved %d models, test_data.csv and metrics to %s", len(models), out_dir)


def run(save: bool = True) -> tuple[list[TrainedModel], pd.DataFrame]:
    """Full pipeline: load -> split -> tune/fit -> score -> (optionally) save."""
    df = load_dataset()
    train_df, test_df = split_dataset(df)
    log.info(
        "Train %d rows (%.1f%% buyers) | Test %d rows (%.1f%% buyers)",
        len(train_df),
        100 * train_df[TARGET].mean(),
        len(test_df),
        100 * test_df[TARGET].mean(),
    )
    models = train_all(train_df)
    table = score_all(models, test_df)
    if save:
        save_artifacts(models, table, test_df)
    return models, table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-save", action="store_true", help="train and score only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    _, table = run(save=not args.no_save)
    log.info("Hold-out results\n%s", table.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
