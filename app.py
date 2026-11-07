"""Shopper Intent Lab: score the five saved classifiers on an uploaded test CSV.

The models in model/*.joblib were trained by train.py. This app only loads
them and predicts; it never refits anything on the uploaded data.

    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from shopintent.evaluation import (  # noqa: E402
    METRIC_NAMES,
    THRESHOLD,
    Evaluation,
    evaluate,
    report_frame,
)
from shopintent.models import MODEL_NAMES, SLUG_BY_NAME  # noqa: E402
from shopintent.schema import (  # noqa: E402
    CLASS_NAMES,
    FEATURES,
    TARGET,
    missing_columns,
    prepare_features,
    prepare_target,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

# Chart tokens: light surface, one blue series, a one-hue blue ramp for counts.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
SERIES = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

METRIC_HELP = {
    "Accuracy": "Share of all sessions classified correctly. Flattering here: "
    "predicting 'no purchase' for everyone already scores ~84%.",
    "AUC": "Probability that a random buyer gets a higher score than a random "
    "non-buyer. Threshold-free: the slider does not change it.",
    "Precision": "Of the sessions flagged as buyers, how many really bought.",
    "Recall": "Of the real buyers, how many were flagged.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between prediction and truth using all four confusion "
    "cells; -1 to +1, 0 = no better than chance. The fairest single number "
    "on imbalanced data.",
}


# --------------------------------------------------------------------------- data


@st.cache_resource(show_spinner="Loading saved models…")
def load_models() -> dict[str, object]:
    """Load the five fitted pipelines once per server process."""
    return {name: joblib.load(MODEL_DIR / f"{SLUG_BY_NAME[name]}.joblib") for name in MODEL_NAMES}


@st.cache_data(show_spinner=False)
def read_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner="Scoring all five models…")
def score_all(df: pd.DataFrame, threshold: float) -> dict[str, Evaluation]:
    """Predict with every saved model on the same rows. No refitting."""
    models = load_models()
    X, y = prepare_features(df), prepare_target(df)
    return {name: evaluate(models[name], X, y, threshold) for name in MODEL_NAMES}


# ------------------------------------------------------------------------- charts


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def confusion_figure(cm: np.ndarray) -> Figure:
    """2x2 heatmap; each cell shows the count and its share of all rows."""
    fig, ax = plt.subplots(figsize=(4.4, 3.6), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(int(cm.max()), 1))
    total = max(int(cm.sum()), 1)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark_cell = cm[i, j] > 0.55 * cm.max()
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{cm[i, j] / total:.1%}\n{names[i][j]}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if dark_cell else INK,
            )
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Actual", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def roc_figure(result: Evaluation, threshold: float) -> Figure:
    """ROC curve with the current threshold marked as the operating point."""
    fpr, tpr, _ = roc_curve(result.y_true, result.y_score)
    tn, fp, fn, tp = result.confusion.ravel()
    op_fpr = fp / max(fp + tn, 1)
    op_tpr = tp / max(tp + fn, 1)

    fig, ax = plt.subplots(figsize=(4.4, 3.6), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=GRID)
    ax.plot(fpr, tpr, linewidth=2, color=SERIES)
    ax.scatter([op_fpr], [op_tpr], s=64, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(
        f"threshold {threshold:.2f}",
        (op_fpr, op_tpr),
        xytext=(10, -14),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("True positive rate (recall)", color=INK_SECONDARY, fontsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def show_figure(fig: Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- page


def sidebar() -> tuple[pd.DataFrame | None, str, str, float]:
    """Return (data, source label, chosen model, decision threshold)."""
    with st.sidebar:
        st.header("1 · Test data")
        uploaded = st.file_uploader(
            "Upload a test CSV",
            type="csv",
            help="Must contain the 17 feature columns and the Revenue target. "
            "Upload test rows only — the free tier has limited memory.",
        )
        st.download_button(
            "Download sample test_data.csv",
            data=SAMPLE_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )
        use_sample = st.toggle(
            "Use bundled test_data.csv",
            value=uploaded is None,
            disabled=uploaded is not None,
            help="The 20% hold-out set the README table was computed on.",
        )

        st.header("2 · Model")
        model_name = st.selectbox("Classifier", MODEL_NAMES)
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=THRESHOLD,
            step=0.05,
            help="A session is predicted 'Purchase' when its predicted "
            "probability is at least this value. The README uses 0.50.",
        )

    if uploaded is not None:
        return read_csv(uploaded.getvalue()), uploaded.name, model_name, threshold
    if use_sample:
        return read_csv(SAMPLE_CSV.read_bytes()), "test_data.csv (bundled)", model_name, threshold
    return None, "", model_name, threshold


def metric_tiles(metrics: dict[str, float]) -> None:
    for col, name in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
        value = metrics[name]
        col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[name], border=True)


def selected_model_tab(name: str, result: Evaluation, threshold: float) -> None:
    st.subheader(name)
    metric_tiles(result.metrics)

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        show_figure(confusion_figure(result.confusion))
    with right:
        st.markdown(f"**ROC curve** · AUC {result.metrics['AUC']:.4f}")
        if np.isnan(result.metrics["AUC"]):
            st.info("The file has only one class, so a ROC curve is undefined.")
        else:
            show_figure(roc_figure(result, threshold))

    st.markdown("**Classification report**")
    st.dataframe(report_frame(result).style.format(precision=4), width="stretch")

    tn, fp, fn, tp = (int(v) for v in result.confusion.ravel())
    st.caption(
        f"Read it as: of {tp + fn:,} real buyers the model caught {tp:,} and missed {fn:,}; "
        f"it also flagged {fp:,} of {tn + fp:,} non-buyers as buyers."
    )


def comparison_tab(results: dict[str, Evaluation]) -> None:
    table = pd.DataFrame({name: r.metrics for name, r in results.items()}).T[list(METRIC_NAMES)]
    table.index.name = "ML Model Name"
    st.dataframe(
        table.style.format(precision=4).highlight_max(
            axis=0, props="font-weight: 700; background-color: #cde2fb; color: #0b0b0b;"
        ),
        width="stretch",
    )
    st.caption("Highlighted cell = best model for that metric on this file.")

    metric = st.radio("Compare models on", METRIC_NAMES, index=METRIC_NAMES.index("MCC"), horizontal=True)
    chart = table[[metric]].reset_index()
    st.bar_chart(chart, x="ML Model Name", y=metric, color=SERIES, horizontal=True, sort=f"-{metric}")

    best = table["MCC"].idxmax()
    st.success(f"Highest MCC on this file: **{best}** ({table.loc[best, 'MCC']:.4f}).")


def data_tab(df: pd.DataFrame, y: pd.Series) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}", border=True)
    c2.metric("Buyers (Revenue = 1)", f"{int(y.sum()):,}", border=True)
    c3.metric("Buyer rate", f"{y.mean():.1%}", border=True)
    extra = [c for c in df.columns if c not in FEATURES and c != TARGET]
    if extra:
        st.caption(f"Ignored extra columns: {', '.join(extra)}")
    st.dataframe(df.head(100), width="stretch", height=320)


def main() -> None:
    st.set_page_config(page_title="Shopper Intent Lab", page_icon="🛒", layout="wide")
    st.title("🛒 Shopper Intent Lab")
    st.markdown(
        "Will an online shopping session end in a purchase? Five classifiers trained on "
        "the UCI *Online Shoppers Purchasing Intention* dataset, scored live on the CSV you "
        "provide. Models are loaded from `model/*.joblib` and **never retrained** here."
    )

    df, source, model_name, threshold = sidebar()
    if df is None:
        st.info("Upload a test CSV in the sidebar, or switch on the bundled test_data.csv.")
        st.stop()

    missing = missing_columns(df)
    if missing:
        st.error(f"`{source}` is missing required columns: {', '.join(missing)}")
        st.stop()
    try:
        y = prepare_target(df)
        results = score_all(df, threshold)
    except ValueError as err:
        st.error(f"Could not read `{source}`: {err}")
        st.stop()

    st.caption(f"Scoring **{source}** · {len(df):,} rows · threshold {threshold:.2f}")
    tab_model, tab_compare, tab_data = st.tabs(["Selected model", "Compare all five", "Data preview"])
    with tab_model:
        selected_model_tab(model_name, results[model_name], threshold)
    with tab_compare:
        comparison_tab(results)
    with tab_data:
        data_tab(df, y)


main()
