"""Column definitions and input cleaning for the UCI Online Shoppers dataset.

Every CSV that reaches a model (the training file, test_data.csv, or a file
uploaded to the app) passes through `prepare_features` / `prepare_target`, so
the models always see the same column order and the same data types.
"""

from __future__ import annotations

import pandas as pd

TARGET = "Revenue"

# Counts of pages visited and seconds spent per page group, plus the
# Google-Analytics style rates. All non-negative and heavily right-skewed.
NUMERIC_FEATURES: tuple[str, ...] = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)

# OperatingSystems / Browser / Region / TrafficType are stored as integers but
# they are IDs, not quantities (browser 4 is not "twice" browser 2), so they
# are one-hot encoded alongside the two text columns.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
)

BINARY_FEATURES: tuple[str, ...] = ("Weekend",)

FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

CLASS_NAMES: tuple[str, str] = ("No purchase", "Purchase")

_TEXT_CATEGORICALS = ("Month", "VisitorType")
_TRUE_TOKENS = frozenset({"true", "1", "1.0", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "0.0", "no", "n", "f"})


def missing_columns(df: pd.DataFrame, include_target: bool = True) -> list[str]:
    """Return the required columns that `df` does not have, in schema order."""
    required = FEATURES + ((TARGET,) if include_target else ())
    present = set(df.columns)
    return [col for col in required if col not in present]


def to_binary(values: pd.Series) -> pd.Series:
    """Map TRUE/FALSE, True/False, yes/no or 1/0 to integers 1/0.

    Raises ValueError naming the offending values instead of guessing.
    """
    tokens = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(tokens) - _TRUE_TOKENS - _FALSE_TOKENS)
    if unknown:
        raise ValueError(f"Column '{values.name}' has non-boolean values: {unknown[:5]}")
    return tokens.isin(_TRUE_TOKENS).astype(int)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 17 model inputs in a fixed order with consistent dtypes."""
    missing = missing_columns(df, include_target=False)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    for col in CATEGORICAL_FEATURES:
        if col in _TEXT_CATEGORICALS:
            # object dtype keeps OneHotEncoder happy on pandas 1.x, 2.x and 3.x
            out[col] = df[col].astype(str).str.strip().astype(object)
        else:
            out[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    for col in BINARY_FEATURES:
        out[col] = to_binary(df[col])
    return out


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Return the Revenue column as 0/1 integers (1 = the session bought)."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing")
    return to_binary(df[TARGET]).rename(TARGET)
