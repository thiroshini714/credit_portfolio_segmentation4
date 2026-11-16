"""Data cleaning: duplicates, inconsistent values, dates, missing values.

Outlier handling is split in two on purpose:

* ``clean_data`` keeps the real (but valid) extreme values so they can be
  studied in the anomaly-detection step.
* ``winsorise`` produces a capped copy used only for clustering, so that a
  handful of extreme incomes do not dominate the distance calculations.
"""

import re

import numpy as np
import pandas as pd

import config

NUMERIC_COLUMNS = ["loan_amnt", "annual_inc", "dti", "fico", "emp_length"]
HOME_MAP = {"mortgage": "mortgage", "rent": "rent", "own": "own",
            "any": "other", "none": "other", "other": "other"}


def to_numeric_clean(series):
    """Convert strings such as '$12,000' or '17.5%' into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype("string").str.replace(r"[$,%\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_emp_length(value):
    """Turn '10+ years', '< 1 year', 'n/a', 3 ... into a number of years."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower()
    if text in ("", "n/a", "na", "nan", "none", "unknown"):
        return np.nan
    if "<" in text:
        return 0.0
    digits = re.findall(r"\d+", text)
    return float(digits[0]) if digits else np.nan


def parse_dates(series):
    """Parse dates such as 'Dec-2015', '2015-12-01' or '12/2015'."""
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%b-%Y", errors="coerce")
    unresolved = parsed.isna() & text.notna()
    if unresolved.any():
        parsed.loc[unresolved] = pd.to_datetime(
            text[unresolved], format="mixed", errors="coerce")
    return parsed


def encode_default(series):
    """Return 1.0 (defaulted), 0.0 (repaid) or NaN (outcome unknown).

    Text statuses follow the usual Lending Club convention: 'Charged Off'
    and 'Default' are defaults, 'Fully Paid' is repaid, and loans still in
    progress ('Current', 'Late', 'In Grace Period') have no final outcome.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        out = (series > 0).astype(float)
        out[series.isna()] = np.nan
        return out
    text = series.astype("string").str.strip().str.lower()
    bad = text.str.contains(r"charged off|default|^yes$|^true$|^1$|^bad",
                            regex=True, na=False)
    good = text.str.contains(r"fully paid|^no$|^false$|^0$|^good",
                             regex=True, na=False)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[good] = 0.0
    out[bad] = 1.0
    return out


def clean_data(df):
    """Clean a standardised dataframe and return ``(clean_df, report)``."""
    report = {"rows_raw": int(len(df))}
    df = df.copy()

    # ---- text normalisation -------------------------------------------
    for col in ("purpose", "home_ownership", "state"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # ---- duplicates ---------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = int(before - len(df))
    if "loan_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="loan_id", keep="first")
        report["duplicate_ids_removed"] = int(before - len(df))
    else:
        report["duplicate_ids_removed"] = 0

    # ---- type conversion ---------------------------------------------
    for col in ("loan_amnt", "annual_inc", "dti", "fico"):
        if col in df.columns:
            df[col] = to_numeric_clean(df[col])
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(parse_emp_length).astype(float)

    if "issue_date" in df.columns:
        df["issue_date"] = parse_dates(df["issue_date"])
        df["issue_year"] = df["issue_date"].dt.year.astype("Float64")
        df["issue_month"] = df["issue_date"].dt.month.astype("Float64")
        df["issue_year"] = df["issue_year"].astype(float)
        df["issue_month"] = df["issue_month"].astype(float)

    # ---- inconsistent categorical values -----------------------------
    if "purpose" in df.columns:
        df["purpose"] = (df["purpose"].str.lower()
                         .str.replace(r"[\s\-]+", "_", regex=True))
    if "home_ownership" in df.columns:
        df["home_ownership"] = (df["home_ownership"].str.lower()
                                .map(HOME_MAP).astype("string"))
    if "state" in df.columns:
        df["state"] = df["state"].str.upper()
        df.loc[~df["state"].str.fullmatch(r"[A-Z]{2}", na=False),
               "state"] = pd.NA

    # ---- impossible numeric values -> missing ------------------------
    invalid = {}
    rules = {
        "fico": lambda s: (s < config.FICO_MIN) | (s > config.FICO_MAX),
        "annual_inc": lambda s: s <= 0,
        "loan_amnt": lambda s: s <= 0,
        "dti": lambda s: (s < config.DTI_MIN) | (s > config.DTI_MAX),
        "emp_length": lambda s: (s < 0) | (s > 50),
    }
    for col, rule in rules.items():
        if col in df.columns:
            mask = rule(df[col]).fillna(False)
            invalid[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
    report["invalid_values_set_missing"] = invalid

    # ---- default status ----------------------------------------------
    if "default" in df.columns:
        df["default"] = encode_default(df["default"])

    # ---- missing values ----------------------------------------------
    feature_cols = [c for c in df.columns
                    if c in NUMERIC_COLUMNS + ["purpose", "home_ownership",
                                               "state"]]
    report["missing_before_imputation"] = {
        c: int(df[c].isna().sum()) for c in feature_cols}
    strategies = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            strategies[col] = "median"
    for col in ("purpose", "home_ownership"):
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "other")
            strategies[col] = "mode"
    if "state" in df.columns:
        df["state"] = df["state"].fillna("UNK")
        strategies["state"] = "'UNK' label"
    report["imputation_strategy"] = strategies

    # Rows without a usable date keep NaN year (excluded from trend charts).
    df = df.reset_index(drop=True)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)

    report["rows_final"] = int(len(df))
    if "default" in df.columns:
        known = df["default"].notna()
        report["default_known_rows"] = int(known.sum())
        report["overall_default_rate"] = (
            float(df.loc[known, "default"].mean()) if known.any() else None)
    return df, report


def outlier_summary(df, columns=None):
    """IQR-based outlier counts per numeric column (reporting only)."""
    columns = columns or [c for c in NUMERIC_COLUMNS if c in df.columns]
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"feature": col, "q1": q1, "q3": q3,
                     "lower_fence": low, "upper_fence": high,
                     "n_outliers": count,
                     "pct_outliers": 100.0 * count / len(df),
                     "skewness": float(df[col].skew())})
    return pd.DataFrame(rows)


def winsorise(df, columns=None, lower=None, upper=None):
    """Cap numeric columns at the given percentiles (returns a copy)."""
    lower = config.WINSOR_LOWER if lower is None else lower
    upper = config.WINSOR_UPPER if upper is None else upper
    columns = columns or [c for c in NUMERIC_COLUMNS if c in df.columns]
    out = df.copy()
    for col in columns:
        lo, hi = out[col].quantile([lower, upper])
        out[col] = out[col].clip(lo, hi)
    return out
