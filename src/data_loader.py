"""Load the raw loan CSV and map its columns to canonical names.

The real Lending Club extract may use different header names from one copy
to another (``addr_state`` vs ``state``, ``issue_d`` vs ``issue_date`` ...).
Columns are therefore detected through an alias table. Any mapping can be
overridden from the command line with ``--map canonical=SourceColumn``.
"""

import re
from pathlib import Path

import pandas as pd

ALIASES = {
    "loan_id": ["loan_id", "id", "member_id", "loanid"],
    "loan_amnt": ["loan_amnt", "loan_amount", "funded_amnt", "loanamount",
                  "amount"],
    "annual_inc": ["annual_inc", "annual_income", "annualincome", "income"],
    "dti": ["dti", "debt_to_income", "debt_to_income_ratio",
            "debt_income_ratio"],
    "fico": ["fico", "fico_score", "credit_score", "fico_range"],
    "emp_length": ["emp_length", "employment_length", "emp_len",
                   "years_employed"],
    "purpose": ["purpose", "loan_purpose"],
    "home_ownership": ["home_ownership", "home_ownership_status",
                       "homeownership"],
    "state": ["addr_state", "state", "address_state", "borrower_state"],
    "issue_date": ["issue_d", "issue_date", "issued", "loan_issue_date"],
    "default": ["default", "loan_status", "default_status", "target",
                "is_default", "bad_loan", "loan_default", "charged_off"],
}
FICO_PAIR = ("fico_range_low", "fico_range_high")
REQUIRED = ["loan_amnt", "annual_inc", "fico"]


def normalise_name(name):
    """Lower-case a header and replace non-alphanumerics with underscores."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def load_csv(path):
    """Read a CSV file, tolerating encodings and a Lending Club notes row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place loans_granting_model_dataset.csv in data/raw/ or pass "
            "--data <path>. To try the project without the real data, run: "
            "python -m src.generate_sample_data"
        )
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    if df.shape[1] == 1:  # Lending Club files sometimes start with a notes row
        df = pd.read_csv(path, low_memory=False, skiprows=1)
    return df


def standardise_columns(df, overrides=None):
    """Return ``(standardised_df, mapping)`` with canonical column names.

    ``mapping`` records which source column(s) fed each canonical column so
    the choice can be documented and audited.
    """
    overrides = overrides or {}
    lookup = {normalise_name(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    mapping = {}

    for canonical, aliases in ALIASES.items():
        source = None
        if canonical in overrides:
            source = overrides[canonical]
            if source not in df.columns:
                raise ValueError(
                    f"--map {canonical}={source}: column not found. "
                    f"Available: {list(df.columns)}"
                )
        else:
            for alias in aliases:
                if alias in lookup:
                    source = lookup[alias]
                    break
        if source is not None:
            out[canonical] = df[source]
            mapping[canonical] = source

    if "fico" not in out.columns and all(c in lookup for c in FICO_PAIR):
        low = pd.to_numeric(df[lookup[FICO_PAIR[0]]], errors="coerce")
        high = pd.to_numeric(df[lookup[FICO_PAIR[1]]], errors="coerce")
        out["fico"] = (low + high) / 2.0
        mapping["fico"] = f"mean({lookup[FICO_PAIR[0]]}, {lookup[FICO_PAIR[1]]})"

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(
            f"Required columns could not be detected: {missing}.\n"
            f"Columns found in file: {list(df.columns)}\n"
            "Use --map canonical=SourceColumn to map them manually."
        )
    return out, mapping


def parse_overrides(pairs):
    """Convert ``["fico=my_score", ...]`` CLI items into a dict."""
    result = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Invalid --map value '{item}'. Use a=b format.")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result
