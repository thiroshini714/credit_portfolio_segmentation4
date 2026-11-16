"""Cluster profiling, default-rate comparison, geographic and trend views.

Important: ``default`` is used here only AFTER clustering, never to form it.
"""

import numpy as np
import pandas as pd
from scipy import stats

PROFILE_NUMERIC = ["loan_amnt", "annual_inc", "dti", "fico", "emp_length",
                   "loan_to_income"]
DESCRIPTORS = {
    "fico": ("Higher credit score", "Lower credit score"),
    "annual_inc": ("Higher income", "Lower income"),
    "dti": ("High DTI", "Low DTI"),
    "loan_amnt": ("Larger loans", "Smaller loans"),
    "emp_length": ("Longer employment", "Shorter employment"),
    "loan_to_income": ("High loan-to-income", "Low loan-to-income"),
}


def ensure_ratio(df):
    """Add loan_to_income when possible."""
    out = df.copy()
    if "loan_to_income" not in out.columns and \
            {"loan_amnt", "annual_inc"} <= set(out.columns):
        out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"]
    return out


def wilson_interval(defaults, total, z=1.96):
    """Wilson score 95% interval for a default rate."""
    if total == 0:
        return np.nan, np.nan
    p = defaults / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mode_with_share(series):
    """Most common value and its share (as a fraction)."""
    counts = series.value_counts(normalize=True)
    if counts.empty:
        return "n/a", np.nan
    return str(counts.index[0]), float(counts.iloc[0])


def profile_clusters(df, label_col, default_col="default"):
    """One row per cluster with size, feature means/medians and top levels."""
    df = ensure_ratio(df)
    numeric = [c for c in PROFILE_NUMERIC if c in df.columns]
    rows = []
    for cluster, part in df.groupby(label_col):
        row = {"cluster": int(cluster), "size": int(len(part)),
               "pct_of_portfolio": 100.0 * len(part) / len(df)}
        for col in numeric:
            row[f"mean_{col}"] = float(part[col].mean())
            row[f"median_{col}"] = float(part[col].median())
        if "purpose" in part:
            row["top_purpose"], row["top_purpose_share"] = _mode_with_share(
                part["purpose"])
        if "home_ownership" in part:
            row["top_home_ownership"], row["top_home_share"] = \
                _mode_with_share(part["home_ownership"])
        if "state" in part:
            row["top_state"], row["top_state_share"] = _mode_with_share(
                part["state"])
        if default_col in part:
            known = part[default_col].dropna()
            row["default_rate"] = (float(known.mean())
                                   if len(known) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def _robust_scale(df, numeric):
    """Median and robust spread (IQR / 1.349) so outliers do not distort."""
    median = df[numeric].median()
    spread = (df[numeric].quantile(0.75) - df[numeric].quantile(0.25)) / 1.349
    fallback = df[numeric].std()
    spread = spread.where(spread > 0, fallback).replace(0, np.nan)
    return median, spread


def name_segments(df, label_col, z_cut=0.4, max_terms=3):
    """Create readable segment names from robust standardised medians."""
    df = ensure_ratio(df)
    numeric = [c for c in DESCRIPTORS if c in df.columns]
    overall_median, overall_spread = _robust_scale(df, numeric)
    names = {}
    for cluster, part in df.groupby(label_col):
        cluster = int(cluster)
        if cluster == -1:
            names[cluster] = "Noise / unclustered (DBSCAN)"
            continue
        z = ((part[numeric].median() - overall_median)
             / overall_spread).dropna()
        strong = z[z.abs() >= z_cut].sort_values(key=abs, ascending=False)
        terms = [DESCRIPTORS[c][0] if v > 0 else DESCRIPTORS[c][1]
                 for c, v in strong.head(max_terms).items()]
        label = ", ".join(terms) if terms else "Mid-range / mainstream"
        names[cluster] = f"Cluster {cluster}: {label}"
    return names


def standardised_profile(df, label_col):
    """Cluster medians as robust z-scores of the portfolio (for heatmaps)."""
    df = ensure_ratio(df)
    numeric = [c for c in PROFILE_NUMERIC if c in df.columns]
    overall_median, overall_spread = _robust_scale(df, numeric)
    medians = df.groupby(label_col)[numeric].median()
    return (medians - overall_median) / overall_spread


def default_rate_table(df, label_col, names=None, default_col="default"):
    """Default rate per cluster with Wilson CI and lift vs the portfolio."""
    known = df[df[default_col].notna()]
    overall = float(known[default_col].mean()) if len(known) else np.nan
    rows = []
    for cluster, part in df.groupby(label_col):
        outcome = part[default_col].dropna()
        defaults = int(outcome.sum())
        total = int(len(outcome))
        low, high = wilson_interval(defaults, total)
        rate = defaults / total if total else np.nan
        if total and low > overall:
            tier = "Higher risk"
        elif total and high < overall:
            tier = "Lower risk"
        else:
            tier = "In line with portfolio"
        rows.append({
            "cluster": int(cluster),
            "segment": (names or {}).get(int(cluster), f"Cluster {cluster}"),
            "n_loans": int(len(part)), "n_with_outcome": total,
            "n_defaults": defaults, "default_rate": rate,
            "ci_low": low, "ci_high": high,
            "lift_vs_portfolio": (rate / overall if overall else np.nan),
            "risk_tier": tier, "portfolio_default_rate": overall})
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def chi_square_default(df, label_col, default_col="default"):
    """Chi-square test of independence between cluster and default."""
    known = df[df[default_col].notna() & (df[label_col] != -1)]
    table = pd.crosstab(known[label_col], known[default_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "dof": 0,
                "cramers_v": np.nan, "n": int(len(known))}
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof),
            "cramers_v": v, "n": int(n)}


def state_summary(df, default_col="default"):
    """Portfolio view by state (volume, averages, default rate)."""
    if "state" not in df.columns:
        return pd.DataFrame()
    df = ensure_ratio(df)
    rows = []
    for state, part in df.groupby("state"):
        outcome = part[default_col].dropna() if default_col in part else []
        rows.append({
            "state": state, "n_loans": int(len(part)),
            "total_loan_amount": float(part["loan_amnt"].sum()),
            "avg_loan_amnt": float(part["loan_amnt"].mean()),
            "avg_annual_inc": float(part["annual_inc"].mean()),
            "avg_fico": float(part["fico"].mean()),
            "avg_dti": float(part["dti"].mean()) if "dti" in part else np.nan,
            "n_with_outcome": int(len(outcome)),
            "default_rate": (float(np.mean(outcome))
                             if len(outcome) else np.nan)})
    return pd.DataFrame(rows).sort_values("n_loans", ascending=False)


def trend_summary(df, default_col="default"):
    """Loan volume, average amount and default rate by issue year."""
    if "issue_year" not in df.columns:
        return pd.DataFrame()
    valid = df[df["issue_year"].notna()]
    rows = []
    for year, part in valid.groupby("issue_year"):
        outcome = part[default_col].dropna()
        rows.append({
            "issue_year": int(year), "n_loans": int(len(part)),
            "avg_loan_amnt": float(part["loan_amnt"].mean()),
            "avg_fico": float(part["fico"].mean()),
            "default_rate": float(outcome.mean()) if len(outcome)
            else np.nan,
            "n_with_outcome": int(len(outcome))})
    return pd.DataFrame(rows).sort_values("issue_year")


def segment_mix_by_year(df, label_col):
    """Share of each cluster in every issue year (portfolio monitoring)."""
    if "issue_year" not in df.columns:
        return pd.DataFrame()
    valid = df[df["issue_year"].notna()]
    return pd.crosstab(valid["issue_year"].astype(int), valid[label_col],
                       normalize="index") * 100.0


def cluster_state_mix(df, label_col, top_n=8):
    """Percentage of each cluster's loans in the largest states."""
    if "state" not in df.columns:
        return pd.DataFrame()
    top = df["state"].value_counts().head(top_n).index
    sub = df[df["state"].isin(top)]
    return pd.crosstab(sub[label_col], sub["state"], normalize="index") * 100
