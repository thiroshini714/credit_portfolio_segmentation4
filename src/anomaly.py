"""Anomaly detection: DBSCAN noise points plus feature-level extremes."""

import numpy as np
import pandas as pd

import config

FEATURES = ["loan_amnt", "annual_inc", "dti", "fico", "emp_length"]
SKEWED = ["loan_amnt", "annual_inc"]


def robust_z(series):
    """Median/MAD based z-score (falls back to std when MAD is zero)."""
    median = series.median()
    mad = (series - median).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        mad = series.std()
    if not mad or np.isnan(mad):
        return pd.Series(0.0, index=series.index)
    return (series - median) / mad


def detect_anomalies(df, dbscan_labels, threshold=None):
    """Flag unusual borrower profiles.

    A borrower is flagged when DBSCAN labels it noise, or when at least one
    feature is extreme (|robust z| >= threshold, computed on the log scale
    for skewed money columns). Uses the un-winsorised data so genuine
    extremes remain visible.
    """
    threshold = threshold or config.ROBUST_Z_THRESHOLD
    out = df.copy()
    out["dbscan_noise"] = np.asarray(dbscan_labels) == -1

    reasons = [[] for _ in range(len(out))]
    counts = np.zeros(len(out), dtype=int)
    for col in FEATURES:
        if col not in out.columns:
            continue
        values = np.log1p(out[col]) if col in SKEWED else out[col]
        z = robust_z(values).to_numpy()
        extreme = np.abs(z) >= threshold
        counts += extreme.astype(int)
        for i in np.flatnonzero(extreme):
            direction = "high" if z[i] > 0 else "low"
            reasons[i].append(f"{col} {direction} (z={z[i]:.1f})")

    out["n_extreme_features"] = counts
    out["extreme_features"] = ["; ".join(r) for r in reasons]
    feature_flag = counts >= 1
    out["is_anomaly"] = out["dbscan_noise"] | feature_flag
    out["anomaly_type"] = np.select(
        [out["dbscan_noise"] & feature_flag, out["dbscan_noise"],
         feature_flag],
        ["Density outlier + extreme feature", "Density outlier (DBSCAN noise)",
         "Extreme feature value"], default="Normal")
    return out


def anomaly_summary(df):
    """Counts and default-rate comparison for flagged versus normal loans."""
    flagged = df[df["is_anomaly"]]
    normal = df[~df["is_anomaly"]]

    def rate(frame):
        known = frame["default"].dropna() if "default" in frame else []
        return float(np.mean(known)) if len(known) else np.nan

    by_type = (df["anomaly_type"].value_counts().rename_axis("type")
               .reset_index(name="count"))
    return {
        "n_total": int(len(df)),
        "n_anomalies": int(len(flagged)),
        "pct_anomalies": 100.0 * len(flagged) / max(len(df), 1),
        "n_dbscan_noise": int(df["dbscan_noise"].sum()),
        "n_feature_extreme": int((df["n_extreme_features"] >= 1).sum()),
        "default_rate_anomalies": rate(flagged),
        "default_rate_normal": rate(normal),
        "by_type": by_type,
    }
