"""Feature engineering and scaling for clustering.

* ``default`` and ``loan_id`` are never part of the feature matrix.
* ``state`` and the issue date are kept out of the clustering features and
  used afterwards for geographic and trend analysis (documented choice).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class FeatureSet:
    """Result of feature engineering."""

    X: pd.DataFrame
    numeric_features: list
    categorical_features: list
    log_transformed: list = field(default_factory=list)
    rare_grouped: dict = field(default_factory=dict)
    scaler: StandardScaler = None


def add_loan_to_income(df):
    """Add loan-to-income ratio (loan amount / annual income)."""
    out = df.copy()
    if {"loan_amnt", "annual_inc"} <= set(out.columns):
        out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"]
    return out


def group_rare_categories(series, threshold):
    """Group categories below ``threshold`` share into 'other'."""
    share = series.value_counts(normalize=True)
    rare = share[share < threshold].index.tolist()
    return series.where(~series.isin(rare), "other"), rare


def build_features(df_model):
    """Create the scaled feature matrix used by all clustering algorithms.

    Steps: loan-to-income ratio -> log1p on skewed columns -> standard
    scaling of numerics -> one-hot encoding of categoricals (rare levels
    grouped). One-hot columns stay 0/1 so rare levels are not inflated.
    """
    work = add_loan_to_income(df_model)
    numeric = [c for c in config.NUMERIC_FEATURES + ["loan_to_income"]
               if c in work.columns]
    num = work[numeric].astype(float).copy()

    log_cols = [c for c in numeric
                if num[c].min() >= 0 and abs(num[c].skew())
                > config.SKEW_THRESHOLD]
    for col in log_cols:
        num[col] = np.log1p(num[col])

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(num), columns=numeric,
                          index=work.index)

    cat_frames, rare_info, cats_used = [], {}, []
    for col in config.CATEGORICAL_FEATURES:
        if col not in work.columns:
            continue
        grouped, rare = group_rare_categories(
            work[col].astype(str), config.RARE_CATEGORY_THRESHOLD)
        rare_info[col] = rare
        cats_used.append(col)
        cat_frames.append(pd.get_dummies(grouped, prefix=col, dtype=float))

    X = pd.concat([scaled] + cat_frames, axis=1)
    return FeatureSet(X=X, numeric_features=numeric,
                      categorical_features=cats_used,
                      log_transformed=log_cols, rare_grouped=rare_info,
                      scaler=scaler)
