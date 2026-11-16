"""Exploratory data analysis figures and summary statistics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import config  # noqa: E402
from src.visuals import PALETTE, save_figure  # noqa: E402

LABELS = {"loan_amnt": "Loan amount ($)", "annual_inc": "Annual income ($)",
          "dti": "Debt-to-income ratio (%)", "fico": "FICO score",
          "emp_length": "Employment length (years)",
          "loan_to_income": "Loan-to-income ratio"}


def _numeric(df):
    return [c for c in LABELS if c in df.columns]


def fig_missing(missing_before):
    """Bar chart of missing values found before imputation."""
    data = pd.Series(missing_before).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(data.index, data.values, color=PALETTE[2])
    ax.set_ylabel("Missing / invalid entries")
    ax.set_title("Missing values before imputation")
    ax.tick_params(axis="x", rotation=30)
    return fig


def fig_distributions(df):
    """Histograms of the numeric variables (shows skew and outliers)."""
    cols = _numeric(df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], bins=40, ax=ax, color=PALETTE[0])
        ax.set_title(f"{LABELS[col]} (skew {df[col].skew():.2f})")
        ax.set_xlabel(LABELS[col])
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distributions of numeric features", y=1.02)
    fig.tight_layout()
    return fig


def fig_boxplots(df):
    """Boxplots to inspect outliers per numeric feature."""
    cols = _numeric(df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(x=df[col], ax=ax, color=PALETTE[1])
        ax.set_xlabel(LABELS[col])
        ax.set_title(LABELS[col])
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Outlier inspection (boxplots)", y=1.02)
    fig.tight_layout()
    return fig


def fig_correlation(df):
    """Correlation heatmap of numeric features (and default if known)."""
    cols = _numeric(df) + (["default"] if "default" in df.columns else [])
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix (default shown for context only)")
    return fig


def fig_category_counts(df, column, title, top_n=12):
    """Loan counts by a categorical variable."""
    counts = df[column].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(counts) + 2))
    ax.barh(counts.index, counts.values, color=PALETTE[0])
    ax.set_xlabel("Loans")
    ax.set_title(title)
    return fig


def fig_default_by_category(df, column, title, min_count=30):
    """Default rate by category with volume filter."""
    known = df[df["default"].notna()]
    table = known.groupby(column)["default"].agg(["mean", "count"])
    table = table[table["count"] >= min_count].sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(table) + 2))
    ax.barh(table.index, table["mean"] * 100, color=PALETTE[3])
    overall = known["default"].mean() * 100
    ax.axvline(overall, ls="--", color="black", label=f"Portfolio {overall:.1f}%")
    ax.set_xlabel("Default rate (%)")
    ax.set_title(title)
    ax.legend()
    return fig


def fig_default_by_fico_band(df):
    """Default rate by FICO band (a basic sanity check on the data)."""
    known = df[df["default"].notna()].copy()
    bins = [300, 620, 660, 700, 740, 780, 851]
    names = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
    known["fico_band"] = pd.cut(known["fico"], bins=bins, labels=names,
                                right=False)
    table = known.groupby("fico_band", observed=True)["default"].agg(
        ["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table.index.astype(str), table["mean"] * 100, color=PALETTE[3])
    ax.set_xlabel("FICO band")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate by FICO band")
    return fig


def fig_default_by_dti_band(df):
    """Default rate by debt-to-income band."""
    if "dti" not in df.columns:
        return None
    known = df[df["default"].notna()].copy()
    known["dti_band"] = pd.cut(known["dti"], bins=[0, 10, 15, 20, 25, 30, 101],
                               labels=["<10", "10-15", "15-20", "20-25",
                                       "25-30", "30+"], include_lowest=True)
    table = known.groupby("dti_band", observed=True)["default"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table.index.astype(str), table.values * 100, color=PALETTE[3])
    ax.set_xlabel("DTI band (%)")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate by debt-to-income band")
    return fig


def run_eda(df, cleaning_report, figure_dir=None):
    """Create all EDA figures and return ``(paths, summary_dict)``."""
    figure_dir = figure_dir or config.FIGURE_DIR / "eda"
    figures = {
        "missing_values": fig_missing(
            cleaning_report["missing_before_imputation"]),
        "distributions": fig_distributions(df),
        "boxplots": fig_boxplots(df),
        "correlation": fig_correlation(df),
    }
    if "purpose" in df.columns:
        figures["purpose_counts"] = fig_category_counts(
            df, "purpose", "Loans by purpose")
        if df["default"].notna().any():
            figures["default_by_purpose"] = fig_default_by_category(
                df, "purpose", "Default rate by loan purpose")
    if "home_ownership" in df.columns and df["default"].notna().any():
        figures["default_by_home"] = fig_default_by_category(
            df, "home_ownership", "Default rate by home ownership")
    if df["default"].notna().any():
        figures["default_by_fico"] = fig_default_by_fico_band(df)
        dti_fig = fig_default_by_dti_band(df)
        if dti_fig is not None:
            figures["default_by_dti"] = dti_fig

    paths = {}
    for name, fig in figures.items():
        paths[name] = save_figure(fig, figure_dir / f"{name}.png")

    numeric = _numeric(df)
    summary = {
        "describe": df[numeric].describe().T.round(2).reset_index().rename(
            columns={"index": "feature"}),
        "skewness": {c: round(float(df[c].skew()), 2) for c in numeric},
        "strongest_correlations": _top_correlations(df, numeric),
    }
    return paths, summary


def _top_correlations(df, columns, top_n=5):
    """Strongest pairwise correlations between numeric features."""
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pairs.append((a, b, float(df[a].corr(df[b]))))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return [{"feature_1": a, "feature_2": b, "correlation": round(r, 3)}
            for a, b, r in pairs[:top_n]]
