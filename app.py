"""Streamlit dashboard: Credit Portfolio Segmentation and Risk Patterns.

Run with:  streamlit run app.py
The dashboard reads the files written by ``python run_pipeline.py``.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import clustering, profiling, visuals
from src.insights import ETHICS_NOTE

st.set_page_config(page_title="Credit Portfolio Segmentation",
                   page_icon=":bar_chart:", layout="wide")

TABLES = config.OUTPUT_DIR / "tables"
DATA = config.OUTPUT_DIR / "data"
FIGURES = config.OUTPUT_DIR / "figures"
ALGO_LABELS = config.ALGORITHMS
PAGES = ["Overview", "Data & EDA", "Dimensionality Reduction",
         "Clustering Comparison", "Cluster Profiles", "Default Patterns",
         "Anomaly Detection", "Geographic & Trends", "Business Insights",
         "Segment Explorer (k)"]


# ------------------------------------------------------------- loading
@st.cache_data
def load_csv(path):
    """Read a CSV produced by the pipeline."""
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    """Read a JSON file produced by the pipeline."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def show(fig):
    """Render a matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def show_image(path, caption=None):
    """Display a saved figure if it exists."""
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not available: {path.name} (re-run the pipeline).")


def check_outputs():
    """Stop with a friendly message when the pipeline has not been run."""
    needed = [
        DATA / "clustered_borrowers.csv",
        TABLES / "run_metadata.json",
        TABLES / "cleaning_report.json",
        TABLES / "segment_names.json",
        TABLES / "algorithm_comparison.csv",
        TABLES / "cluster_profiles.csv",
    ]
    missing = [path for path in needed if not path.exists()]
    if missing:
        st.error("Pipeline outputs are missing. Run the analysis before opening the dashboard.")
        st.markdown(
            "**If you do not have the original Lending Club CSV yet, run this "
            "demo command in the VS Code terminal:**\n\n"
            "```bash\npython run_pipeline.py --sample --skip-umap --skip-tsne\n```\n\n"
            "This creates clearly labelled synthetic demo data and generates "
            "the dashboard outputs. It is not real borrower data.\n\n"
            "If you have the real dataset, place it at "
            "`data/raw/loans_granting_model_dataset.csv` and run "
            "`python run_pipeline.py`. Then refresh this page."
        )
        with st.expander("Missing output files"):
            for path in missing:
                st.code(str(path.relative_to(config.BASE_DIR)))
        st.stop()


def pct(value):
    """Format a fraction as a percentage string."""
    return "n/a" if value is None or pd.isna(value) else f"{100 * value:.1f}%"


# ------------------------------------------------------------- start up
check_outputs()
frame = load_csv(DATA / "clustered_borrowers.csv")
meta = load_json(TABLES / "run_metadata.json")
cleaning = load_json(TABLES / "cleaning_report.json")
segment_names = load_json(TABLES / "segment_names.json")
comparison = load_csv(TABLES / "algorithm_comparison.csv")
has_default = meta.get("has_default", False)

st.sidebar.title("Credit Portfolio Segmentation")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
algo = st.sidebar.selectbox(
    "Clustering algorithm", list(ALGO_LABELS),
    format_func=lambda key: ALGO_LABELS[key],
    index=list(ALGO_LABELS).index(meta.get("primary_algorithm", "kmeans")))
label_col = f"{algo}_cluster"
names = {int(k): v for k, v in segment_names[algo].items()}
all_clusters = sorted(int(c) for c in frame[label_col].unique())
chosen = st.sidebar.multiselect(
    "Show clusters", all_clusters, default=all_clusters,
    format_func=lambda c: names.get(c, f"Cluster {c}"))
if not chosen:
    chosen = all_clusters
view = frame[frame[label_col].isin(chosen)]
st.sidebar.caption(f"{len(view):,} of {len(frame):,} sampled borrowers shown")
st.sidebar.caption("Segments describe groups, not individuals.")

if meta.get("data_is_synthetic", False) or \
        "synthetic" in str(meta.get("data_file", "")).lower():
    st.warning("Demo mode: these results come from SYNTHETIC data, not the "
               "real Lending Club dataset.")

profiles_all = load_csv(TABLES / "cluster_profiles.csv")
profiles = profiles_all[profiles_all["algorithm"] == algo]
defaults_all = (load_csv(TABLES / "default_rates_by_cluster.csv")
                if has_default else pd.DataFrame())
defaults = (defaults_all[defaults_all["algorithm"] == algo]
            if has_default else pd.DataFrame())


# ------------------------------------------------------------- pages
def page_overview():
    st.title("Credit Portfolio Segmentation and Risk Pattern Analysis")
    st.markdown(
        "Unsupervised learning (K-Means, Hierarchical, DBSCAN) groups "
        "borrowers with similar financial and credit characteristics. The "
        "**Default** column is never used to build clusters; it is used "
        "afterwards to compare historical repayment outcomes.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw rows", f"{cleaning['rows_raw']:,}")
    c2.metric("Clean rows", f"{cleaning['rows_final']:,}")
    c3.metric("Clustered sample", f"{meta['rows_clustered']:,}")
    rate = cleaning.get("overall_default_rate")
    c4.metric("Portfolio default rate", pct(rate))
    best = comparison.loc[comparison["algorithm"] == ALGO_LABELS[algo]]
    sil = best["silhouette"].iloc[0] if len(best) else np.nan
    c5.metric("Silhouette (selected)", "n/a" if pd.isna(sil)
              else f"{sil:.3f}")
    st.subheader("Segments at a glance")
    cols = [c for c in ("cluster", "segment", "size", "pct_of_portfolio",
                        "median_annual_inc", "median_fico", "median_dti",
                        "median_loan_amnt", "default_rate")
            if c in profiles.columns]
    st.dataframe(profiles[cols].round(3))
    insights = load_json(TABLES / "insights.json")
    st.subheader("Key findings")
    for item in insights["insights"][:4]:
        st.markdown(f"**{item['title']}** - {item['detail']}")
    st.info(ETHICS_NOTE)


def page_eda():
    st.title("Data quality and exploratory analysis")
    tab_q, tab_d, tab_r, tab_i = st.tabs(
        ["Data quality", "Distributions", "Relationships to default",
         "Interactive explorer"])
    with tab_q:
        st.markdown(
            f"- Duplicate rows removed: **{cleaning['duplicate_rows_removed']:,}**\n"
            f"- Duplicate IDs removed: **{cleaning['duplicate_ids_removed']:,}**\n"
            f"- Impossible values set to missing: "
            f"`{cleaning['invalid_values_set_missing']}`\n"
            f"- Imputation: `{cleaning['imputation_strategy']}`\n"
            f"- Skewed features log-transformed: `{meta['log_transformed']}`")
        show_image(FIGURES / "eda" / "missing_values.png")
        st.subheader("Outliers (IQR rule)")
        st.dataframe(load_csv(TABLES / "outlier_summary.csv").round(2))
    with tab_d:
        show_image(FIGURES / "eda" / "distributions.png")
        show_image(FIGURES / "eda" / "boxplots.png")
        st.subheader("Descriptive statistics")
        st.dataframe(load_csv(TABLES / "eda_describe.csv"))
    with tab_r:
        show_image(FIGURES / "eda" / "correlation.png")
        for name in ("default_by_fico", "default_by_dti",
                     "default_by_purpose", "default_by_home"):
            show_image(FIGURES / "eda" / f"{name}.png")
    with tab_i:
        numeric = [c for c in ("loan_amnt", "annual_inc", "dti", "fico",
                               "emp_length", "loan_to_income")
                   if c in view.columns]
        feature = st.selectbox("Feature", numeric)
        by_cluster = st.checkbox("Split by cluster", value=True)
        log_axis = st.checkbox("Log x-axis (for skewed features)")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if by_cluster:
            colours = visuals.cluster_colours(view[label_col])
            for c in sorted(colours):
                part = view[view[label_col] == c][feature]
                ax.hist(part, bins=40, alpha=0.5, color=colours[c],
                        label=names.get(c, f"Cluster {c}"))
            ax.legend(fontsize=8)
        else:
            ax.hist(view[feature], bins=40, color=visuals.PALETTE[0])
        if log_axis and (view[feature] > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel(feature)
        ax.set_ylabel("Borrowers")
        ax.set_title(f"Distribution of {feature}")
        show(fig)


def page_dimred():
    st.title("Dimensionality reduction")
    pca = meta["pca"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Components kept", f"{pca['n_components']} of "
              f"{pca['n_features']}")
    c2.metric("Variance retained", pct(pca["variance_retained"]))
    c3.metric("PC1 + PC2 variance", pct(pca["variance_pc12"]))
    table = load_csv(TABLES / "pca_variance.csv")
    show(visuals.plot_pca_variance(table, pca["target"]))
    with st.expander("PCA loadings (which features drive PC1-PC3)"):
        st.dataframe(load_csv(TABLES / "pca_loadings.csv").round(3))
    options = {"PCA": ("pca_1", "pca_2"), "t-SNE": ("tsne_1", "tsne_2"),
               "UMAP": ("umap_1", "umap_2")}
    available = [k for k, (x, _) in options.items() if x in view.columns]
    if "UMAP" not in available:
        st.caption("UMAP was not computed (umap-learn not installed when "
                   "the pipeline ran). Install it and re-run the pipeline.")
    projection = st.radio("Projection", available, horizontal=True)
    colour_by = st.radio("Colour by", ["Cluster", "Historical default"],
                         horizontal=True)
    x, y = options[projection]
    if colour_by == "Cluster":
        show(visuals.plot_embedding(
            view, x, y, label_col,
            f"{projection} - {ALGO_LABELS[algo]}", names=names))
    elif has_default:
        data = view.dropna(subset=[x, y])
        fig, ax = plt.subplots(figsize=(8, 6))
        for value, text, colour in ((0.0, "Repaid", visuals.PALETTE[0]),
                                    (1.0, "Defaulted", visuals.PALETTE[3])):
            part = data[data["default"] == value]
            ax.scatter(part[x], part[y], s=8, alpha=0.5, color=colour,
                       label=text, linewidths=0)
        unknown = data[data["default"].isna()]
        ax.scatter(unknown[x], unknown[y], s=6, alpha=0.2, color="#9a9a9a",
                   label="Outcome unknown", linewidths=0)
        ax.set_xlabel(x.upper())
        ax.set_ylabel(y.upper())
        ax.set_title(f"{projection} coloured by historical default "
                     "(not used for clustering)")
        ax.legend(markerscale=2)
        show(fig)
    st.caption("t-SNE and UMAP distances between groups are approximate; "
               "use them to explore structure, not to measure it.")


def page_clustering():
    st.title("Clustering algorithm comparison")
    show_table = comparison.drop(columns=["key"]).round(3)
    st.dataframe(show_table)
    show(visuals.plot_algorithm_comparison(comparison))
    st.markdown(
        "**Reading the table:** silhouette near 1 means tight, well "
        "separated clusters; Davies-Bouldin is better when lower; "
        "Calinski-Harabasz is better when higher. DBSCAN scores exclude "
        "noise points, which are reported in `noise_pct`.")
    t1, t2, t3, t4 = st.tabs(["K-Means", "Hierarchical", "DBSCAN",
                              "Agreement"])
    with t1:
        search = load_csv(TABLES / "kmeans_search.csv")
        show(visuals.plot_k_search(search))
        st.dataframe(search.round(3))
        st.caption(f"Selected k = {meta['kmeans_k']} (best silhouette with "
                   "k >= 3).")
    with t2:
        show_image(FIGURES / "clustering" / "dendrogram.png")
        st.dataframe(load_csv(
            TABLES / "hierarchical_linkage_comparison.csv").round(3))
    with t3:
        db = meta["dbscan"]
        st.markdown(f"Selected eps = **{db['eps']:.3f}**, min_samples = "
                    f"**{int(db['min_samples'])}**, clusters = "
                    f"**{int(db['n_clusters'])}**, noise = "
                    f"**{db['noise_pct']:.1f}%**")
        st.dataframe(load_csv(TABLES / "dbscan_search.csv").round(3))
    with t4:
        matrix = load_csv(TABLES / "algorithm_agreement.csv").set_index(
            "algorithm")
        show(visuals.plot_agreement(matrix))


def page_profiles():
    st.title(f"Cluster profiles - {ALGO_LABELS[algo]}")
    prof = profiles[profiles["cluster"].isin(chosen)]
    show(visuals.plot_cluster_sizes(prof, names))
    st.dataframe(prof.drop(columns=["algorithm"]).round(3))
    z = profiling.standardised_profile(view, label_col)
    show(visuals.plot_profile_heatmap(z, names))
    st.subheader("Inspect one cluster")
    cluster = st.selectbox("Cluster", chosen,
                           format_func=lambda c: names.get(c, f"Cluster {c}"))
    part = view[view[label_col] == cluster]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        part["purpose"].value_counts(normalize=True).head(8).sort_values() \
            .mul(100).plot.barh(ax=ax, color=visuals.PALETTE[0])
        ax.set_xlabel("% of cluster")
        ax.set_title("Loan purpose")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        part["home_ownership"].value_counts(normalize=True).sort_values() \
            .mul(100).plot.barh(ax=ax, color=visuals.PALETTE[1])
        ax.set_xlabel("% of cluster")
        ax.set_title("Home ownership")
        show(fig)
    feature = st.selectbox("Compare a feature across clusters", [
        c for c in ("annual_inc", "loan_amnt", "dti", "fico", "emp_length",
                    "loan_to_income") if c in view.columns])
    fig, ax = plt.subplots(figsize=(9, 4))
    groups = [view[view[label_col] == c][feature].dropna() for c in chosen]
    ax.boxplot(groups, showfliers=False)
    ax.set_xticklabels([f"C{c}" if c != -1 else "Noise" for c in chosen])
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by cluster (outliers hidden)")
    show(fig)


def page_default():
    st.title("Historical default patterns by segment")
    if not has_default:
        st.warning("No usable default column was found in the dataset.")
        return
    table = defaults[defaults["cluster"].isin(chosen)]
    show(visuals.plot_default_rates(
        table, f"Default rate by segment - {ALGO_LABELS[algo]}"))
    cols = ["cluster", "segment", "n_with_outcome", "n_defaults",
            "default_rate", "ci_low", "ci_high", "lift_vs_portfolio",
            "risk_tier"]
    st.dataframe(table[cols].round(3))
    chi = load_json(TABLES / "chi_square_results.json")[algo]
    if chi.get("p_value") is not None:
        verdict = ("significant" if chi["p_value"] < 0.05
                   else "not significant")
        st.markdown(
            f"**Chi-square test** (cluster vs default): "
            f"chi2 = {chi['chi2']:.1f}, p = {chi['p_value']:.3g}, "
            f"Cramer's V = {chi['cramers_v']:.2f} -> difference is "
            f"**{verdict}**.")
    st.caption("Default was excluded from clustering; intervals are 95% "
               "Wilson intervals. Loans still in progress have no outcome "
               "and are excluded from the rates.")
    st.subheader("Drill down: default rate by cluster and a second factor")
    factor = st.selectbox("Second factor", [
        c for c in ("purpose", "home_ownership", "issue_year", "state")
        if c in view.columns])
    known = view[view["default"].notna()]
    pivot = known.pivot_table(index=label_col, columns=factor,
                              values="default", aggfunc="mean")
    counts = known.pivot_table(index=label_col, columns=factor,
                               values="default", aggfunc="count")
    pivot = pivot.where(counts >= 20)
    pivot = pivot.loc[:, pivot.notna().sum() > 0]
    pivot.index = [names.get(int(i), f"Cluster {i}") for i in pivot.index]
    if pivot.empty:
        st.info("Not enough loans with known outcomes for this view.")
        return
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(pivot) + 2.5))
    import seaborn as sns
    sns.heatmap(pivot * 100, annot=True, fmt=".0f", cmap="Reds", ax=ax,
                cbar_kws={"label": "Default rate (%)"})
    ax.set_title(f"Default rate (%) by cluster and {factor} "
                 "(cells with fewer than 20 loans hidden)")
    show(fig)


def page_anomaly():
    st.title("Anomaly detection")
    flagged = view[view["is_anomaly"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged profiles", f"{len(flagged):,}",
              f"{100 * len(flagged) / max(len(view), 1):.1f}% of view")
    c2.metric("DBSCAN noise", f"{int(view['dbscan_noise'].sum()):,}")
    c3.metric("Extreme feature value",
              f"{int((view['n_extreme_features'] >= 1).sum()):,}")
    proj = "tsne" if "tsne_1" in view.columns else "pca"
    show(visuals.plot_anomalies(view, f"{proj}_1", f"{proj}_2",
                                f"Anomalies in {proj.upper()} space"))
    st.markdown(
        "A profile is flagged if DBSCAN treats it as noise (low-density "
        "region) **or** a feature is extreme (robust z-score above "
        f"{config.ROBUST_Z_THRESHOLD:g}). Flags are prompts for review, "
        "not verdicts about a borrower.")
    kinds = ["All"] + sorted(flagged["anomaly_type"].unique())
    kind = st.selectbox("Anomaly type", kinds)
    shown = flagged if kind == "All" else flagged[
        flagged["anomaly_type"] == kind]
    cols = [c for c in ("loan_amnt", "annual_inc", "dti", "fico",
                        "emp_length", "purpose", "home_ownership", "state",
                        "default", "anomaly_type", "extreme_features")
            if c in shown.columns]
    st.dataframe(shown[cols].head(500))
    st.caption("Showing at most 500 rows. Download the full list below.")
    anomalies_csv = TABLES / "anomalies.csv"
    if anomalies_csv.exists():
        st.download_button("Download anomalies.csv",
                           anomalies_csv.read_bytes(),
                           file_name="anomalies.csv", mime="text/csv")
    if has_default:
        known = view[view["default"].notna()]
        compare = known.groupby("is_anomaly")["default"].agg(
            ["mean", "count"]).rename(index={False: "Normal", True: "Flagged"})
        compare["mean"] = (compare["mean"] * 100).round(2)
        compare.columns = ["default_rate_%", "loans_with_outcome"]
        st.subheader("Default rate: flagged vs normal")
        st.dataframe(compare)


def page_geo_trend():
    st.title("Geographic analysis and portfolio trends")
    states = load_csv(TABLES / "portfolio_by_state.csv")
    trend = load_csv(TABLES / "portfolio_trend_by_year.csv")
    st.subheader("Geographic portfolio analysis")
    if len(states):
        if len(states) > 5:
            top_n = st.slider("Top states by volume", 5,
                              min(30, len(states)), min(15, len(states)))
        else:
            top_n = len(states)
        show(visuals.plot_state_bars(states, top_n))
        picks = st.multiselect("Compare states", list(states["state"]),
                               default=list(states["state"].head(3)))
        st.dataframe(states[states["state"].isin(picks)].round(3))
        mix = profiling.cluster_state_mix(view, label_col)
        if len(mix):
            show(visuals.plot_mix_heatmap(
                mix, "State mix of each segment (% of cluster)", names))
    else:
        st.info("No state column in the dataset.")
    st.subheader("Portfolio trend monitoring")
    if len(trend):
        show(visuals.plot_trend(trend))
        st.dataframe(trend.round(3))
        mix_year = profiling.segment_mix_by_year(view, label_col)
        if len(mix_year):
            mix_year.columns = [names.get(int(c), f"Cluster {c}")
                                for c in mix_year.columns]
            st.markdown("**Segment mix by issue year (% of loans)**")
            st.line_chart(mix_year)
        st.caption("Recent vintages have had less time to default, so their "
                   "observed default rates are understated.")
    else:
        st.info("No usable issue-date column in the dataset.")


def page_insights():
    st.title("Business insights and recommendations")
    data = load_json(TABLES / "insights.json")
    if not data["insights"]:
        st.warning("Insights need a default column.")
    for item in data["insights"]:
        st.markdown(f"**{item['title']}**")
        st.write(item["detail"])
    st.subheader("Segment recommendations (primary K-Means segmentation)")
    if data["recommendations"]:
        rec = pd.DataFrame(data["recommendations"])
        rec["default_rate"] = rec["default_rate"].round(3)
        rec["share_of_portfolio_pct"] = rec[
            "share_of_portfolio_pct"].round(1)
        st.dataframe(rec)
    st.subheader("Business use cases covered")
    st.markdown(
        "- **Borrower segmentation / customer profiling** - Cluster Profiles\n"
        "- **Default pattern analysis** and **loan portfolio risk "
        "profiling** - Default Patterns\n"
        "- **Anomaly detection** - Anomaly Detection\n"
        "- **Personalised loan product design** - recommendations above\n"
        "- **Geographic portfolio analysis** and **portfolio trend "
        "monitoring** - Geographic & Trends")
    st.subheader("Limitations")
    st.markdown(
        "- Clustering ran on a random sample "
        f"({meta['rows_clustered']:,} rows) for tractability.\n"
        "- Loans without a final outcome are excluded from default rates.\n"
        "- Segments are data-driven groupings, not causal explanations.\n"
        "- Historical default rates may not hold for future borrowers.")
    st.info(ETHICS_NOTE)


@st.cache_data
def kmeans_for_k(k):
    """K-Means labels on the saved scaled features (cached per k)."""
    features = load_csv(DATA / "features_scaled.csv").set_index("row_id")
    labels, _ = clustering.fit_kmeans(features.to_numpy(), k)
    metrics = clustering.evaluate_labels(features.to_numpy(), labels)
    return pd.Series(labels, index=features.index), metrics


def page_explorer():
    st.title("Segment explorer: try a different number of clusters")
    k = st.slider("Number of K-Means clusters (k)", 2, 10,
                  int(meta["kmeans_k"]))
    labels, metrics = kmeans_for_k(k)
    data = frame.set_index("row_id").join(labels.rename("explore_cluster"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette", f"{metrics['silhouette']:.3f}")
    c2.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}")
    c3.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.0f}")
    explore_names = profiling.name_segments(data, "explore_cluster")
    if has_default:
        table = profiling.default_rate_table(data, "explore_cluster",
                                             explore_names)
        show(visuals.plot_default_rates(
            table, f"Default rate by segment (k = {k})"))
        st.dataframe(table[["cluster", "segment", "n_with_outcome",
                            "default_rate", "lift_vs_portfolio",
                            "risk_tier"]].round(3))
    if "pca_1" in data.columns:
        show(visuals.plot_embedding(data.reset_index(), "pca_1", "pca_2",
                                    "explore_cluster",
                                    f"PCA view of K-Means with k = {k}",
                                    names=explore_names))


PAGE_FUNCTIONS = {
    "Overview": page_overview, "Data & EDA": page_eda,
    "Dimensionality Reduction": page_dimred,
    "Clustering Comparison": page_clustering,
    "Cluster Profiles": page_profiles, "Default Patterns": page_default,
    "Anomaly Detection": page_anomaly,
    "Geographic & Trends": page_geo_trend,
    "Business Insights": page_insights,
    "Segment Explorer (k)": page_explorer,
}
PAGE_FUNCTIONS[page]()
