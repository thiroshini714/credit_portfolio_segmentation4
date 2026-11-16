"""Turn analysis tables into plain-language business insights.

All statements are about SEGMENTS, not individual borrowers (ethical-use
guideline in the project brief).
"""

import numpy as np

ETHICS_NOTE = (
    "Segments describe groups with similar financial characteristics. "
    "Cluster membership must not be used on its own to approve, decline or "
    "price an individual loan, and historical default rates do not guarantee "
    "future behaviour."
)

RECOMMENDATIONS = {
    "Higher risk": (
        "Review underwriting for this segment: consider tighter limits or "
        "risk-based pricing, closer early-warning monitoring, and "
        "repayment-support outreach."),
    "Lower risk": (
        "Candidate for preferential pricing, pre-approved offers and "
        "cross-sell of larger or longer-term products, subject to normal "
        "credit policy."),
    "In line with portfolio": (
        "Keep standard credit policy; monitor the segment for drift and "
        "test product tweaks (e.g. term or amount options)."),
}


def _fmt_pct(value):
    return "n/a" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def segment_recommendations(default_table, profile_table):
    """One recommendation row per cluster (noise cluster excluded)."""
    merged = default_table.merge(
        profile_table[["cluster", "size", "pct_of_portfolio"]
                      + [c for c in ("top_purpose", "top_home_ownership")
                         if c in profile_table.columns]],
        on="cluster", how="left")
    rows = []
    for rec in merged.itertuples():
        if rec.cluster == -1:
            continue
        extra = []
        if hasattr(rec, "top_purpose"):
            extra.append(f"main purpose: {rec.top_purpose}")
        if hasattr(rec, "top_home_ownership"):
            extra.append(f"typical housing: {rec.top_home_ownership}")
        rows.append({
            "cluster": rec.cluster, "segment": rec.segment,
            "risk_tier": rec.risk_tier,
            "default_rate": rec.default_rate,
            "share_of_portfolio_pct": rec.pct_of_portfolio,
            "context": "; ".join(extra),
            "recommendation": RECOMMENDATIONS[rec.risk_tier]})
    return rows


def generate_insights(default_table, chi2, comparison, pca_info,
                      anomaly, state_table, trend_table, k_selected):
    """Return a list of ``{"title", "detail"}`` insight dictionaries."""
    insights = []
    valid = default_table[(default_table["cluster"] != -1)
                          & default_table["default_rate"].notna()]
    if len(valid) >= 2:
        high = valid.loc[valid["default_rate"].idxmax()]
        low = valid.loc[valid["default_rate"].idxmin()]
        spread = (high["default_rate"] / low["default_rate"]
                  if low["default_rate"] else np.nan)
        insights.append({
            "title": "Default rates differ across segments",
            "detail": (
                f"{high['segment']} has the highest historical default rate "
                f"({_fmt_pct(high['default_rate'])}) and {low['segment']} "
                f"the lowest ({_fmt_pct(low['default_rate'])}), a "
                f"{spread:.1f}x gap. Portfolio average: "
                f"{_fmt_pct(high['portfolio_default_rate'])}.")})
    if chi2 and not np.isnan(chi2.get("p_value", np.nan)):
        verdict = ("statistically significant" if chi2["p_value"] < 0.05
                   else "not statistically significant")
        insights.append({
            "title": "Is the default-rate difference real?",
            "detail": (
                f"A chi-square test gives p = {chi2['p_value']:.3g} "
                f"(Cramer's V = {chi2['cramers_v']:.2f}), so the difference "
                f"between segments is {verdict}. Default was NOT used to "
                f"build the clusters, so this is an independent check that "
                f"the segments carry risk information.")})
    if comparison is not None and len(comparison):
        ranked = comparison.dropna(subset=["silhouette"]).sort_values(
            "silhouette", ascending=False)
        if len(ranked):
            best = ranked.iloc[0]
            insights.append({
                "title": "Algorithm comparison",
                "detail": (
                    f"{best['algorithm']} has the highest silhouette score "
                    f"({best['silhouette']:.3f}). K-Means was selected as "
                    f"the primary segmentation (k = {k_selected}) because it "
                    f"gives a full, easy-to-explain partition; see the "
                    f"comparison tab for all three algorithms.")})
    if pca_info:
        insights.append({
            "title": "Dimensionality reduction",
            "detail": (
                f"{pca_info['n_components']} principal components retain "
                f"{100 * pca_info['variance_retained']:.1f}% of the variance "
                f"(target {100 * pca_info['target']:.0f}%); the first two "
                f"components explain "
                f"{100 * pca_info['variance_pc12']:.1f}%.")})
    if anomaly:
        detail = (f"{anomaly['n_anomalies']:,} of {anomaly['n_total']:,} "
                  f"profiles ({anomaly['pct_anomalies']:.1f}%) were flagged "
                  f"({anomaly['n_dbscan_noise']:,} DBSCAN noise, "
                  f"{anomaly['n_feature_extreme']:,} with an extreme "
                  f"feature).")
        if not np.isnan(anomaly["default_rate_anomalies"]):
            detail += (f" Their default rate is "
                       f"{_fmt_pct(anomaly['default_rate_anomalies'])} vs "
                       f"{_fmt_pct(anomaly['default_rate_normal'])} for "
                       f"other profiles. Flagged profiles are for manual "
                       f"review, not automatic rejection.")
        insights.append({"title": "Unusual borrower profiles",
                         "detail": detail})
    if state_table is not None and len(state_table):
        big = state_table[state_table["n_with_outcome"] >= 100]
        if len(big) >= 2:
            worst = big.loc[big["default_rate"].idxmax()]
            best = big.loc[big["default_rate"].idxmin()]
            insights.append({
                "title": "Geographic view",
                "detail": (
                    f"Among states with at least 100 resolved loans, "
                    f"{worst['state']} has the highest default rate "
                    f"({_fmt_pct(worst['default_rate'])}) and "
                    f"{best['state']} the lowest "
                    f"({_fmt_pct(best['default_rate'])}). State was not used "
                    f"for clustering.")})
    if trend_table is not None and len(trend_table) >= 2:
        usable = trend_table.dropna(subset=["default_rate"])
        if len(usable) >= 2:
            first, last = usable.iloc[0], usable.iloc[-1]
            insights.append({
                "title": "Portfolio trend",
                "detail": (
                    f"Default rate moves from {_fmt_pct(first['default_rate'])}"
                    f" ({int(first['issue_year'])} vintage) to "
                    f"{_fmt_pct(last['default_rate'])} "
                    f"({int(last['issue_year'])} vintage). Recent vintages "
                    f"are still maturing, so compare like-for-like ages.")})
    return insights
