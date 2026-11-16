"""End-to-end pipeline: raw CSV -> cleaned data -> clusters -> outputs.

Run through ``python run_pipeline.py`` (see README). Every artefact the
Streamlit dashboard needs is written to ``outputs/``.
"""

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import config
from src import (anomaly, clustering, data_loader, dimred, eda, features,
                 insights, preprocessing, profiling, report, visuals)


def _json_default(obj):
    """Make numpy / pandas values JSON serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(data, path):
    """Write ``data`` as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def log(message):
    """Timestamped console message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(data_path=None, max_rows=None, overrides=None, skip_umap=False,
        skip_tsne=False, output_dir=None):
    """Execute the full analysis and return a dictionary of results."""
    started = time.time()
    data_path = Path(data_path or config.DEFAULT_DATA_FILE)
    max_rows = max_rows or config.MAX_ROWS
    out = Path(output_dir or config.OUTPUT_DIR)
    fig_dir, tab_dir = out / "figures", out / "tables"
    data_dir, model_dir, rep_dir = out / "data", out / "models", out / "reports"
    for folder in (fig_dir, tab_dir, data_dir, model_dir, rep_dir):
        folder.mkdir(parents=True, exist_ok=True)
    seed = config.RANDOM_STATE
    np.random.seed(seed)

    # 1. load ------------------------------------------------------------
    log(f"Loading {data_path}")
    raw = data_loader.load_csv(data_path)
    standard, mapping = data_loader.standardise_columns(raw, overrides)
    log(f"Rows: {len(raw):,}; column mapping: {mapping}")

    # 2. clean -----------------------------------------------------------
    log("Cleaning data")
    clean, cleaning_report = preprocessing.clean_data(standard)
    outliers = preprocessing.outlier_summary(clean)
    outliers.to_csv(tab_dir / "outlier_summary.csv", index=False)
    has_default = ("default" in clean.columns
                   and clean["default"].notna().any())
    if not has_default:
        log("WARNING: no usable default column; default analysis skipped")

    # 3. EDA -------------------------------------------------------------
    log("Exploratory data analysis")
    eda_paths, eda_summary = eda.run_eda(
        profiling.ensure_ratio(clean), cleaning_report, fig_dir / "eda")
    eda_summary["describe"].to_csv(tab_dir / "eda_describe.csv", index=False)

    state_full = profiling.state_summary(clean)
    trend_full = profiling.trend_summary(clean)
    state_full.to_csv(tab_dir / "portfolio_by_state.csv", index=False)
    trend_full.to_csv(tab_dir / "portfolio_trend_by_year.csv", index=False)
    if len(state_full):
        visuals.save_figure(visuals.plot_state_bars(state_full),
                            fig_dir / "eda" / "state_overview.png")
    if len(trend_full):
        visuals.save_figure(visuals.plot_trend(trend_full),
                            fig_dir / "eda" / "portfolio_trend.png")

    # 4. working sample + features --------------------------------------
    if len(clean) > max_rows:
        work = clean.sample(n=max_rows, random_state=seed).sort_index()
        log(f"Using a random working sample of {max_rows:,} rows")
    else:
        work = clean.copy()
    model_frame = preprocessing.winsorise(work)
    feature_set = features.build_features(model_frame)
    X = feature_set.X
    log(f"Feature matrix: {X.shape[0]:,} x {X.shape[1]} "
        f"(log-transformed: {feature_set.log_transformed})")

    # 5. dimensionality reduction ---------------------------------------
    log("PCA")
    X_pca, pca_model, pca_table, n_comp = dimred.run_pca(X)
    pca_info = {
        "n_components": n_comp, "target": config.PCA_VARIANCE_TARGET,
        "variance_retained": float(pca_table["cumulative_variance"]
                                   .iloc[n_comp - 1]),
        "variance_pc12": float(pca_table["explained_variance_ratio"]
                               .iloc[:2].sum()),
        "n_features": int(X.shape[1])}
    loadings = dimred.pca_loadings(pca_model, X.columns)
    pca_table.to_csv(tab_dir / "pca_variance.csv", index=False)
    loadings.to_csv(tab_dir / "pca_loadings.csv", index_label="feature")
    visuals.save_figure(
        visuals.plot_pca_variance(pca_table, config.PCA_VARIANCE_TARGET),
        fig_dir / "dimred" / "pca_variance.png")

    embeddings = pd.DataFrame(X_pca[:, :2], columns=["pca_1", "pca_2"],
                              index=X.index)
    if not skip_tsne:
        log("t-SNE (this can take a minute)")
        embeddings = embeddings.join(dimred.run_tsne(X_pca, X.index))
    umap_done = False
    if not skip_umap:
        log("UMAP")
        umap_coords = dimred.run_umap(X_pca, X.index)
        if umap_coords is None:
            log("umap-learn not available; UMAP step skipped")
        else:
            embeddings = embeddings.join(umap_coords)
            umap_done = True

    # 6. clustering ------------------------------------------------------
    log("K-Means search")
    k_table = clustering.kmeans_search(X)
    k_best = clustering.choose_k(k_table)
    km_labels, km_model = clustering.fit_kmeans(X, k_best)
    log(f"K-Means selected k={k_best}")
    k_table.to_csv(tab_dir / "kmeans_search.csv", index=False)
    visuals.save_figure(visuals.plot_k_search(k_table),
                        fig_dir / "clustering" / "kmeans_k_search.png")

    log("Hierarchical clustering")
    hc_labels, _ = clustering.fit_hierarchical(X, k_best)
    linkage_table = clustering.linkage_comparison(X, k_best)
    linkage_table.to_csv(tab_dir / "hierarchical_linkage_comparison.csv",
                         index=False)
    visuals.save_figure(visuals.plot_dendrogram(X, seed=seed),
                        fig_dir / "clustering" / "dendrogram.png")

    log("DBSCAN parameter search")
    db_table, db_best, db_labels = clustering.dbscan_search(X_pca, X)
    db_table.to_csv(tab_dir / "dbscan_search.csv", index=False)
    log(f"DBSCAN selected eps={db_best['eps']:.3f}, "
        f"min_samples={int(db_best['min_samples'])}, "
        f"clusters={int(db_best['n_clusters'])}, "
        f"noise={db_best['noise_pct']:.1f}%")

    label_sets = {"kmeans": km_labels, "hierarchical": hc_labels,
                  "dbscan": db_labels}
    comparison = clustering.compare_algorithms(X, label_sets)
    agreement = clustering.agreement_between(label_sets)
    comparison.to_csv(tab_dir / "algorithm_comparison.csv", index=False)
    agreement.to_csv(tab_dir / "algorithm_agreement.csv",
                     index_label="algorithm")
    visuals.save_figure(visuals.plot_algorithm_comparison(comparison),
                        fig_dir / "clustering" / "algorithm_comparison.png")
    visuals.save_figure(visuals.plot_agreement(agreement),
                        fig_dir / "clustering" / "algorithm_agreement.png")

    # 7. assemble output frame ------------------------------------------
    result = work.copy()
    for key, labels in label_sets.items():
        result[f"{key}_cluster"] = labels
    result = result.join(embeddings)
    result = anomaly.detect_anomalies(result, db_labels)
    anomaly_info = anomaly.anomaly_summary(result)
    anomaly_info["by_type"].to_csv(tab_dir / "anomaly_by_type.csv",
                                   index=False)
    anomaly_cols = [c for c in ("loan_id", "loan_amnt", "annual_inc", "dti",
                                "fico", "emp_length", "purpose",
                                "home_ownership", "state", "default",
                                "anomaly_type", "extreme_features")
                    if c in result.columns]
    result[result["is_anomaly"]][anomaly_cols].to_csv(
        tab_dir / "anomalies.csv", index=False)

    dashboard_frame = result.drop(columns=["loan_id"], errors="ignore")
    dashboard_frame.to_csv(data_dir / "clustered_borrowers.csv",
                           index_label="row_id")
    X.to_csv(data_dir / "features_scaled.csv", index_label="row_id")

    # 8. profiling + default analysis -----------------------------------
    all_profiles, all_defaults, all_names = [], [], {}
    chi_results = {}
    for key in label_sets:
        col = f"{key}_cluster"
        names = profiling.name_segments(result, col)
        all_names[key] = names
        prof = profiling.profile_clusters(result, col)
        prof.insert(0, "algorithm", key)
        prof.insert(2, "segment", prof["cluster"].map(names))
        all_profiles.append(prof)
        if has_default:
            table = profiling.default_rate_table(result, col, names)
            table.insert(0, "algorithm", key)
            all_defaults.append(table)
            chi_results[key] = profiling.chi_square_default(result, col)
    profiles = pd.concat(all_profiles, ignore_index=True)
    profiles.to_csv(tab_dir / "cluster_profiles.csv", index=False)
    defaults = (pd.concat(all_defaults, ignore_index=True)
                if all_defaults else pd.DataFrame())
    defaults.to_csv(tab_dir / "default_rates_by_cluster.csv", index=False)
    write_json(all_names, tab_dir / "segment_names.json")
    write_json(chi_results, tab_dir / "chi_square_results.json")

    primary = config.PRIMARY_ALGORITHM
    p_col = f"{primary}_cluster"
    p_names = all_names[primary]
    p_profile = profiles[profiles["algorithm"] == primary]
    p_default = (defaults[defaults["algorithm"] == primary]
                 if has_default else pd.DataFrame())

    # 9. figures that depend on clusters --------------------------------
    cl_dir = fig_dir / "clustering"
    z_prof = profiling.standardised_profile(result, p_col)
    visuals.save_figure(visuals.plot_profile_heatmap(z_prof, p_names),
                        cl_dir / "kmeans_profile_heatmap.png")
    visuals.save_figure(visuals.plot_cluster_sizes(p_profile, p_names),
                        cl_dir / "kmeans_cluster_sizes.png")
    for proj, (x, y) in {"pca": ("pca_1", "pca_2"),
                         "tsne": ("tsne_1", "tsne_2"),
                         "umap": ("umap_1", "umap_2")}.items():
        if x not in result.columns:
            continue
        for key in label_sets:
            visuals.save_figure(
                visuals.plot_embedding(
                    result, x, y, f"{key}_cluster",
                    f"{proj.upper()} projection - {config.ALGORITHMS[key]}",
                    names=all_names[key]),
                fig_dir / "dimred" / f"{proj}_{key}.png")
    anomaly_proj = "tsne" if "tsne_1" in result.columns else "pca"
    visuals.save_figure(
        visuals.plot_anomalies(result, f"{anomaly_proj}_1",
                               f"{anomaly_proj}_2",
                               f"Anomalies in {anomaly_proj.upper()} space"),
        fig_dir / "anomaly" / "anomalies.png")
    if has_default:
        visuals.save_figure(
            visuals.plot_default_rates(
                p_default, "Historical default rate by K-Means segment"),
            fig_dir / "default" / "default_rate_by_cluster.png")
    mix = profiling.cluster_state_mix(result, p_col)
    if len(mix):
        visuals.save_figure(
            visuals.plot_mix_heatmap(mix, "State mix of each segment (%)",
                                     p_names),
            fig_dir / "geographic" / "segment_state_mix.png")

    # 10. insights -------------------------------------------------------
    insight_list, recommendations = [], []
    if has_default:
        insight_list = insights.generate_insights(
            p_default, chi_results[primary], comparison, pca_info,
            anomaly_info, state_full, trend_full, k_best)
        recommendations = insights.segment_recommendations(
            p_default, p_profile)
    write_json({"insights": insight_list,
                "recommendations": recommendations,
                "ethics_note": insights.ETHICS_NOTE},
               tab_dir / "insights.json")

    # 11. models + metadata ---------------------------------------------
    joblib.dump({"scaler": feature_set.scaler, "pca": pca_model,
                 "kmeans": km_model, "feature_names": list(X.columns)},
                model_dir / "models.joblib")
    metadata = {
        "data_file": str(data_path),
        "data_is_synthetic": "synthetic" in data_path.name.lower(),
        "column_mapping": mapping,
        "random_state": seed, "rows_raw": cleaning_report["rows_raw"],
        "rows_clean": cleaning_report["rows_final"],
        "rows_clustered": int(len(work)),
        "feature_columns": list(X.columns),
        "log_transformed": feature_set.log_transformed,
        "rare_categories_grouped": feature_set.rare_grouped,
        "pca": pca_info, "kmeans_k": k_best,
        "dbscan": {k: db_best[k] for k in ("eps", "min_samples",
                                           "n_clusters", "noise_pct")},
        "umap_computed": umap_done, "tsne_computed": not skip_tsne,
        "has_default": has_default,
        "primary_algorithm": primary,
        "versions": {"python": platform.python_version(),
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
        "runtime_seconds": round(time.time() - started, 1)}
    write_json(metadata, tab_dir / "run_metadata.json")
    write_json(cleaning_report, tab_dir / "cleaning_report.json")

    results = {
        "metadata": metadata, "cleaning": cleaning_report,
        "outliers": outliers, "eda_summary": eda_summary,
        "pca_table": pca_table, "pca_info": pca_info, "k_table": k_table,
        "linkage_table": linkage_table, "dbscan_table": db_table,
        "comparison": comparison, "agreement": agreement,
        "profile": p_profile, "default": p_default,
        "chi2": chi_results.get(primary), "anomaly": anomaly_info,
        "insights": insight_list, "recommendations": recommendations,
        "state": state_full, "trend": trend_full, "names": p_names,
        "result_frame": result, "feature_set": feature_set,
    }
    report.write_report(results, rep_dir / "analysis_report.md")
    log(f"Done in {time.time() - started:.1f}s. Outputs in {out}")
    return results
