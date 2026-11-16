"""K-Means, Hierarchical (Ward) and DBSCAN clustering plus evaluation."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors

import config


# ---------------------------------------------------------------- metrics
def evaluate_labels(X, labels, random_state=config.RANDOM_STATE):
    """Silhouette, Davies-Bouldin and Calinski-Harabasz for a labelling.

    DBSCAN noise points (label -1) are excluded from the scores and
    reported separately as ``noise_pct``.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    keep = labels != -1
    n_clusters = len(set(labels[keep].tolist()))
    result = {"n_clusters": n_clusters,
              "noise_pct": 100.0 * float((~keep).mean()),
              "silhouette": np.nan, "davies_bouldin": np.nan,
              "calinski_harabasz": np.nan}
    if n_clusters >= 2 and keep.sum() > n_clusters:
        Xk, lk = X[keep], labels[keep]
        sample = min(config.SILHOUETTE_SAMPLE, len(Xk))
        result["silhouette"] = float(silhouette_score(
            Xk, lk, sample_size=sample, random_state=random_state))
        result["davies_bouldin"] = float(davies_bouldin_score(Xk, lk))
        result["calinski_harabasz"] = float(calinski_harabasz_score(Xk, lk))
    return result


# ---------------------------------------------------------------- K-Means
def kmeans_search(X, k_range=None, random_state=config.RANDOM_STATE):
    """Fit K-Means for each k and return inertia and quality metrics."""
    rows = []
    for k in (k_range or config.K_RANGE):
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        metrics = evaluate_labels(X, labels, random_state)
        rows.append({"k": k, "inertia": float(model.inertia_),
                     "silhouette": metrics["silhouette"],
                     "davies_bouldin": metrics["davies_bouldin"],
                     "calinski_harabasz": metrics["calinski_harabasz"]})
    return pd.DataFrame(rows)


def choose_k(search_df, min_k=None):
    """Pick the k with the best silhouette among k >= ``min_k``."""
    min_k = min_k or config.MIN_BUSINESS_K
    eligible = search_df[search_df["k"] >= min_k]
    if eligible.empty:
        eligible = search_df
    return int(eligible.loc[eligible["silhouette"].idxmax(), "k"])


def fit_kmeans(X, k, random_state=config.RANDOM_STATE):
    """Fit and return ``(labels, model)``."""
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return model.fit_predict(X), model


# ----------------------------------------------------------- Hierarchical
def fit_hierarchical(X, k, linkage=None):
    """Agglomerative clustering (Ward by default)."""
    model = AgglomerativeClustering(
        n_clusters=k, linkage=linkage or config.HIERARCHICAL_LINKAGE)
    return model.fit_predict(X), model


def linkage_comparison(X, k, n_rows=2000, random_state=config.RANDOM_STATE):
    """Compare linkage methods on a small subsample (cheap, informative)."""
    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    take = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    rows = []
    for linkage in ("ward", "average", "complete", "single"):
        labels = AgglomerativeClustering(
            n_clusters=k, linkage=linkage).fit_predict(X[take])
        metrics = evaluate_labels(X[take], labels, random_state)
        rows.append({"linkage": linkage, "n_rows": len(take),
                     "silhouette": metrics["silhouette"],
                     "davies_bouldin": metrics["davies_bouldin"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- DBSCAN
def dbscan_search(X_fit, X_eval, random_state=config.RANDOM_STATE):
    """Grid-search eps (from k-distance quantiles) and min_samples.

    ``X_fit`` is the PCA-reduced matrix DBSCAN is fitted on (density is more
    meaningful in fewer dimensions); ``X_eval`` is the full scaled matrix so
    the silhouette is comparable with the other algorithms.

    Returns ``(search_table, best_row_dict, best_labels)``.
    """
    X_fit = np.asarray(X_fit)
    rows, store = [], {}
    for min_samples in config.DBSCAN_MIN_SAMPLES_OPTIONS:
        neighbours = NearestNeighbors(n_neighbors=min_samples).fit(X_fit)
        distances, _ = neighbours.kneighbors(X_fit)
        k_distance = np.sort(distances[:, -1])
        for quantile in config.DBSCAN_EPS_QUANTILES:
            eps = float(np.quantile(k_distance, quantile))
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
                X_fit)
            metrics = evaluate_labels(X_eval, labels, random_state)
            valid = (2 <= metrics["n_clusters"] <= config.DBSCAN_MAX_CLUSTERS
                     and metrics["noise_pct"] / 100.0
                     <= config.DBSCAN_MAX_NOISE)
            row = {"min_samples": min_samples, "eps_quantile": quantile,
                   "eps": eps, "valid": valid, **metrics}
            rows.append(row)
            store[(min_samples, quantile)] = labels
    table = pd.DataFrame(rows)

    candidates = table[table["valid"]]
    if candidates.empty:  # relax: any result with at least one cluster
        candidates = table[(table["n_clusters"] >= 1)
                           & (table["noise_pct"] <= 100 * config.DBSCAN_MAX_NOISE)]
    if candidates.empty:
        candidates = table[table["n_clusters"] >= 1]
    if candidates.empty:
        candidates = table
    ranked = candidates.assign(
        _score=candidates["silhouette"].fillna(-1.0)).sort_values(
        ["_score", "n_clusters"], ascending=False)
    best = ranked.iloc[0]
    key = (int(best["min_samples"]), float(best["eps_quantile"]))
    table["selected"] = [(int(r.min_samples), float(r.eps_quantile)) == key
                         for r in table.itertuples()]
    return table, best.drop("_score").to_dict(), store[key]


# ------------------------------------------------------------- comparison
def compare_algorithms(X_eval, label_sets, random_state=config.RANDOM_STATE):
    """Build the algorithm comparison table (Silhouette etc.)."""
    rows = []
    for key, labels in label_sets.items():
        metrics = evaluate_labels(X_eval, labels, random_state)
        sizes = pd.Series(np.asarray(labels)[np.asarray(labels) != -1])
        largest = (100.0 * sizes.value_counts(normalize=True).max()
                   if len(sizes) else np.nan)
        rows.append({"algorithm": config.ALGORITHMS[key], "key": key,
                     "n_clusters": metrics["n_clusters"],
                     "silhouette": metrics["silhouette"],
                     "davies_bouldin": metrics["davies_bouldin"],
                     "calinski_harabasz": metrics["calinski_harabasz"],
                     "noise_pct": metrics["noise_pct"],
                     "largest_cluster_pct": largest})
    return pd.DataFrame(rows)


def agreement_between(label_sets):
    """Adjusted Rand Index between algorithm labelings (noise kept as is)."""
    from sklearn.metrics import adjusted_rand_score
    keys = list(label_sets)
    matrix = pd.DataFrame(index=[config.ALGORITHMS[k] for k in keys],
                          columns=[config.ALGORITHMS[k] for k in keys],
                          dtype=float)
    for a in keys:
        for b in keys:
            matrix.loc[config.ALGORITHMS[a], config.ALGORITHMS[b]] = (
                adjusted_rand_score(label_sets[a], label_sets[b]))
    return matrix
