"""Dimensionality reduction: PCA (feature reduction), t-SNE and UMAP."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config


def run_pca(X, variance_target=None, random_state=config.RANDOM_STATE):
    """Fit PCA and return ``(reduced_array, pca_model, variance_table, n)``.

    ``n`` is the smallest number of components reaching ``variance_target``
    of the explained variance.
    """
    variance_target = variance_target or config.PCA_VARIANCE_TARGET
    full = PCA(random_state=random_state).fit(X)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = max(2, min(n_components, X.shape[1]))
    table = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(cumulative))],
        "explained_variance_ratio": full.explained_variance_ratio_,
        "cumulative_variance": cumulative,
    })
    model = PCA(n_components=n_components, random_state=random_state)
    reduced = model.fit_transform(X)
    return reduced, model, table, n_components


def pca_loadings(model, feature_names, top_components=3):
    """Loadings of the first components, for interpreting the axes."""
    count = min(top_components, model.n_components_)
    return pd.DataFrame(model.components_[:count].T,
                        index=feature_names,
                        columns=[f"PC{i + 1}" for i in range(count)])


def run_tsne(X_reduced, index, n_rows=None, perplexity=None,
             random_state=config.RANDOM_STATE):
    """Embed a random subset with t-SNE; returns a 2-column DataFrame."""
    n_rows = min(n_rows or config.TSNE_ROWS, len(X_reduced))
    perplexity = perplexity or config.TSNE_PERPLEXITY
    rng = np.random.default_rng(random_state)
    chosen = np.sort(rng.choice(len(X_reduced), size=n_rows, replace=False))
    model = TSNE(n_components=2, perplexity=min(perplexity, n_rows - 1),
                 init="pca", learning_rate="auto", max_iter=1000,
                 random_state=random_state)
    coords = model.fit_transform(X_reduced[chosen])
    return pd.DataFrame(coords, columns=["tsne_1", "tsne_2"],
                        index=np.asarray(index)[chosen])


def umap_available():
    """True when the optional umap-learn package can be imported."""
    try:
        import umap  # noqa: F401
        return True
    except Exception:  # ImportError or numba problems
        return False


def run_umap(X_reduced, index, random_state=config.RANDOM_STATE):
    """UMAP embedding, or ``None`` when umap-learn is not installed."""
    if not umap_available():
        return None
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=config.UMAP_NEIGHBORS,
                        min_dist=config.UMAP_MIN_DIST,
                        random_state=random_state)
    coords = reducer.fit_transform(X_reduced)
    return pd.DataFrame(coords, columns=["umap_1", "umap_2"],
                        index=np.asarray(index))
