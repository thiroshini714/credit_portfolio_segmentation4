"""Matplotlib / Seaborn figures shared by the pipeline and the dashboard.

Every function returns a ``matplotlib.figure.Figure`` so the same code can
be saved to PNG by the pipeline and rendered by Streamlit.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("colorblind", 12)
NOISE_COLOUR = "#9a9a9a"


def cluster_colours(labels):
    """Stable colour per cluster label (grey for DBSCAN noise)."""
    colours = {}
    ordered = sorted(set(int(v) for v in labels))
    index = 0
    for label in ordered:
        if label == -1:
            colours[label] = NOISE_COLOUR
        else:
            colours[label] = PALETTE[index % len(PALETTE)]
            index += 1
    return colours


def save_figure(fig, path):
    """Save a figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_variance(variance_table, target=0.90):
    """Scree plot with cumulative explained variance."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(variance_table) + 1)
    ax.bar(x, variance_table["explained_variance_ratio"] * 100,
           color=PALETTE[0], label="Per component")
    ax.plot(x, variance_table["cumulative_variance"] * 100, marker="o",
            color=PALETTE[3], label="Cumulative")
    ax.axhline(target * 100, ls="--", color="grey",
               label=f"{target:.0%} target")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.set_xticks(x)
    ax.legend()
    return fig


def plot_k_search(search):
    """Elbow (inertia) and silhouette against k for K-Means."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(search["k"], search["inertia"], marker="o")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")
    axes[1].plot(search["k"], search["silhouette"], marker="o",
                 color=PALETTE[2])
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score by k")
    fig.tight_layout()
    return fig


def plot_dendrogram(X, n_rows=300, seed=42):
    """Truncated Ward dendrogram on a small random sample."""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    merged = linkage(X[take], method="ward")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dendrogram(merged, truncate_mode="lastp", p=20, ax=ax,
               leaf_rotation=90, no_labels=False)
    ax.set_title(f"Ward dendrogram (sample of {len(take)} borrowers)")
    ax.set_xlabel("Merged group (number of borrowers)")
    ax.set_ylabel("Merge distance")
    return fig


def plot_embedding(df, x, y, label_col, title, names=None, alpha=0.6,
                   size=10):
    """2-D scatter of an embedding coloured by cluster label."""
    data = df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = cluster_colours(data[label_col])
    for label in sorted(colours):
        part = data[data[label_col] == label]
        text = (names or {}).get(label, f"Cluster {label}")
        if label == -1:
            text = "Noise"
        ax.scatter(part[x], part[y], s=size, alpha=alpha, label=text,
                   color=colours[label], linewidths=0)
    ax.set_xlabel(x.upper().replace("_", " "))
    ax.set_ylabel(y.upper().replace("_", " "))
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=True)
    return fig


def plot_algorithm_comparison(comparison):
    """Bar chart of silhouette score per algorithm."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = comparison.copy()
    data["silhouette"] = data["silhouette"].fillna(0)
    ax.bar(data["algorithm"], data["silhouette"], color=PALETTE[:len(data)])
    for i, row in enumerate(data.itertuples()):
        ax.text(i, row.silhouette, f"{row.silhouette:.3f}", ha="center",
                va="bottom")
    ax.set_ylabel("Silhouette score (higher is better)")
    ax.set_title("Clustering algorithm comparison")
    return fig


def plot_profile_heatmap(standardised, names=None):
    """Heatmap of cluster means as z-scores versus the whole portfolio."""
    data = standardised.copy()
    data.index = [(names or {}).get(int(i), f"Cluster {i}")
                  for i in data.index]
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(data) + 2.5))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Robust z-score vs portfolio median"})
    ax.set_title("Cluster profile: feature medians vs portfolio median")
    ax.set_xlabel("")
    return fig


def plot_default_rates(table, title="Historical default rate by segment"):
    """Bar chart of default rate with 95% Wilson intervals."""
    data = table.dropna(subset=["default_rate"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"C{c}" if c != -1 else "Noise" for c in data["cluster"]]
    rates = data["default_rate"] * 100
    errors = [(data["default_rate"] - data["ci_low"]) * 100,
              (data["ci_high"] - data["default_rate"]) * 100]
    colours = [NOISE_COLOUR if c == -1 else PALETTE[i % len(PALETTE)]
               for i, c in enumerate(data["cluster"])]
    ax.bar(labels, rates, yerr=errors, capsize=4, color=colours)
    overall = data["portfolio_default_rate"].iloc[0] * 100 if len(data) else 0
    ax.axhline(overall, ls="--", color="black",
               label=f"Portfolio {overall:.1f}%")
    ax.set_ylabel("Default rate (%)")
    ax.set_xlabel("Cluster")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_cluster_sizes(profile, names=None):
    """Horizontal bar chart of cluster sizes."""
    labels = [(names or {}).get(int(c), f"Cluster {c}")
              for c in profile["cluster"]]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(labels) + 2))
    ax.barh(labels, profile["size"], color=PALETTE[:len(labels)])
    ax.set_xlabel("Number of borrowers")
    ax.set_title("Cluster sizes")
    ax.invert_yaxis()
    return fig


def plot_anomalies(df, x, y, title):
    """Embedding with flagged anomalies highlighted."""
    data = df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(8, 6))
    normal = data[~data["is_anomaly"]]
    flagged = data[data["is_anomaly"]]
    ax.scatter(normal[x], normal[y], s=6, alpha=0.35, color="#9a9a9a",
               label=f"Normal ({len(normal):,})", linewidths=0)
    ax.scatter(flagged[x], flagged[y], s=22, alpha=0.9, color=PALETTE[3],
               label=f"Anomaly ({len(flagged):,})", linewidths=0)
    ax.set_xlabel(x.upper().replace("_", " "))
    ax.set_ylabel(y.upper().replace("_", " "))
    ax.set_title(title)
    ax.legend()
    return fig


def plot_agreement(matrix):
    """Adjusted Rand Index heatmap between algorithms."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".2f", cmap="Blues",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Adjusted Rand"})
    ax.set_title("Agreement between algorithms")
    return fig


def plot_trend(trend):
    """Loan volume (bars) and default rate (line) by issue year."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(trend["issue_year"].astype(int).astype(str), trend["n_loans"],
           color=PALETTE[0], alpha=0.8)
    ax.set_ylabel("Loans issued")
    ax.set_xlabel("Issue year")
    ax2 = ax.twinx()
    ax2.plot(trend["issue_year"].astype(int).astype(str),
             trend["default_rate"] * 100, color=PALETTE[3], marker="o")
    ax2.set_ylabel("Default rate (%)")
    ax2.grid(False)
    ax.set_title("Portfolio trend: volume and default rate by issue year")
    return fig


def plot_state_bars(states, top_n=15):
    """Top states by volume with their default rate."""
    data = states.head(top_n).sort_values("n_loans")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.35 * len(data) + 2.5),
                             sharey=True)
    axes[0].barh(data["state"], data["n_loans"], color=PALETTE[0])
    axes[0].set_xlabel("Loans")
    axes[0].set_title("Loan volume by state")
    axes[1].barh(data["state"], data["default_rate"] * 100, color=PALETTE[3])
    axes[1].set_xlabel("Default rate (%)")
    axes[1].set_title("Default rate by state")
    fig.tight_layout()
    return fig


def plot_mix_heatmap(mix, title, names=None):
    """Heatmap of a row-normalised crosstab (percentages)."""
    data = mix.copy()
    if names:
        data.index = [names.get(int(i), f"Cluster {i}") for i in data.index]
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(data) + 2.5))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "% of cluster"})
    ax.set_title(title)
    return fig
