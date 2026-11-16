"""Central configuration for the Credit Portfolio Segmentation project.

All tunable parameters live here so that results are reproducible and the
parameters can be documented in one place (PDF guideline: Reproducibility).
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
DEFAULT_DATA_FILE = RAW_DATA_DIR / "loans_granting_model_dataset.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_OUT_DIR = OUTPUT_DIR / "data"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

# --------------------------------------------------------------------------
# Sampling (keeps hierarchical clustering / t-SNE / silhouette tractable)
# --------------------------------------------------------------------------
MAX_ROWS = 10000          # working sample used for clustering
TSNE_ROWS = 4000          # rows embedded with t-SNE
SILHOUETTE_SAMPLE = 5000  # rows used when computing silhouette scores

# --------------------------------------------------------------------------
# Cleaning and feature engineering
# --------------------------------------------------------------------------
FICO_MIN, FICO_MAX = 300, 850
DTI_MIN, DTI_MAX = 0, 100
WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99
SKEW_THRESHOLD = 1.0               # |skew| above this -> log1p transform
RARE_CATEGORY_THRESHOLD = 0.02     # categories rarer than 2% -> "other"

# --------------------------------------------------------------------------
# Dimensionality reduction
# --------------------------------------------------------------------------
PCA_VARIANCE_TARGET = 0.90
TSNE_PERPLEXITY = 30
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
K_RANGE = list(range(2, 11))
MIN_BUSINESS_K = 3   # a 2-segment split is rarely useful for segmentation
HIERARCHICAL_LINKAGE = "ward"
DBSCAN_MIN_SAMPLES_OPTIONS = (10, 20, 30)
# Include lower k-distance quantiles: on mixed borrower data, larger eps
# values can merge every dense region into one cluster. The search still
# constrains selected results by cluster count and maximum noise percentage.
DBSCAN_EPS_QUANTILES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                        0.70, 0.80, 0.85, 0.90, 0.95)
DBSCAN_MAX_NOISE = 0.20
DBSCAN_MAX_CLUSTERS = 12

# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
ROBUST_Z_THRESHOLD = 5.0

# Column names used downstream (canonical names after standardisation)
NUMERIC_FEATURES = ["loan_amnt", "annual_inc", "dti", "fico", "emp_length"]
CATEGORICAL_FEATURES = ["purpose", "home_ownership"]
TARGET_COLUMN = "default"
ID_COLUMN = "loan_id"

ALGORITHMS = {
    "kmeans": "K-Means",
    "hierarchical": "Hierarchical (Ward)",
    "dbscan": "DBSCAN",
}
PRIMARY_ALGORITHM = "kmeans"
