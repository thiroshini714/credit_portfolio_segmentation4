"""Unit and integration tests. Run with:  python -m unittest -v"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import (anomaly, clustering, data_loader, features,  # noqa: E402
                 generate_sample_data, pipeline, preprocessing, profiling)


class TestParsing(unittest.TestCase):
    def test_emp_length(self):
        parse = preprocessing.parse_emp_length
        self.assertEqual(parse("10+ years"), 10.0)
        self.assertEqual(parse("< 1 year"), 0.0)
        self.assertEqual(parse("3 years"), 3.0)
        self.assertTrue(np.isnan(parse("n/a")))
        self.assertTrue(np.isnan(parse(np.nan)))

    def test_encode_default(self):
        series = pd.Series(["Fully Paid", "Charged Off", "Current",
                            "Default", None])
        out = preprocessing.encode_default(series)
        self.assertEqual(out.iloc[0], 0.0)
        self.assertEqual(out.iloc[1], 1.0)
        self.assertTrue(np.isnan(out.iloc[2]))
        self.assertEqual(out.iloc[3], 1.0)
        self.assertTrue(np.isnan(out.iloc[4]))
        numeric = preprocessing.encode_default(pd.Series([0, 1, 1, 0]))
        self.assertEqual(list(numeric), [0.0, 1.0, 1.0, 0.0])

    def test_dates(self):
        parsed = preprocessing.parse_dates(
            pd.Series(["Dec-2015", "2016-03-01", "garbage"]))
        self.assertEqual(parsed.iloc[0].year, 2015)
        self.assertEqual(parsed.iloc[1].month, 3)
        self.assertTrue(pd.isna(parsed.iloc[2]))


class TestLoader(unittest.TestCase):
    def test_alias_detection_and_fico_pair(self):
        raw = pd.DataFrame({
            "ID": [1, 2], "Loan Amount": [1000, 2000],
            "Annual Income": [50000, 60000],
            "fico_range_low": [680, 700], "fico_range_high": [684, 704]})
        out, mapping = data_loader.standardise_columns(raw)
        self.assertIn("loan_amnt", out.columns)
        self.assertEqual(out["fico"].tolist(), [682.0, 702.0])
        self.assertIn("fico", mapping)

    def test_missing_required_columns_raises(self):
        with self.assertRaises(ValueError):
            data_loader.standardise_columns(pd.DataFrame({"a": [1]}))

    def test_manual_override(self):
        raw = pd.DataFrame({"amt": [1], "inc": [2], "score": [700]})
        out, _ = data_loader.standardise_columns(
            raw, {"loan_amnt": "amt", "annual_inc": "inc", "fico": "score"})
        self.assertEqual(out["fico"].iloc[0], 700)


class TestCleaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = generate_sample_data.generate(3000, seed=1)
        std, _ = data_loader.standardise_columns(raw)
        cls.raw_rows = len(raw)
        cls.clean, cls.report = preprocessing.clean_data(std)

    def test_duplicates_removed(self):
        self.assertLess(len(self.clean), self.raw_rows)
        self.assertGreater(self.report["duplicate_rows_removed"], 0)

    def test_no_missing_features(self):
        cols = ["loan_amnt", "annual_inc", "dti", "fico", "emp_length",
                "purpose", "home_ownership"]
        self.assertEqual(int(self.clean[cols].isna().sum().sum()), 0)

    def test_values_valid(self):
        self.assertTrue(self.clean["fico"].between(300, 850).all())
        self.assertTrue((self.clean["annual_inc"] > 0).all())
        self.assertTrue(self.clean["dti"].between(0, 100).all())
        self.assertTrue(set(self.clean["home_ownership"]) <=
                        {"mortgage", "rent", "own", "other"})

    def test_default_encoded(self):
        self.assertTrue(set(self.clean["default"].dropna()) <= {0.0, 1.0})
        self.assertGreater(self.clean["default"].isna().sum(), 0)

    def test_winsorise_caps_outliers(self):
        capped = preprocessing.winsorise(self.clean)
        self.assertLess(capped["annual_inc"].max(),
                        self.clean["annual_inc"].max())


class TestFeatures(unittest.TestCase):
    def test_default_and_id_excluded(self):
        raw = generate_sample_data.generate(2000, seed=2)
        std, _ = data_loader.standardise_columns(raw)
        clean, _ = preprocessing.clean_data(std)
        feature_set = features.build_features(preprocessing.winsorise(clean))
        columns = " ".join(feature_set.X.columns).lower()
        self.assertNotIn("default", columns)
        self.assertNotIn("loan_id", columns)
        self.assertFalse(feature_set.X.isna().any().any())
        for col in feature_set.numeric_features:
            self.assertAlmostEqual(feature_set.X[col].mean(), 0, places=6)


class TestClustering(unittest.TestCase):
    def test_kmeans_recovers_blobs(self):
        X, _ = make_blobs(n_samples=600, centers=4, random_state=0)
        search = clustering.kmeans_search(X, range(2, 8))
        self.assertEqual(clustering.choose_k(search), 4)

    def test_dbscan_search_returns_labels(self):
        X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.6,
                          random_state=0)
        table, best, labels = clustering.dbscan_search(X, X)
        self.assertEqual(len(labels), 500)
        self.assertTrue(table["selected"].sum() == 1)
        self.assertGreaterEqual(best["n_clusters"], 2)
        self.assertLessEqual(best["noise_pct"],
                             100 * config.DBSCAN_MAX_NOISE)

    def test_evaluate_handles_noise_only(self):
        metrics = clustering.evaluate_labels(np.zeros((10, 2)),
                                             np.full(10, -1))
        self.assertTrue(np.isnan(metrics["silhouette"]))
        self.assertEqual(metrics["noise_pct"], 100.0)


class TestProfilingAndAnomaly(unittest.TestCase):
    def test_wilson_interval_contains_rate(self):
        low, high = profiling.wilson_interval(20, 100)
        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.2)

    def test_default_rate_table(self):
        df = pd.DataFrame({"c": [0] * 100 + [1] * 100,
                           "default": [0.0] * 90 + [1.0] * 10
                           + [0.0] * 50 + [1.0] * 50})
        table = profiling.default_rate_table(df, "c")
        self.assertAlmostEqual(table.loc[0, "default_rate"], 0.1)
        self.assertEqual(table.loc[1, "risk_tier"], "Higher risk")
        self.assertEqual(table.loc[0, "risk_tier"], "Lower risk")
        chi = profiling.chi_square_default(df, "c")
        self.assertLess(chi["p_value"], 0.001)

    def test_extreme_value_flagged(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "loan_amnt": rng.normal(10000, 1000, 500).clip(1000),
            "annual_inc": rng.normal(60000, 5000, 500).clip(1000),
            "dti": rng.normal(18, 3, 500), "fico": rng.normal(700, 20, 500),
            "emp_length": rng.integers(0, 10, 500).astype(float),
            "default": 0.0})
        df.loc[0, "annual_inc"] = 5_000_000
        flagged = anomaly.detect_anomalies(df, np.zeros(500, dtype=int))
        self.assertTrue(flagged.loc[0, "is_anomaly"])
        self.assertIn("annual_inc", flagged.loc[0, "extreme_features"])


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_runs_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = tmp / "loans.csv"
            generate_sample_data.generate(2500, seed=3).to_csv(
                csv, index=False)
            results = pipeline.run(data_path=csv, max_rows=1500,
                                   skip_umap=True, skip_tsne=True,
                                   output_dir=tmp / "out")
            out = tmp / "out"
            for name in ("data/clustered_borrowers.csv",
                         "tables/algorithm_comparison.csv",
                         "tables/default_rates_by_cluster.csv",
                         "tables/anomalies.csv", "tables/insights.json",
                         "reports/analysis_report.md",
                         "models/models.joblib"):
                self.assertTrue((out / name).exists(), name)
            columns = list(results["feature_set"].X.columns)
            self.assertFalse(any("default" in c for c in columns))
            self.assertEqual(len(results["comparison"]), 3)
            self.assertGreaterEqual(results["metadata"]["kmeans_k"],
                                    config.MIN_BUSINESS_K)
            dashboard = pd.read_csv(out / "data/clustered_borrowers.csv")
            self.assertNotIn("loan_id", dashboard.columns)

    def test_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = tmp / "loans.csv"
            generate_sample_data.generate(2000, seed=4).to_csv(
                csv, index=False)
            runs = [pipeline.run(data_path=csv, max_rows=1200,
                                 skip_umap=True, skip_tsne=True,
                                 output_dir=tmp / f"out{i}")
                    for i in range(2)]
            a = runs[0]["result_frame"]["kmeans_cluster"].to_numpy()
            b = runs[1]["result_frame"]["kmeans_cluster"].to_numpy()
            self.assertTrue((a == b).all())


if __name__ == "__main__":
    unittest.main()
