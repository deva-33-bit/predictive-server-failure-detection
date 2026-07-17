"""
Basic test suite for ServerGuard AI.

Run with: pytest tests/ -v

These are not exhaustive production-grade tests, but they cover the
things most likely to silently break the pipeline: data generation
producing a sane shape/failure rate, feature engineering not leaking
NaNs or the target column into features, and the trained model
producing valid probability outputs.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_data import generate_dataset  # noqa: E402
from preprocess import engineer_features  # noqa: E402
from trend_features import add_trend_features, _slope  # noqa: E402

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


class TestDataGeneration:
    def test_dataset_shape(self):
        df = generate_dataset()
        assert len(df) > 0
        assert "failure" in df.columns

    def test_failure_rate_is_realistic(self):
        """Failure rate should be imbalanced (not 50/50, not near-zero)."""
        df = generate_dataset()
        rate = df["failure"].mean()
        assert 0.02 < rate < 0.30, f"Failure rate {rate} is outside a realistic range"

    def test_no_missing_values(self):
        df = generate_dataset()
        assert df.isnull().sum().sum() == 0

    def test_failure_is_binary(self):
        df = generate_dataset()
        assert set(df["failure"].unique()).issubset({0, 1})


class TestTrendFeatures:
    def test_slope_of_flat_series_is_zero(self):
        assert _slope([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-6)

    def test_slope_of_increasing_series_is_positive(self):
        assert _slope([1, 2, 3, 4, 5]) > 0

    def test_trend_features_preserve_row_count(self):
        df = generate_dataset()
        df_trend = add_trend_features(df)
        assert len(df_trend) == len(df)

    def test_trend_features_no_nans(self):
        df = generate_dataset()
        df_trend = add_trend_features(df)
        trend_cols = [
            "temp_trend_slope", "cpu_trend_slope", "error_count_rolling_avg",
            "temp_delta", "error_delta"
        ]
        assert df_trend[trend_cols].isnull().sum().sum() == 0


class TestFeatureEngineering:
    def test_engineered_features_no_target_leakage_columns_missing(self):
        """Failure column should still be present (not accidentally dropped
        here -- it gets dropped explicitly later in preprocess.main())."""
        df = generate_dataset()
        engineered = engineer_features(df)
        assert "failure" in engineered.columns

    def test_engineered_features_no_nans(self):
        df = generate_dataset()
        engineered = engineer_features(df)
        assert engineered.isnull().sum().sum() == 0

    def test_engineered_adds_expected_columns(self):
        df = generate_dataset()
        engineered = engineer_features(df)
        for col in ["disk_io_total", "network_io_total", "cpu_mem_pressure",
                    "temp_error_interaction"]:
            assert col in engineered.columns


class TestTrainedModel:
    @pytest.fixture(scope="class")
    def model(self):
        model_path = os.path.join(ROOT_DIR, "models", "random_forest.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Model not trained yet -- run src/train_models.py first")
        return joblib.load(model_path)

    def test_model_predicts_valid_probabilities(self, model):
        X_test_path = os.path.join(ROOT_DIR, "data", "processed", "X_test.csv")
        if not os.path.exists(X_test_path):
            pytest.skip("X_test.csv not found -- run src/preprocess.py first")
        X_test = pd.read_csv(X_test_path)
        proba = model.predict_proba(X_test)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_model_feature_count_matches_training(self, model):
        X_test_path = os.path.join(ROOT_DIR, "data", "processed", "X_test.csv")
        if not os.path.exists(X_test_path):
            pytest.skip("X_test.csv not found -- run src/preprocess.py first")
        X_test = pd.read_csv(X_test_path)
        assert list(X_test.columns) == list(model.feature_names_in_)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
