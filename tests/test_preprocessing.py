"""Unit tests for the split + scaling stage — the two things Part A's
notebook did in cells that had no tests at all: (1) the split must be
time-ordered, not random, and hold the configured ratio; (2) the scaler
must be fit on train only and actually normalize Amount.
"""
import numpy as np

from src.fraud_pipeline.preprocessing import scale_amount, time_ordered_split


def test_split_respects_train_fraction(synthetic_df):
    X_train, X_test, y_train, y_test = time_ordered_split(synthetic_df, train_fraction=0.8)
    expected_train_len = int(len(synthetic_df) * 0.8)
    assert len(X_train) == expected_train_len
    assert len(X_test) == len(synthetic_df) - expected_train_len
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_is_time_ordered_not_random(synthetic_df):
    X_train, X_test, _, _ = time_ordered_split(synthetic_df, train_fraction=0.8)
    # every train timestamp must be <= every test timestamp
    assert X_train["Time"].max() <= X_test["Time"].min()


def test_class_column_dropped_from_features(synthetic_df):
    X_train, X_test, _, _ = time_ordered_split(synthetic_df)
    assert "Class" not in X_train.columns
    assert "Class" not in X_test.columns


def test_scaler_fit_on_train_only(synthetic_df):
    X_train, X_test, _, _ = time_ordered_split(synthetic_df, train_fraction=0.8)
    X_train_scaled, X_test_scaled, scaler = scale_amount(X_train, X_test)

    # scaled train Amount should be ~mean 0, ~std 1
    assert abs(X_train_scaled["Amount"].mean()) < 1e-8
    assert abs(X_train_scaled["Amount"].std(ddof=0) - 1.0) < 1e-8

    # scaler params must come from train, and test must use the same transform
    manual_test_scaled = (X_test["Amount"] - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
    assert np.allclose(X_test_scaled["Amount"].values, manual_test_scaled.values)


def test_other_features_untouched_by_scaling(synthetic_df):
    X_train, X_test, _, _ = time_ordered_split(synthetic_df)
    X_train_scaled, _, _ = scale_amount(X_train, X_test)
    assert np.allclose(X_train_scaled["V1"].values, X_train["V1"].values)
