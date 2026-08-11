"""Unit tests for the pandera schema — code correctness, not model quality.
Checks the validation stage catches exactly the malformed inputs it
should and passes clean ones through unchanged.
"""
import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.fraud_pipeline.validation import validate_data, validation_summary


def test_valid_data_passes(synthetic_df):
    validated = validate_data(synthetic_df)
    assert len(validated) == len(synthetic_df)


def test_negative_amount_rejected(synthetic_df):
    bad_df = synthetic_df.copy()
    bad_df.loc[0, "Amount"] = -50.0
    with pytest.raises(SchemaError):
        validate_data(bad_df)


def test_negative_time_rejected(synthetic_df):
    bad_df = synthetic_df.copy()
    bad_df.loc[0, "Time"] = -1.0
    with pytest.raises(SchemaError):
        validate_data(bad_df)


def test_invalid_class_label_rejected(synthetic_df):
    bad_df = synthetic_df.copy()
    bad_df.loc[0, "Class"] = 2  # only 0/1 are valid
    with pytest.raises(SchemaError):
        validate_data(bad_df)


def test_validation_summary_counts_missing_and_duplicates(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    summary = validation_summary(df)
    assert summary["rows"] == len(synthetic_df) + 1
    assert summary["duplicate_rows"] == 1
