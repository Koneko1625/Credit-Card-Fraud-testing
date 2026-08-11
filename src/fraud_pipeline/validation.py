"""Stage 2: Data validation.

This is the pandera schema from Part A's "Data Validation" section,
lifted verbatim into a reusable function so both the pipeline and the
test suite can call it against any DataFrame (not just the one global
`df` the notebook relied on).
"""
from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera import Check

logger = logging.getLogger(__name__)

SCHEMA = pa.DataFrameSchema(
    {
        "Time": pa.Column(float, Check.ge(0)),
        "Amount": pa.Column(float, Check.ge(0)),
        "Class": pa.Column(int, Check.isin([0, 1])),
    },
    coerce=True,
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates df against SCHEMA and returns it (coerced) on success.

    Raises pandera.errors.SchemaError on failure — the pipeline should
    let that propagate and stop the run rather than silently continuing
    on bad data.
    """
    validated_df = SCHEMA.validate(df)
    logger.info(
        "Validation passed: %s rows, %s columns, %s missing, %s duplicates",
        validated_df.shape[0],
        validated_df.shape[1],
        int(validated_df.isnull().sum().sum()),
        int(validated_df.duplicated().sum()),
    )
    return validated_df


def validation_summary(df: pd.DataFrame) -> dict:
    """Same summary Part A printed, returned as a dict so it can be
    logged to MLflow as run params/tags instead of just stdout."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
