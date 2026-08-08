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

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
