import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n_rows = 200
    data = {
        "Time": np.sort(rng.uniform(0, 100_000, n_rows)),
        **{f"V{i}": rng.normal(0, 1, n_rows) for i in range(1, 29)},
        "Amount": rng.uniform(0, 500, n_rows),
        "Class": rng.choice([0, 1], size=n_rows, p=[0.98, 0.02]),
    }
    return pd.DataFrame(data)
