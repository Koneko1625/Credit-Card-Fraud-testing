import pytest
from pandera.errors import SchemaError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.fraud_pipeline.inference import predict_batch


@pytest.fixture
def fitted_model_and_scaler(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    y = synthetic_df["Class"]
    scaler = StandardScaler().fit(X[["Amount"]])
    X_scaled = X.copy()
    X_scaled["Amount"] = scaler.transform(X[["Amount"]])
    model = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    return model, scaler


def test_predict_batch_adds_expected_columns(synthetic_df, fitted_model_and_scaler):
    model, scaler = fitted_model_and_scaler
    batch = synthetic_df.drop(columns=["Class"])
    scored = predict_batch(batch, model, scaler, threshold=0.5)
    assert "fraud_probability" in scored.columns
    assert "fraud_prediction" in scored.columns
    assert len(scored) == len(batch)


def test_predict_batch_preserves_row_count_with_labels_present(synthetic_df, fitted_model_and_scaler):
    model, scaler = fitted_model_and_scaler
    scored = predict_batch(synthetic_df, model, scaler, threshold=0.5)
    assert len(scored) == len(synthetic_df)
    assert "Class" in scored.columns  # ground truth preserved for drift comparison


def test_threshold_controls_predicted_positive_rate(synthetic_df, fitted_model_and_scaler):
    model, scaler = fitted_model_and_scaler
    batch = synthetic_df.drop(columns=["Class"])
    low_thresh = predict_batch(batch, model, scaler, threshold=0.01)
    high_thresh = predict_batch(batch, model, scaler, threshold=0.99)
    assert low_thresh["fraud_prediction"].sum() >= high_thresh["fraud_prediction"].sum()


def test_malformed_batch_is_rejected(synthetic_df, fitted_model_and_scaler):
    model, scaler = fitted_model_and_scaler
    bad_batch = synthetic_df.drop(columns=["Class"]).copy()
    bad_batch.loc[0, "Amount"] = -100.0
    with pytest.raises(SchemaError):
        predict_batch(bad_batch, model, scaler, threshold=0.5)
