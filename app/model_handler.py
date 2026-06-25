import joblib
import pandas as pd


def load_model(model_path: str = "models/credit_default_lr.joblib"):
    return joblib.load(model_path)


def prepare_input(data: dict, expected_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([data])

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required fields: {missing_cols}")

    df = df[expected_columns]
    return df


def predict_default(model, data: dict, expected_columns: list[str], threshold: float = 0.5):
    input_df = prepare_input(data, expected_columns)
    probability = float(model.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": probability
    }
