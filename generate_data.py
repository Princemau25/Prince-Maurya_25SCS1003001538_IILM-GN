"""
generate_data.py
Simulates patient clinical data, trains an XGBoost classifier, and saves the
pre-trained model to xgboost_model.pkl for use by main.py and evaluate_model.py.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SAMPLES = 2000
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "modern_patients.csv"


def simulate_data(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic clinical dataset for cardiovascular risk prediction."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(25, 80, n_samples).astype(float)
    gender = rng.integers(0, 2, n_samples).astype(float)          # 0=Female, 1=Male
    chest_pain = rng.integers(0, 2, n_samples).astype(float)      # 0=No, 1=Yes
    blood_pressure = rng.uniform(80, 200, n_samples)
    cholesterol = rng.uniform(100, 400, n_samples)
    diabetes = rng.integers(0, 2, n_samples).astype(float)        # 0=No, 1=Yes
    bmi = rng.uniform(15, 45, n_samples)
    smoking = rng.integers(0, 2, n_samples).astype(float)         # 0=No, 1=Yes
    family_history = rng.integers(0, 2, n_samples).astype(float)  # 0=No, 1=Yes

    # Deterministic risk score to create a realistic (non-random) label
    risk_score = (
        0.03 * (age - 25)
        + 0.15 * gender
        + 0.20 * chest_pain
        + 0.005 * (blood_pressure - 80)
        + 0.003 * (cholesterol - 100)
        + 0.25 * diabetes
        + 0.02 * (bmi - 18)
        + 0.20 * smoking
        + 0.15 * family_history
    )
    noise = rng.normal(0, 0.3, n_samples)
    heart_disease = (risk_score + noise > 1.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "chest_pain": chest_pain,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "diabetes": diabetes,
        "bmi": bmi,
        "smoking": smoking,
        "family_history": family_history,
        "heart_disease": heart_disease,
    })
    return df


def train_and_save_model(df: pd.DataFrame) -> dict:
    """Train an XGBoost model and persist it together with the scaler."""
    feature_cols = [
        "age", "gender", "chest_pain", "blood_pressure",
        "cholesterol", "diabetes", "bmi", "smoking", "family_history",
    ]
    X = df[feature_cols].values
    y = df["heart_disease"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    payload = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    # Return test split so evaluate_model.py can report metrics without re-training
    return {
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


def main():
    print("=== Heartbeat — Data Generation & Model Training ===\n")
    print(f"Simulating {N_SAMPLES} patient records …")
    df = simulate_data()

    # Basic data cleaning: drop any rows with NaN (none expected from simulation)
    df.dropna(inplace=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to '{DATA_PATH}'  ({len(df)} rows × {len(df.columns)} columns)")

    print("Training XGBoost classifier …")
    result = train_and_save_model(df)
    print(f"Model saved to '{MODEL_PATH}'")

    # Quick sanity-check accuracy on held-out test set
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    preds = payload["model"].predict(result["X_test"])
    accuracy = (preds == result["y_test"]).mean()
    print(f"Hold-out accuracy: {accuracy:.2%}\n")
    print("Done. You can now run  python main.py  or  python evaluate_model.py")


if __name__ == "__main__":
    main()
