"""
evaluate_model.py
Loads the pre-trained XGBoost model and the synthetic dataset, then reports
classification performance metrics: accuracy, precision, recall, and F1-score.
"""

import pickle
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "modern_patients.csv"
RANDOM_STATE = 42


def load_model_payload():
    """Load the saved model payload (model + scaler + feature list)."""
    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
    except FileNotFoundError:
        print(
            f"[ERROR] '{MODEL_PATH}' not found.\n"
            "        Please run  python generate_data.py  first."
        )
        sys.exit(1)
    return payload


def load_data(feature_cols):
    """Load and return feature matrix X and label vector y from the CSV."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(
            f"[ERROR] '{DATA_PATH}' not found.\n"
            "        Please run  python generate_data.py  first."
        )
        sys.exit(1)
    X = df[feature_cols].values
    y = df["heart_disease"].values
    return X, y


def main():
    print("=== Heartbeat — Model Evaluation ===\n")

    payload = load_model_payload()
    model = payload["model"]
    scaler = payload["scaler"]
    feature_cols = payload["feature_cols"]

    X, y = load_data(feature_cols)

    # Reproduce the same train/test split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["No Risk", "Heart Disease Risk"]
    )

    print(f"Accuracy : {acc:.4f}  ({acc:.2%})\n")
    print("Confusion Matrix:")
    print(f"  True Negatives  : {cm[0][0]}")
    print(f"  False Positives : {cm[0][1]}")
    print(f"  False Negatives : {cm[1][0]}")
    print(f"  True Positives  : {cm[1][1]}\n")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    main()
