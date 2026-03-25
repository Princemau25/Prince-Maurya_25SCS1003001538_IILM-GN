"""
main.py
Heartbeat — Clinical Risk Prediction System
Collects nine patient health indicators from the user, runs them through the
pre-trained XGBoost model, displays the risk assessment, and archives the
result in a local SQLite database (modern_patients.db).
"""

import pickle
import sqlite3
import sys
from datetime import datetime

import numpy as np

MODEL_PATH = "xgboost_model.pkl"
DB_PATH = "modern_patients.db"

BANNER = r"""
  _   _                _   _                _
 | | | | ___  __ _ _ _| |_| |__   ___  __ _| |_
 | |_| |/ _ \/ _` | '_|  _| '_ \ / _ \/ _` | __|
 |  _  |  __/ (_| | |  | |_| |_) |  __/ (_| | |_
 |_| |_|\___|\__,_|_|   \__|_.__/ \___|\__,_|\__|

   Clinical Risk Prediction System
"""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    """Create the patients table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            age             REAL    NOT NULL,
            gender          INTEGER NOT NULL,
            chest_pain      INTEGER NOT NULL,
            blood_pressure  REAL    NOT NULL,
            cholesterol     REAL    NOT NULL,
            diabetes        INTEGER NOT NULL,
            bmi             REAL    NOT NULL,
            smoking         INTEGER NOT NULL,
            family_history  INTEGER NOT NULL,
            prediction      INTEGER NOT NULL,
            risk_label      TEXT    NOT NULL
        )
        """
    )
    conn.commit()


def save_record(conn: sqlite3.Connection, inputs: dict, prediction: int, risk_label: str) -> None:
    """Insert a prediction record into the database."""
    conn.execute(
        """
        INSERT INTO patients
            (timestamp, age, gender, chest_pain, blood_pressure, cholesterol,
             diabetes, bmi, smoking, family_history, prediction, risk_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            inputs["age"],
            inputs["gender"],
            inputs["chest_pain"],
            inputs["blood_pressure"],
            inputs["cholesterol"],
            inputs["diabetes"],
            inputs["bmi"],
            inputs["smoking"],
            inputs["family_history"],
            prediction,
            risk_label,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _prompt_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print(f"  [!] Please enter a numeric value between {lo} and {hi}.")
            continue
        if not (lo <= val <= hi):
            print(f"  [!] Value must be between {lo} and {hi}.")
            continue
        return val


def _prompt_binary(prompt: str) -> int:
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("0", "n", "no"):
            return 0
        if raw in ("1", "y", "yes"):
            return 1
        print("  [!] Please enter 0 (No) or 1 (Yes).")


def collect_inputs() -> dict:
    """Interactively collect and validate the nine clinical parameters."""
    print("\n  Please enter the following patient details:\n")

    age = _prompt_float("  Age (25–100): ", 25, 100)
    print("  Gender: 1 = Male, 0 = Female")
    gender = _prompt_binary("  Gender (0/1): ")
    print("  Chest Pain: 1 = Yes, 0 = No")
    chest_pain = _prompt_binary("  Chest Pain (0/1): ")
    blood_pressure = _prompt_float("  Resting Blood Pressure in mm Hg (60–250): ", 60, 250)
    cholesterol = _prompt_float("  Serum Cholesterol in mg/dl (50–600): ", 50, 600)
    print("  Diabetes: 1 = Yes, 0 = No")
    diabetes = _prompt_binary("  Diabetes (0/1): ")
    bmi = _prompt_float("  BMI (10–60): ", 10, 60)
    print("  Smoking: 1 = Yes, 0 = No")
    smoking = _prompt_binary("  Smoking (0/1): ")
    print("  Family History of Heart Disease: 1 = Yes, 0 = No")
    family_history = _prompt_binary("  Family History (0/1): ")

    return {
        "age": age,
        "gender": gender,
        "chest_pain": chest_pain,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "diabetes": diabetes,
        "bmi": bmi,
        "smoking": smoking,
        "family_history": family_history,
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_PATH) -> dict:
    """Load the serialised model payload."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(
            f"[ERROR] Model file '{path}' not found.\n"
            "        Please run  python generate_data.py  first."
        )
        sys.exit(1)


def predict(payload: dict, inputs: dict) -> tuple[int, str]:
    """Scale inputs and return (binary prediction, risk label).

    Args:
        payload: Dict containing keys 'model' (XGBClassifier),
                 'scaler' (StandardScaler), and 'feature_cols' (list[str]).
        inputs:  Dict mapping each feature name to its numeric value.

    Returns:
        A tuple of (prediction, risk_label) where prediction is 0 or 1 and
        risk_label is 'No Risk' or 'Heart Disease Risk'.
    """
    feature_cols = payload["feature_cols"]
    X = np.array([[inputs[col] for col in feature_cols]], dtype=float)
    X_scaled = payload["scaler"].transform(X)
    pred = int(payload["model"].predict(X_scaled)[0])
    label = "Heart Disease Risk" if pred == 1 else "No Risk"
    return pred, label


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print(BANNER)

    payload = load_model()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    while True:
        inputs = collect_inputs()
        pred, label = predict(payload, inputs)
        save_record(conn, inputs, pred, label)

        print("\n" + "=" * 50)
        if pred == 1:
            print("  ⚠  RESULT: HEART DISEASE RISK DETECTED")
            print("     Please consult a medical professional immediately.")
        else:
            print("  ✓  RESULT: NO SIGNIFICANT HEART DISEASE RISK")
            print("     Continue maintaining a healthy lifestyle.")
        print("=" * 50)
        print(f"  Record saved to '{DB_PATH}'.\n")

        again = input("  Assess another patient? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Thank you for using Heartbeat. Stay healthy!\n")
            break

    conn.close()


if __name__ == "__main__":
    main()
