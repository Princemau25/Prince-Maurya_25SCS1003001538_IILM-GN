import os
import pandas as pd
import sqlite3
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_model_if_needed():
    model_filename = 'modern_xgboost_model.pkl'
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
        
    if not os.path.exists("modern_heart_data.csv"):
        print("Error: Run 'generate_data.py' first to create the dataset.")
        exit()
        
    print("⚙️ Training AI Model...")
    df = pd.read_csv("modern_heart_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(eval_metric='logloss', random_state=42, max_depth=4, learning_rate=0.05)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    return model

def init_db():
    conn = sqlite3.connect('modern_patients.db') 
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date_tested TEXT, patient_name TEXT, 
            age INTEGER, gender TEXT, chest_pain INTEGER, blood_pressure REAL, 
            cholesterol REAL, diabetes INTEGER, bmi REAL, smoking INTEGER, 
            family_history INTEGER, ai_prediction TEXT, ai_confidence REAL)''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    ai_model = train_model_if_needed()
    
    try:
        print("\n--- NEW PATIENT ENTRY ---")
        patient_name = input("Patient Full Name: ").strip() or "Unknown_Patient"
        age = int(input("1. Age (years): "))
        gender_input = input("2. Gender (M/F): ").strip().upper()
        gender = 1 if gender_input == 'M' else 0
        gender_text = "Male" if gender == 1 else "Female"
        chest_pain = int(input("3. Frequent chest pain? (1=Yes, 0=No): "))
        blood_pressure = float(input("4. Resting Blood Pressure: "))
        cholesterol = float(input("5. Cholesterol Level: "))
        diabetes = int(input("6. Diagnosed with Diabetes? (1=Yes, 0=No): "))
        bmi = float(input("7. BMI (Body Mass Index): "))
        smoking = int(input("8. Smoker? (1=Yes, 0=No): "))
        family_history = int(input("9. Family history of heart disease? (1=Yes, 0=No): "))

        feature_names = ['age', 'gender', 'chest_pain', 'blood_pressure', 'cholesterol', 'diabetes', 'bmi', 'smoking', 'family_history']
        patient_data = pd.DataFrame([[age, gender, chest_pain, blood_pressure, cholesterol, diabetes, bmi, smoking, family_history]], columns=feature_names)

        pred = ai_model.predict(patient_data)[0]
        proba = ai_model.predict_proba(patient_data)[0][1]

        prediction_text = "HIGH RISK" if pred == 1 else "LOW RISK"
        confidence = round(proba * 100, 2)

        print(f"\n--- RESULTS FOR {patient_name.upper()} ---")
        print(f"PREDICTION: {prediction_text}")
        print(f"CONFIDENCE: {confidence}%")

        conn = sqlite3.connect('modern_patients.db')
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = (timestamp, patient_name, age, gender_text, chest_pain, blood_pressure, cholesterol, diabetes, bmi, smoking, family_history, prediction_text, confidence)
        cursor.execute('''INSERT INTO patient_records (date_tested, patient_name, age, gender, chest_pain, blood_pressure, cholesterol, diabetes, bmi, smoking, family_history, ai_prediction, ai_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', record)
        conn.commit()
        conn.close()

        report_filename = f"{patient_name.replace(' ', '_')}_report.txt"
        with open(report_filename, "w") as file:
            file.write(f"--- HEART DISEASE PREDICTION REPORT ---\nDate: {timestamp}\nPatient: {patient_name}\nPrediction: {prediction_text} ({confidence}% Confidence)\n")
        print(f"✅ Saved DB record and generated '{report_filename}'")

    except ValueError:
        print("\n❌ Error: Invalid input.")