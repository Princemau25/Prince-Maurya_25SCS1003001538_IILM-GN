import pandas as pd
import numpy as np

def generate_realistic_medical_data(num_patients=2500):
    print(f"🧬 Generating synthetic medical records for {num_patients} patients...")
    np.random.seed(42)
    
    age = np.random.normal(loc=55, scale=15, size=num_patients).astype(int)
    age = np.clip(age, 25, 85) 
    gender = np.random.choice([0, 1], size=num_patients, p=[0.48, 0.52])
    bmi = np.random.normal(loc=28.0, scale=6.0, size=num_patients)
    bmi = np.clip(bmi, 18.5, 45.0).round(1)
    
    base_bp = np.random.normal(loc=110, scale=10, size=num_patients)
    blood_pressure = (base_bp + (age * 0.2) + (bmi * 0.5)).astype(int)
    base_chol = np.random.normal(loc=180, scale=30, size=num_patients)
    cholesterol = (base_chol + (age * 0.5)).astype(int)
    
    smoking = np.random.choice([0, 1], size=num_patients, p=[0.75, 0.25])
    diabetes = np.random.choice([0, 1], size=num_patients, p=[0.85, 0.15])
    family_history = np.random.choice([0, 1], size=num_patients, p=[0.70, 0.30])
    chest_pain = np.random.choice([0, 1], size=num_patients, p=[0.80, 0.20])
    
    risk_score = ((age * 0.03) + (gender * 0.5) + (bmi * 0.05) + 
                  ((blood_pressure - 120) * 0.02) + ((cholesterol - 200) * 0.01) + 
                  (smoking * 1.5) + (diabetes * 1.2) + (family_history * 1.0) +
                  (chest_pain * 2.0))
    
    probability = 1 / (1 + np.exp(-(risk_score - 4.5)))
    target = np.random.binomial(1, probability)
    
    df = pd.DataFrame({
        'age': age, 'gender': gender, 'chest_pain': chest_pain,
        'blood_pressure': blood_pressure, 'cholesterol': cholesterol,
        'diabetes': diabetes, 'bmi': bmi, 'smoking': smoking,
        'family_history': family_history, 'target': target
    })
    
    df.to_csv("modern_heart_data.csv", index=False)
    print("✅ Successfully saved to 'modern_heart_data.csv'")

if __name__ == "__main__":
    generate_realistic_medical_data()