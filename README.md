Heartbeat: Advanced Cardiovascular Predictive System
Heartbeat is a modernized machine learning pipeline designed to predict cardiovascular disease using high-performance gradient boosting. Unlike basic scripts, this project features a modular architecture, a pre-trained XGBoost model, and a local SQLite database for patient record management.

📂 Project Structure
Based on the extracted source files, the project is organized as follows:


main.py: The primary entry point for the application.


generate_data.py: Handles data simulation and preprocessing for the model.


evaluate_model.py: Contains functions to test model accuracy and performance metrics.

modern/:


heart_data.csv: The core clinical dataset.


xgboost_model.pkl: A high-accuracy, pre-trained XGBoost classifier.


modern_patients.db: A SQLite database to store and track patient history.


requirements.txt: Lists essential dependencies including xgboost, pandas, scikit-learn, and joblib.

🚀 Technical Upgrades
This version of the project introduces several "Pro" features over standard Logistic Regression models:


XGBoost Integration: Utilizes extreme gradient boosting for superior non-linear pattern recognition.


Database Persistence: Uses a SQL backend (.db file) to ensure patient data isn't lost after a session ends.


Modular Scripting: Separates data generation, evaluation, and execution into distinct Python modules for better maintainability.

🛠️ Setup and Installation
Clone the repository:


1. Bash
git clone https://github.com/YOUR_USERNAME/Heartbeat.git
cd Heartbeat
Install dependencies:

2. Bash
pip install -r requirements.txt
Run the System:

3. Bash
python main.py



ID,Feature,Format,Description
1,Age,Integer,Patient's age in years
2,Gender,M / F,"Converted internally to binary (1 = Male, 0 = Female)"
3,Chest Pain,1 / 0,"Presence of frequent chest pain/angina (1 = Yes, 0 = No)"
4,Blood Pressure,Float,Resting blood pressure in mm Hg
5,Cholesterol,Float,Serum cholesterol level in mg/dl
6,Diabetes,1 / 0,"Clinical diagnosis of Diabetes (1 = Yes, 0 = No)"
7,BMI,Float,Body Mass Index
8,Smoking,1 / 0,"Current smoking status (1 = Yes, 0 = No)"
9,Family History,1 / 0,"Genetic history of heart disease (1 = Yes, 0 = No)"
