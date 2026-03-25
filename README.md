### **Student Profile**
**[Prince Maurya]**
*First-Year B.Tech Student at IILM University*
Focused on leveraging Machine Learning to solve real-world healthcare challenges. Currently exploring Python, XGBoost, and Data Engineering to build intelligent, patient-centric applications.

---

### **Project: Heartbeat (Heartdisease Risk Predictor)**
**Concept:** A diagnostic tool that predicts cardiovascular risk using the XGBoost gradient boosting algorithm.
**Key Features:**
* **High-Performance Engine:** Uses a pre-trained XGBoost model to analyze non-linear health patterns.
* **Data Persistence:** Integrated with a SQLite database to log and track patient history over time.
* **Clinical Inputs:** Processes 9 critical metrics, including Age, Blood Pressure, BMI, and Family History.
* **Accuracy:** Achieved an initial baseline accuracy of ~99.8% during model testing[cite: 1].

---

### **Quick Start (VS Code)**
1.  **Install Libraries:**
    `pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib`
2.  **Generate Data:**
    `python generate_data.py`
3.  **Run Predictor:**
    `python main.py`
