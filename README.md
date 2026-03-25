
# 💓 Heartbeat AI: Predictive Health Analysis
**By Prince Maurya** | *B.Tech First Year, IILM University*

---

### 🚀 The Mission
This is my **first engineering project**. It explores the intersection of Healthcare and Artificial Intelligence by using an **XGBoost Classifier** to predict heart health risks based on patient metrics.

### 🛠️ Tech Stack
* **Language:** Python 3.x
* **AI Model:** XGBoost (Extreme Gradient Boosting)
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn

---

### 💻 Fast-Track Installation (VS Code)

To get this project running in your local environment, follow these steps:

**1. Clone & Setup Environment**
Install all dependencies in one go:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

**2. Data & Model Pipeline**
Run these scripts in order to see the AI in action:
* **Step A:** Generate the synthetic dataset.
  ```bash
  python generate_data.py
  ```
* **Step B:** Run the core prediction engine.
  ```bash
  python main.py
  ```

---

### 📂 Repository Architecture
* `main.py`: The "brain" of the project—handles the prediction logic.
* `generate_data.py`: Creates the `modern_patients.db` and `heart_data.csv` files.
* `evaluate_model.py`: Generates accuracy reports and performance graphs.
* `requirements.txt`: The blueprint for your Python environment.

---

### 🌟 Advanced Tip: Virtual Environments
For a more "pro" way to manage your project, instead of installing libraries globally, use a **Virtual Environment**. This keeps your project clean:

```bash
# 1. Create the environment
python -m venv venv

# 2. Activate it
# On Windows:
.\venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt
```

---
**Prince Maurya**
*IILM University, Greater Noida*
