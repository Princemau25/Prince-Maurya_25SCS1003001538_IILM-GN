# Heartbeat — Clinical Risk Prediction System

A machine-learning-driven diagnostic tool that predicts the likelihood of cardiovascular disease using an XGBoost classifier trained on nine clinical parameters.

## Project Structure

| File | Purpose |
|------|---------|
| `generate_data.py` | Simulates patient data, trains the XGBoost model, saves `xgboost_model.pkl` |
| `evaluate_model.py` | Loads the model and reports accuracy, precision, recall, and F1 metrics |
| `main.py` | Interactive CLI — collects patient inputs, predicts risk, stores in SQLite |
| `requirements.txt` | Python dependencies |

## Input Features

The model analyses nine clinical indicators:

| # | Feature | Type |
|---|---------|------|
| 1 | Age | Integer (25–100) |
| 2 | Gender | Binary (0 = Female, 1 = Male) |
| 3 | Chest Pain | Binary (0 = No, 1 = Yes) |
| 4 | Blood Pressure (mm Hg) | Float |
| 5 | Cholesterol (mg/dl) | Float |
| 6 | Diabetes | Binary (0 = No, 1 = Yes) |
| 7 | BMI | Float |
| 8 | Smoking | Binary (0 = No, 1 = Yes) |
| 9 | Family History | Binary (0 = No, 1 = Yes) |

## How to Run (VS Code / Terminal)

### Step 1 — Open the project folder in VS Code

```
File > Open Folder…  →  select the Heartbeat folder
```

### Step 2 — Open the integrated terminal

```
Terminal > New Terminal   (or Ctrl + `)
```

### Step 3 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4 — Install required libraries

```bash
pip install -r requirements.txt
```

### Step 5 — Generate data and train the model

```bash
python generate_data.py
```

This creates `modern_patients.csv` and `xgboost_model.pkl`.

### Step 6 — (Optional) Evaluate model performance

```bash
python evaluate_model.py
```

### Step 7 — Run the prediction tool

```bash
python main.py
```

Follow the on-screen prompts to enter patient data. Each prediction is saved to `modern_patients.db`.

## Data Persistence

All patient records and predictions are stored in a local SQLite database (`modern_patients.db`), which persists across sessions without requiring external cloud hosting.
