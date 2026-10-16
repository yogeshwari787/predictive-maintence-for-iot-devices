"""
main.py
-------
Master orchestration script for the AI-Powered Predictive Maintenance project.
Run this ONCE to execute the full pipeline end-to-end:

    python main.py

Pipeline:
    1. Generate / load dataset
    2. Preprocess + feature engineering
    3. Train Random Forest model
    4. Evaluate and save metrics
    5. Generate all 7 visualizations
    6. Run live IoT simulation
    7. Print final summary report
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(text: str):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────
banner("PHASE 1 — Dataset Generation")
# ─────────────────────────────────────────────────────────────────
data_path = os.path.join("data", "iot_sensor_data.csv")
if not os.path.exists(data_path):
    print("[*] Dataset not found. Generating fresh IoT sensor data...")
    from data.generate_dataset import generate_sensor_data
    os.makedirs("data", exist_ok=True)
    df_raw = generate_sensor_data(n_samples=10000)
    df_raw.to_csv(data_path, index=False)
else:
    print(f"[✓] Dataset already exists: {data_path}")


# ─────────────────────────────────────────────────────────────────
banner("PHASE 2–4 — Preprocessing, Feature Engineering, Scaling")
# ─────────────────────────────────────────────────────────────────
from src.preprocess import run_preprocessing, FEATURE_COLS
X_train, X_test, y_train, y_test, scaler, df = run_preprocessing()


# ─────────────────────────────────────────────────────────────────
banner("PHASE 5 — Model Training (Random Forest)")
# ─────────────────────────────────────────────────────────────────
from src.train import train_model, evaluate_model, save_model, save_metrics
model   = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test, FEATURE_COLS)
save_model(model)
save_metrics(metrics)


# ─────────────────────────────────────────────────────────────────
banner("PHASE 6 — Evaluation Artifacts")
# ─────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Save predictions CSV for notebooks
import pandas as pd
os.makedirs("outputs", exist_ok=True)
pred_df = pd.DataFrame({
    "actual":      y_test,
    "predicted":   y_pred,
    "probability": np.round(y_proba, 4)
})
pred_df.to_csv(os.path.join("outputs", "predictions.csv"), index=False)
print(f"[✓] Predictions saved to: outputs/predictions.csv")


# ─────────────────────────────────────────────────────────────────
banner("PHASE 7 — Visualization (7 Charts)")
# ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no GUI needed)
from src.visualize import run_all_visualizations
run_all_visualizations(df, model, scaler, FEATURE_COLS, y_test, y_pred, y_proba)


# ─────────────────────────────────────────────────────────────────
banner("PHASE 8 — Live IoT Simulation")
# ─────────────────────────────────────────────────────────────────
from src.simulate import run_simulation
run_simulation(n_readings=30, delay_sec=0.0)   # delay=0 for fast demo in CI


# ─────────────────────────────────────────────────────────────────
banner("PHASE 9 — Final Summary Report")
# ─────────────────────────────────────────────────────────────────
total_samples = len(df)
failures      = int(df["failure"].sum())
failure_rate  = round(failures / total_samples * 100, 2)

print(f"""
  ┌─────────────────────────────────────────────────┐
  │     AI-POWERED PREDICTIVE MAINTENANCE SYSTEM    │
  │              FINAL PROJECT SUMMARY              │
  └─────────────────────────────────────────────────┘

  Dataset
  ├── Total Records   : {total_samples:,}
  ├── Failure Records : {failures:,} ({failure_rate}%)
  └── Features Used   : {len(FEATURE_COLS)} (incl. 4 engineered)

  Model Performance
  ├── Accuracy        : {metrics['accuracy']*100:.2f}%
  ├── Precision       : {metrics['precision']*100:.2f}%
  ├── Recall          : {metrics['recall']*100:.2f}%
  ├── F1-Score        : {metrics['f1_score']*100:.2f}%
  └── ROC-AUC         : {metrics['roc_auc']:.4f}

  Artifacts Generated
  ├── models/predictive_maintenance_model.pkl
  ├── models/scaler.pkl
  ├── outputs/predictions.csv
  ├── outputs/model_metrics.json
  └── images/  (7 visualizations)

  To run Flask API:
  └── python app.py   →  http://127.0.0.1:5000
""")

print("✅  Pipeline complete! Your project is ready for GitHub.\n")
