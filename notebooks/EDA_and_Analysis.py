"""
notebooks/EDA_and_Analysis.py
-----------------------------
Exploratory Data Analysis (EDA) script — run this as a standalone script
or copy cells into Jupyter Notebook for interactive exploration.

To open as Jupyter Notebook:
    jupyter notebook   →  create a new notebook  →  paste each section as a cell
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("   EDA — AI Predictive Maintenance for IoT Devices")
print("=" * 60)

# ── 1. Load Dataset ────────────────────────────────────────────────
df = pd.read_csv("data/iot_sensor_data.csv")

print(f"\n1️⃣  DATASET SHAPE: {df.shape}")
print("\nFirst 5 rows:")
print(df.head().to_string())

# ── 2. Basic Info ──────────────────────────────────────────────────
print("\n2️⃣  COLUMN TYPES & NULL COUNTS:")
print(df.info())

print("\n3️⃣  DESCRIPTIVE STATISTICS:")
print(df.describe().round(2).to_string())

# ── 3. Class Balance ───────────────────────────────────────────────
print("\n4️⃣  CLASS DISTRIBUTION (Failure Labels):")
counts = df["failure"].value_counts()
print(f"   Normal (0):  {counts[0]:,} samples  ({counts[0]/len(df)*100:.1f}%)")
print(f"   Failure (1): {counts[1]:,} samples  ({counts[1]/len(df)*100:.1f}%)")

print("\n5️⃣  FAILURE MODES BREAKDOWN:")
print(df["failure_mode"].value_counts().to_string())

# ── 4. Correlations ────────────────────────────────────────────────
print("\n6️⃣  TOP CORRELATIONS WITH 'failure' COLUMN:")
numeric_df = df.select_dtypes(include=np.number)
corr_with_failure = numeric_df.corr()["failure"].drop("failure").abs().sort_values(ascending=False)
print(corr_with_failure.to_string())

# ── 5. Device-level breakdown ──────────────────────────────────────
print("\n7️⃣  FAILURES PER DEVICE:")
device_failure = df.groupby("device_id")["failure"].agg(["sum", "count"])
device_failure.columns = ["failures", "total"]
device_failure["failure_rate_%"] = (device_failure["failures"] / device_failure["total"] * 100).round(2)
print(device_failure.sort_values("failure_rate_%", ascending=False).to_string())

# ── 6. Threshold Analysis (key stats at failure boundary) ──────────
print("\n8️⃣  SENSOR STATS: Normal vs Failure")
for col in ["temperature", "vibration", "current", "pressure", "tool_wear"]:
    n_mean = df[df["failure"]==0][col].mean()
    f_mean = df[df["failure"]==1][col].mean()
    print(f"   {col:<20}  Normal: {n_mean:7.2f}   |   Failure: {f_mean:7.2f}")

print("\n✅  EDA complete.")
