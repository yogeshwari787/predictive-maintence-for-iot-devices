"""
src/preprocess.py
-----------------
Handles all data loading, cleaning, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
DATA_PATH   = os.path.join("data", "iot_sensor_data.csv")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Feature columns used for training
FEATURE_COLS = [
    "air_temperature", "temperature", "vibration",
    "current", "rotational_speed", "torque", "pressure", "tool_wear",
    # Engineered features (created below)
    "temp_delta", "power_kw", "torque_x_wear", "vibration_sq"
]

TARGET_COL = "failure"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw sensor CSV."""
    print(f"[*] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[✓] Loaded {len(df)} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, outliers, and type issues."""
    print("[*] Cleaning data...")

    # Drop rows with all NaN sensor values (hardware dropout)
    sensor_cols = ["temperature", "vibration", "current", "pressure"]
    before = len(df)
    df = df.dropna(subset=sensor_cols)
    dropped = before - len(df)
    if dropped > 0:
        print(f"    Dropped {dropped} rows with missing sensor readings.")

    # Clip extreme outliers using IQR
    for col in ["vibration", "current", "temperature", "pressure"]:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    print(f"[✓] Clean dataset: {len(df)} rows")
    return df.reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create physics-inspired features from raw sensor readings."""
    print("[*] Engineering features...")

    # 1. Temperature delta: difference between motor and ambient temperature
    #    High delta = poor heat dissipation
    df["temp_delta"] = df["temperature"] - df["air_temperature"]

    # 2. Mechanical power (watts): Power = torque × angular velocity
    #    omega (rad/s) = RPM × 2π / 60
    df["power_kw"] = (df["torque"] * df["rotational_speed"] * 2 * np.pi / 60) / 1000.0

    # 3. Torque × Tool Wear: overstrain indicator (key industrial predictor)
    df["torque_x_wear"] = df["torque"] * df["tool_wear"]

    # 4. Vibration squared: accentuates high-vibration anomalies
    df["vibration_sq"] = df["vibration"] ** 2

    print(f"[✓] Feature engineering complete. Total features: {len(FEATURE_COLS)}")
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets and apply StandardScaler."""
    print("[*] Splitting and scaling data...")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler for Flask API reuse
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    print(f"[✓] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"[✓] Scaler saved to: {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


def run_preprocessing():
    """End-to-end preprocessing pipeline."""
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    return X_train, X_test, y_train, y_test, scaler, df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = run_preprocessing()
    print("\n--- Sample Engineered Data ---")
    print(df[FEATURE_COLS + [TARGET_COL]].head(5))
