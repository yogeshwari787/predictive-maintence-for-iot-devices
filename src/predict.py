"""
src/predict.py
--------------
Single-inference utility: takes raw sensor readings and returns failure prediction + probability.
Used internally and by the Flask API.
"""

import numpy as np
import joblib
import os

MODEL_PATH  = os.path.join("models", "predictive_maintenance_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

FEATURE_COLS = [
    "air_temperature", "temperature", "vibration",
    "current", "rotational_speed", "torque", "pressure", "tool_wear",
    "temp_delta", "power_kw", "torque_x_wear", "vibration_sq"
]


def load_artifacts():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_failure(sensor_input: dict) -> dict:
    """
    Given a dict of raw sensor values, compute engineered features
    and return prediction + failure probability.

    Required keys in sensor_input:
        air_temperature, temperature, vibration, current,
        rotational_speed, torque, pressure, tool_wear
    """
    model, scaler = load_artifacts()

    # Compute engineered features from raw values
    sensor_input["temp_delta"]    = sensor_input["temperature"] - sensor_input["air_temperature"]
    sensor_input["power_kw"]      = (
        sensor_input["torque"] * sensor_input["rotational_speed"] * 2 * 3.14159 / 60
    ) / 1000.0
    sensor_input["torque_x_wear"] = sensor_input["torque"] * sensor_input["tool_wear"]
    sensor_input["vibration_sq"]  = sensor_input["vibration"] ** 2

    X = np.array([[sensor_input[f] for f in FEATURE_COLS]])
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0][1]
    pred  = int(proba >= 0.50)

    if proba >= 0.80:
        risk_level = "CRITICAL"
    elif proba >= 0.50:
        risk_level = "HIGH"
    elif proba >= 0.25:
        risk_level = "MODERATE"
    else:
        risk_level = "NORMAL"

    return {
        "prediction":         pred,
        "failure_probability": round(proba, 4),
        "risk_level":         risk_level,
        "status":             "Machine failure predicted!" if pred == 1 else "Machine running normally.",
    }


if __name__ == "__main__":
    # Quick test: simulate a degraded machine
    test_input = {
        "air_temperature":  26.0,
        "temperature":      82.5,   # Overheating
        "vibration":        9.2,    # High vibration
        "current":          19.3,   # Overload current
        "rotational_speed": 1280.0,
        "torque":           62.0,
        "pressure":         40.5,
        "tool_wear":        218.0,
    }
    result = predict_failure(test_input)
    print("\n--- Prediction Result ---")
    for k, v in result.items():
        print(f"  {k:<25}: {v}")
