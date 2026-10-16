"""
src/simulate.py
---------------
Virtual IoT simulation: streams live sensor readings through the trained model
and fires alerts when failure probability crosses the threshold.
Simulates what a real edge device (Raspberry Pi / ESP32) would do.
"""

import time
import random
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

ALERT_THRESHOLD = 0.50   # 50% failure probability triggers alert


def generate_live_reading(degraded: bool = False) -> dict:
    """Simulate one IoT sensor reading. Degraded=True injects fault conditions."""
    if degraded:
        # Simulate a degrading machine: high temp, vibration, wear
        air_temp        = random.uniform(25, 30)
        motor_temp      = random.uniform(76, 95)   # Overheating
        vibration       = random.uniform(7.0, 12.0) # High vibration (bearing defect)
        current         = random.uniform(17.0, 25.0) # Overload
        rotational_speed= random.uniform(1200, 1380) # RPM drop (overload)
        torque          = random.uniform(55, 80)
        pressure        = random.uniform(38, 43)    # Low coolant pressure
        tool_wear       = random.uniform(200, 240)  # High wear
    else:
        air_temp        = random.uniform(22, 28)
        motor_temp      = random.uniform(45, 68)
        vibration       = random.uniform(1.0, 4.0)
        current         = random.uniform(10.5, 14.5)
        rotational_speed= random.uniform(1380, 1620)
        torque          = random.uniform(28, 48)
        pressure        = random.uniform(46, 58)
        tool_wear       = random.uniform(10, 180)

    # Engineered features (same as training pipeline)
    temp_delta    = motor_temp - air_temp
    power_kw      = (torque * rotational_speed * 2 * np.pi / 60) / 1000.0
    torque_x_wear = torque * tool_wear
    vibration_sq  = vibration ** 2

    return {
        "air_temperature":  round(air_temp, 2),
        "temperature":       round(motor_temp, 2),
        "vibration":         round(vibration, 3),
        "current":           round(current, 2),
        "rotational_speed":  round(rotational_speed, 1),
        "torque":            round(torque, 2),
        "pressure":          round(pressure, 2),
        "tool_wear":         round(tool_wear, 1),
        "temp_delta":        round(temp_delta, 2),
        "power_kw":          round(power_kw, 3),
        "torque_x_wear":     round(torque_x_wear, 2),
        "vibration_sq":      round(vibration_sq, 3),
    }


def run_simulation(n_readings: int = 30, delay_sec: float = 0.4):
    """
    Stream n_readings sensor samples through the trained model.
    First 20 are 'normal', last 10 simulate a degrading/failing machine.
    """
    print("\n" + "=" * 65)
    print("  🏭  AI-POWERED PREDICTIVE MAINTENANCE  |  LIVE SIMULATION")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[✗] Model or scaler not found. Please run `main.py` first.")
        return

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[✓] Model loaded: {MODEL_PATH}")
    print(f"    Monitoring {n_readings} sensor readings...\n")

    header = f"{'#':>3} | {'Temp':>6} | {'Vib':>5} | {'Curr':>6} | {'P(Fail)':>8} | STATUS"
    print(header)
    print("-" * 65)

    alerts = 0
    for i in range(1, n_readings + 1):
        # Last 10 readings simulate a degrading machine
        degraded = (i > n_readings - 10)
        reading  = generate_live_reading(degraded=degraded)

        # Scale & predict
        import pandas as pd
        X = pd.DataFrame([reading])[FEATURE_COLS]
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0][1]
        pred  = int(proba >= ALERT_THRESHOLD)

        if pred == 1:
            alerts += 1
            status = "⚠️  FAILURE ALERT  ⚠️"
        else:
            status = "✅ Normal"

        print(
            f"{i:>3} | {reading['temperature']:>6.1f}°C | "
            f"{reading['vibration']:>5.2f} | "
            f"{reading['current']:>5.1f}A | "
            f"{proba:>8.4f} | {status}"
        )
        time.sleep(delay_sec)

    print("-" * 65)
    print(f"\n  Simulation complete. Total alerts fired: {alerts}/{n_readings}")
    if alerts > 0:
        print("  📧 In production: engineers would receive instant SMS/email alerts.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_simulation(n_readings=30, delay_sec=0.3)
