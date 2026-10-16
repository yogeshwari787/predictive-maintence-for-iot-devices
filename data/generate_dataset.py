"""
data/generate_dataset.py
------------------------
Generates a realistic industrial IoT machine sensor dataset (10,000 records).
Follows industrial standards (inspired by UCI AI4I 2020 Predictive Maintenance & NASA C-MAPSS).
Simulates thermal, mechanical, electrical, and degradation wear physics.
"""

import numpy as np
import pandas as pd
import os

def generate_sensor_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    print(f"[*] Generating {n_samples} synthetic industrial IoT sensor telemetry records...")
    
    # 1. Base telemetry parameters under normal operating conditions
    # Temperature: Motor casing temp in Celsius (normal: 40 - 70 C)
    air_temp = np.random.normal(loc=25.0, scale=2.5, size=n_samples)
    motor_temp = air_temp + np.random.normal(loc=30.0, scale=4.0, size=n_samples)
    
    # Vibration: Velocity in mm/s (ISO 10816 standards: normal < 4.5 mm/s)
    vibration = np.random.gamma(shape=2.5, scale=1.0, size=n_samples)
    
    # Current: Motor operating current in Amperes (normal: 10 - 15 A)
    current = np.random.normal(loc=12.5, scale=1.5, size=n_samples)
    
    # Rotational speed in RPM (normal: 1350 - 1700 RPM)
    rotational_speed = np.random.normal(loc=1500, scale=120, size=n_samples)
    
    # Torque in Nm (inversely related to speed with noise)
    power_kw = 5.5 # rated 5.5 kW motor
    torque = (power_kw * 1000) / (rotational_speed * (2 * np.pi / 60)) + np.random.normal(0, 3, size=n_samples)
    
    # Pressure: Coolant/Lubrication line pressure in PSI (normal: 45 - 60 PSI)
    pressure = np.random.normal(loc=50.0, scale=4.0, size=n_samples)
    
    # Tool Wear / Cumulative operation hours (0 - 240 minutes per shift)
    tool_wear = np.random.uniform(0, 240, size=n_samples)
    
    # Device IDs
    devices = [f"IOT-MTR-{i:03d}" for i in range(1, 11)]
    device_id = np.random.choice(devices, size=n_samples)
    
    # 2. Physics-based Failure Modes Injection (~3.5% industrial incident rate)
    failure = np.zeros(n_samples, dtype=int)
    failure_mode = np.array(["No Failure"] * n_samples, dtype=object)
    
    for i in range(n_samples):
        # Rule 1: Tool Wear Failure (TWF) - excessive tool wear leads to chatter & failure
        if tool_wear[i] > 200 and np.random.rand() < 0.28:
            failure[i] = 1
            failure_mode[i] = "Tool Wear Failure"
            vibration[i] += np.random.uniform(3.0, 6.0)
            
        # Rule 2: Heat Dissipation Failure (HDF) - casing temp spikes > 78C & insufficient cooling
        elif motor_temp[i] > 72.0 and pressure[i] < 44.0 and np.random.rand() < 0.65:
            failure[i] = 1
            failure_mode[i] = "Heat Dissipation Failure"
            motor_temp[i] += np.random.uniform(10.0, 22.0)
            
        # Rule 3: Power / Electrical Overload Failure (PWF) - power spike with high current
        elif current[i] > 16.5 and torque[i] > 55.0 and np.random.rand() < 0.55:
            failure[i] = 1
            failure_mode[i] = "Power Overload Failure"
            current[i] += np.random.uniform(5.0, 12.0)
            
        # Rule 4: Mechanical Overstrain Failure (OSF) - extreme torque & tool wear
        elif (torque[i] * tool_wear[i]) > 11000 and np.random.rand() < 0.45:
            failure[i] = 1
            failure_mode[i] = "Overstrain Failure"
            vibration[i] += np.random.uniform(4.0, 8.0)
            
        # Rule 5: Severe Vibration Imbalance (bearing defect)
        elif vibration[i] > 6.5 and np.random.rand() < 0.70:
            failure[i] = 1
            failure_mode[i] = "Vibration Imbalance"
            vibration[i] += np.random.uniform(2.5, 5.0)

    # 3. Create DataFrame
    df = pd.DataFrame({
        "device_id": device_id,
        "air_temperature": np.round(air_temp, 2),
        "temperature": np.round(motor_temp, 2), # Motor Temperature (C)
        "vibration": np.round(vibration, 3),    # mm/s
        "current": np.round(current, 2),        # Amperes
        "rotational_speed": np.round(rotational_speed, 1), # RPM
        "torque": np.round(torque, 2),          # Nm
        "pressure": np.round(pressure, 2),      # PSI
        "tool_wear": np.round(tool_wear, 1),    # Minutes
        "failure": failure,                     # Binary target
        "failure_mode": failure_mode            # Multi-class target description
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sensor_data(n_samples=10000)
    output_path = os.path.join("data", "iot_sensor_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"[✓] Dataset successfully generated and saved to: {output_path}")
    print(f"[✓] Total Records: {len(df)}")
    print(f"[✓] Failure Distribution:\n{df['failure'].value_counts(normalize=True).mul(100).round(2)}%")
    print(f"[✓] Failure Modes Breakdown:\n{df['failure_mode'].value_counts()}")
