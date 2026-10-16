"""
app.py  (Flask REST API)
------------------------
Deploys the trained predictive maintenance model as a REST API.
Endpoint: POST /predict
Accept JSON with sensor readings, returns failure prediction.

Run: python app.py
Test: http://127.0.0.1:5000/health
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import os
import json

app = Flask(__name__)

MODEL_PATH  = os.path.join("models", "predictive_maintenance_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

FEATURE_COLS = [
    "air_temperature", "temperature", "vibration",
    "current", "rotational_speed", "torque", "pressure", "tool_wear",
    "temp_delta", "power_kw", "torque_x_wear", "vibration_sq"
]

# Load model & scaler at startup
model, scaler = None, None

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[✓] Model and scaler loaded successfully.")
    else:
        print("[✗] Model not found. Run main.py first to train the model.")


# ── HTML UI Landing Page ───────────────────────────────────────────────────────
LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Predictive Maintenance API</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif;
         display: flex; justify-content: center; align-items: center; min-height: 100vh; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
          padding: 36px; max-width: 700px; width: 90%; }
  h1 { font-size: 1.6rem; color: #58a6ff; margin-bottom: 8px; }
  .badge { display: inline-block; background: #238636; color: #fff;
           font-size: 0.7rem; padding: 2px 10px; border-radius: 20px; margin-bottom: 20px; }
  pre { background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; font-size: 0.85rem; overflow-x: auto; color: #e6edf3; }
  .endpoint { color: #f0883e; font-weight: 600; }
  .method { color: #3fb950; font-weight: 700; font-size: 0.9rem; }
  hr { border: none; border-top: 1px solid #21262d; margin: 24px 0; }
  .tip { font-size: 0.82rem; color: #8b949e; margin-top: 12px; }
</style>
</head>
<body>
<div class="card">
  <h1>🏭 AI Predictive Maintenance API</h1>
  <span class="badge">● API LIVE</span>
  <p>Send real-time IoT sensor data and get instant failure predictions powered by Random Forest ML.</p>
  <hr>
  <p><span class="method">POST</span> <span class="endpoint">/predict</span></p>
  <pre>{
  "air_temperature":  26.0,
  "temperature":      72.5,
  "vibration":        3.8,
  "current":          13.2,
  "rotational_speed": 1500.0,
  "torque":           42.0,
  "pressure":         51.0,
  "tool_wear":        120.0
}</pre>
  <hr>
  <p><span class="method">GET</span> <span class="endpoint">/health</span> — API health check</p>
  <p class="tip">💡 Test with Postman, curl, or any REST client. No auth required (dev mode).</p>
</div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(LANDING_HTML)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "api":          "AI Predictive Maintenance for IoT Devices",
        "version":      "1.0.0"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with raw sensor readings.
    Returns failure prediction, probability, and risk level.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run main.py to train first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400

    required_fields = [
        "air_temperature", "temperature", "vibration",
        "current", "rotational_speed", "torque", "pressure", "tool_wear"
    ]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    try:
        # Compute engineered features
        air_temp   = float(data["air_temperature"])
        motor_temp = float(data["temperature"])
        vibration  = float(data["vibration"])
        current    = float(data["current"])
        rot_speed  = float(data["rotational_speed"])
        torque     = float(data["torque"])
        pressure   = float(data["pressure"])
        tool_wear  = float(data["tool_wear"])

        temp_delta    = motor_temp - air_temp
        power_kw      = (torque * rot_speed * 2 * 3.14159 / 60) / 1000.0
        torque_x_wear = torque * tool_wear
        vibration_sq  = vibration ** 2

        features = np.array([[
            air_temp, motor_temp, vibration, current,
            rot_speed, torque, pressure, tool_wear,
            temp_delta, power_kw, torque_x_wear, vibration_sq
        ]])

        X_scaled = scaler.transform(features)
        proba    = float(model.predict_proba(X_scaled)[0][1])
        pred     = int(proba >= 0.50)

        risk_level = (
            "CRITICAL" if proba >= 0.80 else
            "HIGH"     if proba >= 0.50 else
            "MODERATE" if proba >= 0.25 else
            "NORMAL"
        )

        return jsonify({
            "prediction":          pred,
            "failure_probability": round(proba, 4),
            "risk_level":          risk_level,
            "status":              "Machine failure predicted!" if pred == 1 else "Machine running normally.",
            "alert":               pred == 1
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400


if __name__ == "__main__":
    load_model()
    print("\n[🚀] Starting Flask API server...")
    print("[📡] API running at: http://127.0.0.1:8080")
    print("[💡] Send POST /predict with sensor JSON to get predictions\n")
    app.run(debug=True, host="0.0.0.0", port=8080)
