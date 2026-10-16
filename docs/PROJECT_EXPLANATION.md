docs/PROJECT_EXPLANATION.md
# 📖 AI-Powered Predictive Maintenance — Complete Project Explanation

## What Is Predictive Maintenance?

**Simple Language:**
Machines break down. When they break without warning, factories shut down, production stops, and companies lose millions. Predictive maintenance uses AI to watch machine health (temperature, vibrations, electrical current) and predict failures *before* they happen — like a health check-up for machines.

**Technical Language:**
Predictive Maintenance (PdM) is an AI/ML technique that uses time-series sensor data from IoT devices to build classification or anomaly detection models that forecast machine Remaining Useful Life (RUL) or binary failure events, enabling condition-based maintenance interventions.

---

## Why Is This Important?

| Industry | Problem Without PdM | Benefit With PdM |
|---|---|---|
| Manufacturing | Assembly lines stop unexpectedly | 15–40% reduction in unplanned downtime |
| Power Plants | Turbine failure = city blackout | Early bearing/thermal fault detection |
| Aviation | Engine fault mid-flight | Predict jet engine degradation |
| Automotive | Robot arm halts entire plant | Monitor welding robot health |
| Oil & Gas | Pipeline pressure failure | Predict valve and pump failures |

**Market Size:** $7.85B (2022) → $60.13B (2030) growing at 29.5% CAGR

---

## Full System Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    IoT SENSOR LAYER                         │
│  Temperature │ Vibration │ Current │ Pressure │ RPM │ Torque │
└────────────────────────┬────────────────────────────────────┘
                         │ Raw Telemetry (CSV / MQTT stream)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING MODULE                      │
│  - Handle missing values                                    │
│  - Clip outliers (IQR method)                               │
│  - Feature engineering (temp_delta, power_kw, etc.)        │
│  - StandardScaler normalization                             │
└────────────────────────┬────────────────────────────────────┘
                         │ Clean, scaled features
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML MODEL LAYER                            │
│  Random Forest Classifier (300 trees)                       │
│  class_weight="balanced" → handles 93/7 imbalance           │
│  Output: P(failure) = 0.0 – 1.0                             │
└────────────────────────┬────────────────────────────────────┘
                         │ Failure probability
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  ALERT / DECISION LAYER                     │
│  P ≥ 0.50  → HIGH RISK: Trigger maintenance alert          │
│  P ≥ 0.80  → CRITICAL: Emergency shutdown signal           │
│  P < 0.25  → NORMAL: Continue operation                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                VISUALIZATION & REPORTING                    │
│  - Time-series dashboard                                    │
│  - Confusion matrix / ROC-AUC                               │
│  - Feature importance chart                                 │
│  - Failure mode breakdown                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack Comparison (3 Options)

| | **Option A (Easy)** | **Option B (Intermediate — SELECTED)** | **Option C (Advanced)** |
|---|---|---|---|
| Model | Logistic Regression | **Random Forest ✅** | LSTM Neural Network |
| Libraries | scikit-learn | **scikit-learn + pandas** | TensorFlow/Keras |
| Dataset | Simple tabular | **Synthetic IoT simulation** | NASA CMAPSS RUL |
| GPU Needed | No | **No** | Yes |
| Accuracy | ~80% | **~98%** | ~99%+ |
| GitHub Proof | Weak | **Strong ✅** | Very strong |
| Beginner Fit | Good | **Best overall ✅** | Hard |

**→ We selected Option B.** It gives ~98% accuracy, rich visualizations, a Flask API, and a live IoT simulation — all without requiring hardware or GPU.

---

## Failure Modes Simulated

| Failure Mode | Root Cause | Sensor Signature |
|---|---|---|
| Tool Wear Failure | Excessive cumulative wear (>200 min) | High vibration + tool_wear > 200 |
| Heat Dissipation Failure | Motor overheating + low coolant pressure | motor_temp > 72°C, pressure < 44 PSI |
| Power Overload Failure | Current surge + high torque | current > 16.5A, torque > 55 Nm |
| Overstrain Failure | Combined torque × wear product spike | torque × tool_wear > 11,000 |
| Vibration Imbalance | Bearing defect causing resonance | vibration > 6.5 mm/s |
