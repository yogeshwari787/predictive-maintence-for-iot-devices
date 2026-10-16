docs/GITHUB_GUIDE.md
# 🚀 GitHub Upload & Proof Strategy — Step-by-Step

## Best Repository Name
```
AI-Predictive-Maintenance-IoT
```

## Repository Description
> AI-powered predictive maintenance system for IoT industrial devices. Uses Random Forest ML on synthetic sensor data (temperature, vibration, current) to detect machine failures with 98.65% accuracy. Includes Flask REST API, 7 visualizations, and live IoT simulation. | Python | scikit-learn | Pandas

## Tags / Topics to Add on GitHub
```
predictive-maintenance, iot, machine-learning, random-forest,
sklearn, flask, industrial-ai, sensor-data, python, data-science,
anomaly-detection, time-series, manufacturing, smart-manufacturing
```

---

## Day-by-Day Git Commit Plan (7-Day Proof Strategy)

### Day 1 — Project Setup
```bash
git init
git add requirements.txt .gitignore README.md
git commit -m "feat: initial project setup with requirements and structure"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Predictive-Maintenance-IoT.git
git push -u origin main
```

### Day 2 — Dataset Generation
```bash
git add data/generate_dataset.py data/iot_sensor_data.csv
git commit -m "data: add IoT sensor dataset generator (10K records, 5 failure modes)"
git push
```

### Day 3 — Preprocessing Pipeline
```bash
git add src/preprocess.py src/__init__.py
git commit -m "feat: preprocessing pipeline with feature engineering and StandardScaler"
git push
```

### Day 4 — Model Training
```bash
git add src/train.py models/ outputs/model_metrics.json
git commit -m "feat: Random Forest classifier (98.65% acc, 0.9968 AUC) with class balancing"
git push
```

### Day 5 — Evaluation & Visualizations
```bash
git add src/visualize.py outputs/predictions.csv images/
git commit -m "viz: add 7 publication-quality charts (ROC, confusion matrix, time-series)"
git push
```

### Day 6 — Flask API + Simulation
```bash
git add app.py src/predict.py src/simulate.py
git commit -m "feat: Flask REST API for real-time prediction + live IoT simulation"
git push
```

### Day 7 — Polish & Final Push
```bash
git add docs/ notebooks/ main.py
git commit -m "docs: add complete documentation, EDA notebook, and master pipeline"
git push
```

---

## How to Make Your GitHub Repo Professional

### 1. Enable GitHub Pages (optional but impressive)
- Settings → Pages → Deploy from `main` branch → `/docs` folder

### 2. Add a pinned repo to your profile
- Go to your GitHub profile → Customize → Pin this repo

### 3. Add a social preview image
- Settings → Social Preview → Upload `images/06_timeseries_prediction.png`

### 4. Create GitHub Issues as "milestones"
```
Issue #1: Add LSTM model for time-series forecasting
Issue #2: Deploy Flask API to AWS EC2 / Render.com
Issue #3: Connect to real MQTT IoT broker (EMQX)
Issue #4: Add Streamlit dashboard
```
→ Closing issues with commits shows recruiters you work like a real engineer.

### 5. Write a good README (see README.md already generated)

---

## curl Test Commands (for README proof)

### Health Check:
```bash
curl http://127.0.0.1:5000/health
```

### Normal Machine:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature":25,"temperature":55,"vibration":2.5,"current":12,"rotational_speed":1500,"torque":38,"pressure":52,"tool_wear":80}'
```

### Failing Machine:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature":26,"temperature":85,"vibration":9.5,"current":20,"rotational_speed":1250,"torque":65,"pressure":40,"tool_wear":220}'
```
