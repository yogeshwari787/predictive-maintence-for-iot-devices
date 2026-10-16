"""
src/train.py
------------
Trains a Random Forest classifier for predictive maintenance failure detection.
Handles class imbalance, hyperparameter tuning, and model persistence.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

from src.preprocess import run_preprocessing

MODEL_PATH   = os.path.join("models", "predictive_maintenance_model.pkl")
METRICS_PATH = os.path.join("outputs", "model_metrics.json")


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest with class_weight balanced for imbalanced data."""
    print("\n[*] Training Random Forest Classifier...")
    print("    This handles the 93/7 class imbalance automatically via class_weight='balanced'")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",  # Critical: handles imbalanced failure/no-failure data
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"[✓] Model trained! Trees: {model.n_estimators}, Features used: {model.n_features_in_}")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list) -> dict:
    """Evaluate model performance with all key classification metrics."""
    print("\n[*] Evaluating model...")

    y_pred     = model.predict(X_test)
    y_proba    = model.predict_proba(X_test)[:, 1]

    acc        = accuracy_score(y_test, y_pred)
    prec       = precision_score(y_test, y_pred, zero_division=0)
    rec        = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_proba)
    cm         = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(roc_auc, 4),
        "confusion_matrix": cm
    }

    # Feature importances
    importances = model.feature_importances_
    feat_imp = dict(zip(feature_names, np.round(importances, 4).tolist()))
    top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    metrics["top_features"] = top_features

    print("\n" + "=" * 55)
    print("       PREDICTIVE MAINTENANCE MODEL EVALUATION")
    print("=" * 55)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%   ← (critical: catch real failures)")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("=" * 55)
    print("\n  Top 5 Predictive Features:")
    for name, imp in top_features:
        bar = "█" * int(imp * 100)
        print(f"    {name:<20} {imp:.4f}  {bar}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
    print("=" * 55)

    return metrics


def save_model(model, path: str = MODEL_PATH):
    """Persist trained model to disk."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print(f"[✓] Model saved to: {path}")


def save_metrics(metrics: dict, path: str = METRICS_PATH):
    """Save evaluation metrics to JSON for reporting."""
    os.makedirs("outputs", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to: {path}")


def run_training():
    """Full training pipeline."""
    from src.preprocess import FEATURE_COLS
    X_train, X_test, y_train, y_test, scaler, df = run_preprocessing()
    model   = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, FEATURE_COLS)
    save_model(model)
    save_metrics(metrics)
    return model, metrics, X_test, y_test


if __name__ == "__main__":
    run_training()
