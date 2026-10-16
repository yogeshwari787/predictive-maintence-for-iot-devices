"""
src/visualize.py
----------------
Generates publication-quality plots for GitHub proof:
  1. Sensor correlation heatmap
  2. Failure vs normal distribution plots
  3. Feature importances bar chart
  4. Confusion matrix heatmap
  5. ROC-AUC curve
  6. Time-series failure prediction simulation
  7. Real-time alert dashboard snapshot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
})

COLORS = {
    "normal":    "#238636",   # green
    "failure":   "#f85149",   # red
    "accent":    "#58a6ff",   # blue
    "accent2":   "#d2a8ff",   # purple
    "warn":      "#e3b341",   # yellow
    "bg_card":   "#161b22",
}

OUTPUTS_DIR = "outputs"
IMAGES_DIR  = "images"


def ensure_dirs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR,  exist_ok=True)


# ── 1. Sensor Distributions ───────────────────────────────────────────────────
def plot_sensor_distributions(df: pd.DataFrame):
    print("[*] Plotting sensor distributions by failure class...")
    sensors = ["temperature", "vibration", "current", "pressure", "torque", "tool_wear"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IoT Sensor Readings: Failure vs Normal Distribution", fontsize=16, y=1.01,
                 color="#c9d1d9", fontweight="bold")

    for ax, sensor in zip(axes.flatten(), sensors):
        data_normal  = df[df["failure"] == 0][sensor]
        data_failure = df[df["failure"] == 1][sensor]
        ax.hist(data_normal,  bins=50, alpha=0.7, color=COLORS["normal"],  label="Normal",  density=True)
        ax.hist(data_failure, bins=50, alpha=0.8, color=COLORS["failure"], label="Failure", density=True)
        ax.set_title(sensor.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "01_sensor_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── 2. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    print("[*] Plotting feature correlation heatmap...")
    numeric_cols = ["temperature", "vibration", "current", "pressure",
                    "rotational_speed", "torque", "tool_wear",
                    "temp_delta", "power_kw", "torque_x_wear", "failure"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
                cmap=cmap, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap\n(Correlation with Failure Column)",
                 color="#c9d1d9", fontsize=14, pad=14)
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "02_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── 3. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list):
    print("[*] Plotting feature importances...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i].replace("_", "\n") for i in indices]
    sorted_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(sorted_names, sorted_imps,
                  color=[COLORS["accent"] if i < 3 else COLORS["accent2"] for i in range(len(sorted_names))],
                  edgecolor="#30363d", linewidth=0.8, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, sorted_imps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#c9d1d9")

    ax.set_title("Random Forest – Feature Importance for Failure Prediction")
    ax.set_ylabel("Importance Score (Gini Impurity Reduction)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_ylim(0, sorted_imps.max() * 1.2)

    p1 = mpatches.Patch(color=COLORS["accent"],  label="Top 3 Features")
    p2 = mpatches.Patch(color=COLORS["accent2"], label="Other Features")
    ax.legend(handles=[p1, p2], loc="upper right")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "03_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── 4. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
    print("[*] Plotting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"],
                linewidths=1.5, linecolor="#30363d",
                annot_kws={"size": 18, "weight": "bold"},
                cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix – Failure Detection")

    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.18, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
            ha="center", transform=ax.transAxes, fontsize=9, color="#8b949e")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "04_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── 5. ROC-AUC Curve ──────────────────────────────────────────────────────────
def plot_roc_curve(y_test, y_proba):
    print("[*] Plotting ROC-AUC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["accent"], lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=COLORS["accent"])
    ax.plot([0, 1], [0, 1], color="#8b949e", lw=1.2, linestyle="--", label="Random Baseline")
    ax.set_title("ROC-AUC Curve – Failure Detection Performance")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "05_roc_auc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── 6. Time-Series Failure Prediction Simulation ──────────────────────────────
def plot_timeseries_prediction(df: pd.DataFrame, model, scaler, feature_names: list):
    print("[*] Plotting time-series failure prediction simulation...")

    # Use last 200 records as a streaming simulation
    sample = df.tail(300).copy().reset_index(drop=True)
    X_sim = scaler.transform(sample[feature_names])
    proba = model.predict_proba(X_sim)[:, 1]
    pred  = model.predict(X_sim)
    time_axis = range(len(sample))

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    fig.suptitle("Real-Time IoT Sensor Monitoring & Failure Prediction Simulation",
                 fontsize=15, y=1.01, color="#c9d1d9")

    # Panel 1: Temperature
    axes[0].plot(time_axis, sample["temperature"], color=COLORS["warn"], lw=1.5, label="Motor Temp (°C)")
    axes[0].axhline(72, color=COLORS["failure"], ls="--", lw=1, alpha=0.7, label="Danger Threshold (72°C)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.4)

    # Panel 2: Vibration
    axes[1].plot(time_axis, sample["vibration"], color=COLORS["accent2"], lw=1.5, label="Vibration (mm/s)")
    axes[1].axhline(6.5, color=COLORS["failure"], ls="--", lw=1, alpha=0.7, label="ISO 10816 Limit (6.5 mm/s)")
    axes[1].set_ylabel("Vibration (mm/s)")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.4)

    # Panel 3: Failure Probability
    axes[2].fill_between(time_axis, proba, alpha=0.25, color=COLORS["failure"])
    axes[2].plot(time_axis, proba, color=COLORS["failure"], lw=1.8, label="Failure Probability")
    axes[2].axhline(0.5, color=COLORS["warn"], ls="--", lw=1.2, alpha=0.8, label="Alert Threshold (0.5)")
    axes[2].set_ylabel("P(Failure)")
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].grid(True, alpha=0.4)

    # Panel 4: Alert Flags
    alert_idx = [i for i, p in enumerate(pred) if p == 1]
    normal_idx = [i for i, p in enumerate(pred) if p == 0]
    axes[3].scatter(normal_idx, [0]*len(normal_idx), color=COLORS["normal"],
                    s=18, label="Normal", marker="o", alpha=0.6)
    axes[3].scatter(alert_idx, [1]*len(alert_idx),  color=COLORS["failure"],
                    s=40, label="⚠ FAILURE PREDICTED", marker="X", zorder=5)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["Normal", "ALERT"])
    axes[3].set_ylabel("System Status")
    axes[3].set_xlabel("Time Steps (sensor readings)")
    axes[3].legend(loc="upper left", fontsize=9)
    axes[3].grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "06_timeseries_prediction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── 7. Failure Mode Breakdown (Pie) ───────────────────────────────────────────
def plot_failure_breakdown(df: pd.DataFrame):
    print("[*] Plotting failure mode breakdown...")
    breakdown = df[df["failure"]==1]["failure_mode"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 7))
    wedge_colors = [COLORS["failure"], COLORS["warn"], COLORS["accent"], COLORS["accent2"], COLORS["normal"]]
    wedges, texts, autotexts = ax.pie(
        breakdown.values, labels=breakdown.index,
        autopct="%1.1f%%", colors=wedge_colors[:len(breakdown)],
        pctdistance=0.82, startangle=140,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=2.5)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title(f"Failure Mode Breakdown\n(Total Failures: {breakdown.sum()})")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "07_failure_mode_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ── Master Runner ─────────────────────────────────────────────────────────────
def run_all_visualizations(df, model, scaler, feature_names, y_test, y_pred, y_proba):
    ensure_dirs()
    plot_sensor_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_importance(model, feature_names)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_timeseries_prediction(df, model, scaler, feature_names)
    plot_failure_breakdown(df)
    print("\n[✓] All 7 visualizations generated in images/")
