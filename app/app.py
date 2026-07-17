"""
app.py
------
Flask dashboard for the Predictive Server Failure Detection project.

Routes:
  /                -> Home: fleet overview (total/healthy/critical servers)
  /server/<id>     -> Server detail page with live-style metrics
  /predict         -> Manual prediction form
  /analytics       -> Model performance plots (ROC, confusion matrix, etc.)
"""

import os
import sys
import joblib
import pandas as pd
import shap
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

app = Flask(__name__)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "random_forest.pkl")
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "server_telemetry.csv")

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)  # built once at startup, reused per request

# Friendly display names for the explanation panel
FEATURE_DISPLAY_NAMES = {
    "cpu_usage": "CPU Usage",
    "memory_usage": "Memory Usage",
    "disk_usage": "Disk Usage",
    "temperature_c": "Temperature",
    "error_count": "Error Count",
    "power_watts": "Power Draw",
    "fan_speed_rpm": "Fan Speed",
    "uptime_hours": "Uptime",
    "disk_io_total": "Disk I/O",
    "network_io_total": "Network I/O",
    "cpu_mem_pressure": "CPU+Memory Pressure",
    "temp_error_interaction": "Temp x Error Interaction",
    "temp_trend_slope": "Temperature Trend",
    "cpu_trend_slope": "CPU Trend",
    "error_count_rolling_avg": "Recent Error Trend",
    "temp_delta": "Temperature Change",
    "error_delta": "Error Count Change",
}

FEATURE_COLUMNS = [
    "cpu_usage", "memory_usage", "disk_usage", "disk_read_mbps",
    "disk_write_mbps", "network_in_mbps", "network_out_mbps",
    "temperature_c", "fan_speed_rpm", "power_watts", "error_count",
    "uptime_hours"
]


def engineer_row(row: dict) -> pd.DataFrame:
    """Apply the same feature engineering used in training to a single row."""
    df = pd.DataFrame([row])
    df["disk_io_total"] = df["disk_read_mbps"] + df["disk_write_mbps"]
    df["network_io_total"] = df["network_in_mbps"] + df["network_out_mbps"]
    df["cpu_mem_pressure"] = (df["cpu_usage"] / 100) * (df["memory_usage"] / 100)
    df["temp_error_interaction"] = df["temperature_c"] * df["error_count"]

    bucket = pd.cut(
        df["uptime_hours"], bins=[0, 1000, 3000, 5000, float("inf")],
        labels=["low", "medium", "high", "very_high"]
    )
    for b in ["low", "medium", "high", "very_high"]:
        df[f"uptime_{b}"] = (bucket == b).astype(int)

    # Ensure column order matches training features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    return df[model_features]


def explain_prediction(X_row: pd.DataFrame, top_n: int = 5):
    """
    Returns the top_n features driving THIS prediction, using SHAP values.
    Positive shap value = pushed prediction toward failure.
    Negative shap value = pushed prediction away from failure (safer).
    """
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        row_shap = shap_values[1][0]           # older shap: list [class0, class1]
    elif shap_values.ndim == 3:
        row_shap = shap_values[0, :, 1]         # newer shap: (samples, features, classes)
    else:
        row_shap = shap_values[0]               # regressor-style single array

    contributions = list(zip(X_row.columns, row_shap))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    total_abs = sum(abs(v) for _, v in contributions) or 1e-9
    explanations = []
    for feat, val in contributions[:top_n]:
        pct = round(float(100 * val / total_abs), 1)
        explanations.append({
            "feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
            "impact_pct": pct,
            "direction": "increases" if val > 0 else "decreases",
        })
    return explanations


def prioritized_recommendations(row: dict, risk: str):
    """Priority-tagged recommendations (1 = most urgent)."""
    recs = []
    if row.get("temperature_c", 0) > 75:
        recs.append({"priority": 1, "text": "Inspect cooling system / airflow — temperature critically elevated"})
    if row.get("error_count", 0) > 10:
        recs.append({"priority": 1, "text": "Investigate hardware error logs — high fault rate detected"})
    if row.get("memory_usage", 0) > 85:
        recs.append({"priority": 2, "text": "Check for memory leaks — usage is very high"})
    if row.get("cpu_usage", 0) > 85:
        recs.append({"priority": 2, "text": "Review workload distribution — CPU saturated"})
    if row.get("disk_usage", 0) > 85:
        recs.append({"priority": 3, "text": "Consider disk cleanup or replacement — storage nearly full"})
    if not recs:
        recs.append({"priority": 3, "text": "No immediate action needed — continue routine monitoring"})
    recs.sort(key=lambda r: r["priority"])
    return recs


def risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "HIGH"
    elif prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def get_fleet_data():
    """Load latest snapshot per server and score with the model."""
    df = pd.read_csv(DATA_PATH)
    latest = df.sort_values("snapshot_index").groupby("server_id").tail(1).copy()

    X = latest[FEATURE_COLUMNS].to_dict(orient="records")
    engineered_rows = [engineer_row(r) for r in X]
    X_full = pd.concat(engineered_rows, ignore_index=True)

    latest["failure_probability"] = model.predict_proba(X_full)[:, 1]
    latest["health_score"] = ((1 - latest["failure_probability"]) * 100).round(1)
    latest["risk"] = latest["failure_probability"].apply(risk_level)
    return latest


@app.route("/")
def home():
    fleet = get_fleet_data()
    total = len(fleet)
    healthy = (fleet["risk"] == "LOW").sum()
    critical = (fleet["risk"] == "HIGH").sum()
    avg_health = round(fleet["health_score"].mean(), 1)

    servers = fleet.sort_values("failure_probability", ascending=False).to_dict(orient="records")

    return render_template(
        "home.html", total=total, healthy=healthy, critical=critical,
        avg_health=avg_health, servers=servers
    )


@app.route("/server/<int:server_id>")
def server_detail(server_id):
    fleet = get_fleet_data()
    server = fleet[fleet["server_id"] == server_id]
    if server.empty:
        return "Server not found", 404
    server = server.iloc[0].to_dict()

    row = {col: server[col] for col in FEATURE_COLUMNS}
    X_full = engineer_row(row)
    explanation = explain_prediction(X_full)
    recommendations = prioritized_recommendations(row, server["risk"])

    return render_template(
        "server_detail.html", server=server,
        explanation=explanation, recommendations=recommendations
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        row = {col: float(request.form.get(col, 0)) for col in FEATURE_COLUMNS}
        X_full = engineer_row(row)
        prob = model.predict_proba(X_full)[:, 1][0]
        risk = risk_level(prob)
        result = {
            "probability": round(prob * 100, 1),
            "risk": risk,
            "inputs": row,
            "explanation": explain_prediction(X_full),
            "recommendations": prioritized_recommendations(row, risk),
        }
    return render_template("predict.html", result=result, columns=FEATURE_COLUMNS)


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
