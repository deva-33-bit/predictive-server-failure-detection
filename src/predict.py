"""
predict.py
----------
Command-line prediction helper: pass telemetry values, get a failure
probability back. Useful for quick testing / scripting without the
Flask app.

Example:
    python src/predict.py --cpu_usage 85 --memory_usage 90 --disk_usage 60 \
        --disk_read_mbps 80 --disk_write_mbps 70 --network_in_mbps 150 \
        --network_out_mbps 140 --temperature_c 78 --fan_speed_rpm 2200 \
        --power_watts 300 --error_count 9 --uptime_hours 4200
"""

import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.pkl"

FEATURE_COLUMNS = [
    "cpu_usage", "memory_usage", "disk_usage", "disk_read_mbps",
    "disk_write_mbps", "network_in_mbps", "network_out_mbps",
    "temperature_c", "fan_speed_rpm", "power_watts", "error_count",
    "uptime_hours"
]


def engineer_row(row: dict, model) -> pd.DataFrame:
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

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    return df[model_features]


def main():
    parser = argparse.ArgumentParser(description="Predict server failure probability")
    for col in FEATURE_COLUMNS:
        parser.add_argument(f"--{col}", type=float, required=True)
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    row = {col: getattr(args, col) for col in FEATURE_COLUMNS}
    X = engineer_row(row, model)

    prob = model.predict_proba(X)[:, 1][0]
    risk = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.3 else "LOW"

    print(f"Failure Probability: {prob * 100:.1f}%")
    print(f"Risk Level: {risk}")


if __name__ == "__main__":
    main()
