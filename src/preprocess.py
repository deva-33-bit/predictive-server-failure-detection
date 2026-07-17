"""
preprocess.py
-------------
Feature engineering + train/test split for the server telemetry dataset.

Feature engineering choices (documented for interview discussion):
- disk_io_total: combined read+write throughput (single "disk activity" signal)
- network_io_total: combined in+out throughput
- cpu_mem_pressure: interaction term, since CPU+memory saturating together
  is a stronger failure signal than either alone
- temp_error_interaction: temperature and error_count together capture
  "thermal stress + faults happening simultaneously"
- uptime_bucket: uptime is non-linear in its effect, so we bucket it

Train/test split is STRATIFIED on the target because failure is imbalanced
(~10%) -- a random split could easily under/over-represent failures in
either set.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from trend_features import add_trend_features

RAW_PATH = "data/raw/server_telemetry.csv"
PROCESSED_DIR = "data/processed"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["disk_io_total"] = df["disk_read_mbps"] + df["disk_write_mbps"]
    df["network_io_total"] = df["network_in_mbps"] + df["network_out_mbps"]
    df["cpu_mem_pressure"] = (df["cpu_usage"] / 100) * (df["memory_usage"] / 100)
    df["temp_error_interaction"] = df["temperature_c"] * df["error_count"]
    df["uptime_bucket"] = pd.cut(
        df["uptime_hours"],
        bins=[0, 1000, 3000, 5000, float("inf")],
        labels=["low", "medium", "high", "very_high"]
    )
    df = pd.get_dummies(df, columns=["uptime_bucket"], prefix="uptime")

    # Time-series degradation features (requires server_id + snapshot_index
    # to already be present, i.e. NOT for a single manual dashboard input row)
    if "server_id" in df.columns and "snapshot_index" in df.columns and len(df) > 1:
        df = add_trend_features(df)

    return df


def main():
    df = pd.read_csv(RAW_PATH)
    df = engineer_features(df)

    # Drop identifiers not useful as model features
    feature_df = df.drop(columns=["server_id", "snapshot_index"])

    X = feature_df.drop(columns=["failure"])
    y = feature_df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    print("Train shape:", X_train.shape, " Failure rate:", y_train.mean().round(4))
    print("Test shape :", X_test.shape, " Failure rate:", y_test.mean().round(4))
    print("Features   :", list(X.columns))


if __name__ == "__main__":
    main()
