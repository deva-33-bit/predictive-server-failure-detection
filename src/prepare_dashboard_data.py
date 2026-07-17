"""
prepare_dashboard_data.py
--------------------------
Builds a "current fleet snapshot" for the dashboard: one latest row per
server_id from the raw data, scored with the trained model. This is what
powers the Home and Server Details pages (simulating a live monitoring feed).
"""

import pandas as pd
import joblib
from preprocess import engineer_features

RAW_PATH = "data/raw/server_telemetry.csv"
MODELS_DIR = "models"


def main():
    df = pd.read_csv(RAW_PATH)

    # Take the latest snapshot per server (as if this were "now")
    latest = df.sort_values("snapshot_index").groupby("server_id").tail(1).reset_index(drop=True)

    features_df = engineer_features(latest)
    X = features_df.drop(columns=["server_id", "snapshot_index", "failure"])

    model = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    proba = model.predict_proba(X)[:, 1]

    latest["failure_probability"] = proba
    latest["risk_level"] = pd.cut(
        proba, bins=[-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]
    )
    latest["health_score"] = ((1 - proba) * 100).round(1)

    latest.to_csv("data/processed/fleet_snapshot.csv", index=False)
    print(f"Saved fleet snapshot for {len(latest)} servers")
    print(latest["risk_level"].value_counts())


if __name__ == "__main__":
    main()
