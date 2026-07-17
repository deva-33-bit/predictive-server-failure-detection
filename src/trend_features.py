"""
trend_features.py
------------------
Adds time-series degradation features per server, using the historical
snapshots already present in the dataset (40 snapshots/server). This
directly addresses the biggest limitation of the original single-snapshot
model: it had no notion of whether a server was STABLE or DEGRADING.

Why trend features instead of jumping straight to an LSTM:
- With only 40 snapshots per server and 300 servers, a deep sequence model
  (LSTM) would be heavily overparameterized and hard to validate honestly.
- Rolling trend/slope features are a standard, well-understood technique
  in real predictive-maintenance systems and are far more defensible in
  an interview than a black-box sequence model trained on this little data.
- This keeps the model interpretable while still capturing "is this server
  getting worse over time" -- the actual gap in the original approach.

New features added (computed per server, using only PAST snapshots to
avoid leaking future information):
- temp_trend_slope       : linear trend of temperature over last 5 snapshots
- error_count_rolling_avg: rolling mean of error_count over last 5 snapshots
- cpu_trend_slope        : linear trend of cpu_usage over last 5 snapshots
- temp_delta             : change in temperature vs previous snapshot
- error_delta            : change in error_count vs previous snapshot
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/raw/server_telemetry.csv"
OUT_PATH = "data/processed/server_telemetry_with_trends.csv"

WINDOW = 5


def _slope(y):
    """Simple linear trend (slope) over a short window."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["server_id", "snapshot_index"]).reset_index(drop=True).copy()
    server_ids = df["server_id"]  # keep a reference; pandas 3.0 groupby.apply drops the group column

    def per_server(group):
        group = group.copy()
        group["temp_trend_slope"] = (
            group["temperature_c"].rolling(WINDOW, min_periods=2).apply(_slope, raw=True)
        )
        group["cpu_trend_slope"] = (
            group["cpu_usage"].rolling(WINDOW, min_periods=2).apply(_slope, raw=True)
        )
        group["error_count_rolling_avg"] = (
            group["error_count"].rolling(WINDOW, min_periods=1).mean()
        )
        group["temp_delta"] = group["temperature_c"].diff()
        group["error_delta"] = group["error_count"].diff()
        return group

    df_out = df.groupby("server_id", group_keys=False)[df.columns].apply(per_server)
    if "server_id" not in df_out.columns:
        df_out.insert(0, "server_id", server_ids.loc[df_out.index])

    # First snapshot per server has no history -- fill with 0 (neutral trend)
    trend_cols = ["temp_trend_slope", "cpu_trend_slope", "temp_delta", "error_delta"]
    df_out[trend_cols] = df_out[trend_cols].fillna(0.0)
    df_out["error_count_rolling_avg"] = df_out["error_count_rolling_avg"].fillna(df_out["error_count"])

    return df_out


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    df_trend = add_trend_features(df)
    df_trend.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with shape {df_trend.shape}")
    print("\nNew trend feature correlation with failure:")
    print(df_trend[
        ["temp_trend_slope", "cpu_trend_slope", "error_count_rolling_avg",
         "temp_delta", "error_delta", "failure"]
    ].corr()["failure"].sort_values(ascending=False))
