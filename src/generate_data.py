"""
generate_data.py
----------------
Generates a synthetic server telemetry dataset for the Predictive Server
Failure Detection project.

WHY SYNTHETIC DATA:
Real hardware-failure telemetry (e.g. Backblaze drive stats) is either
disk-only (SMART attributes) or not accessible in this environment.
To build a project that covers the FULL feature set an interviewer expects
(CPU, memory, disk, network, temperature, power, errors, uptime), we
generate data with deliberately engineered, documented correlations
between telemetry and failure. This keeps the ML problem realistic and
defensible: "I designed the generation logic so failure probability rises
with temperature, error count, and resource saturation -- the same signals
real hardware monitoring systems (like HPE InfoSight) track."

Run:
    python src/generate_data.py
Output:
    data/raw/server_telemetry.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SERVERS = 300          # distinct servers
N_SNAPSHOTS_PER_SERVER = 40   # telemetry snapshots over time per server
N_ROWS = N_SERVERS * N_SNAPSHOTS_PER_SERVER


def generate_dataset():
    rows = []

    for server_id in range(1, N_SERVERS + 1):
        # Each server has a "baseline health" -- some servers are just
        # older / more failure-prone than others (models real fleets).
        baseline_wear = RNG.beta(2, 5)  # 0 (healthy) -> 1 (worn out)

        # Simulate a degradation trajectory across snapshots (time-series-ish)
        degrade_rate = RNG.uniform(0.0, 1.0) * baseline_wear

        for t in range(N_SNAPSHOTS_PER_SERVER):
            wear = min(1.0, baseline_wear + degrade_rate * (t / N_SNAPSHOTS_PER_SERVER))

            cpu_usage = np.clip(RNG.normal(40 + 40 * wear, 12), 0, 100)
            memory_usage = np.clip(RNG.normal(35 + 45 * wear, 15), 0, 100)
            disk_usage = np.clip(RNG.normal(50 + 30 * wear, 15), 0, 100)
            disk_read = np.clip(RNG.normal(120 - 40 * wear, 30), 0, None)
            disk_write = np.clip(RNG.normal(100 - 35 * wear, 25), 0, None)
            network_in = np.clip(RNG.normal(200 - 50 * wear, 40), 0, None)
            network_out = np.clip(RNG.normal(180 - 45 * wear, 40), 0, None)
            temperature = np.clip(RNG.normal(45 + 30 * wear, 6), 20, 100)
            fan_speed = np.clip(RNG.normal(3000 - 800 * wear, 300), 500, 5000)
            power = np.clip(RNG.normal(250 + 60 * wear, 20), 100, 500)
            error_count = RNG.poisson(1 + 8 * wear)
            uptime_hours = np.clip(RNG.normal(4000 - 1500 * wear, 800), 1, None)

            # ---- Failure probability logic (documented, tunable) ----
            # Weighted combination of the strongest real-world failure
            # indicators: temperature, error count, and resource saturation.
            risk_score = (
                0.30 * (temperature - 45) / 30 +
                0.25 * (error_count / 10) +
                0.15 * (cpu_usage / 100) +
                0.15 * (memory_usage / 100) +
                0.15 * wear
            )
            failure_prob = 1 / (1 + np.exp(-6 * (risk_score - 0.85)))  # sigmoid, tuned for ~8-10% failure rate
            failure = RNG.binomial(1, np.clip(failure_prob, 0, 1))

            rows.append([
                server_id, t, round(cpu_usage, 2), round(memory_usage, 2),
                round(disk_usage, 2), round(disk_read, 2), round(disk_write, 2),
                round(network_in, 2), round(network_out, 2), round(temperature, 2),
                round(fan_speed, 0), round(power, 2), int(error_count),
                round(uptime_hours, 1), int(failure)
            ])

    columns = [
        "server_id", "snapshot_index", "cpu_usage", "memory_usage",
        "disk_usage", "disk_read_mbps", "disk_write_mbps",
        "network_in_mbps", "network_out_mbps", "temperature_c",
        "fan_speed_rpm", "power_watts", "error_count",
        "uptime_hours", "failure"
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/raw/server_telemetry.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Failure rate: {df['failure'].mean():.2%}")
    print(df.head())
