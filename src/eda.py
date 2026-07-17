"""
eda.py
------
Quick exploratory data analysis for the server telemetry dataset.
Produces summary stats and saves plots to reports/.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/server_telemetry.csv")

print("Shape:", df.shape)
print("\nFailure rate:", df["failure"].mean())
print("\nMissing values:\n", df.isnull().sum().sum())
print("\nDescribe:\n", df.describe().T[["mean", "std", "min", "max"]])

# Correlation of each numeric feature with failure
corr = df.drop(columns=["server_id", "snapshot_index"]).corr()["failure"].sort_values(ascending=False)
print("\nCorrelation with failure:\n", corr)

# Plot: key features vs failure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(
    axes.flat,
    ["temperature_c", "error_count", "cpu_usage", "memory_usage"]
):
    df.boxplot(column=col, by="failure", ax=ax)
    ax.set_title(f"{col} vs failure")
    ax.set_xlabel("failure (0=No, 1=Yes)")

plt.suptitle("")
plt.tight_layout()
plt.savefig("reports/eda_boxplots.png", dpi=120)
print("\nSaved reports/eda_boxplots.png")

# Correlation bar chart
plt.figure(figsize=(8, 6))
corr.drop("failure").plot(kind="barh")
plt.title("Feature correlation with failure")
plt.tight_layout()
plt.savefig("reports/correlation_with_failure.png", dpi=120)
print("Saved reports/correlation_with_failure.png")
