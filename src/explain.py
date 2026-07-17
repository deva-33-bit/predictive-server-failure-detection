"""
explain.py
----------
Feature importance (Random Forest) + SHAP values for model explainability.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
model = joblib.load(f"{MODELS_DIR}/random_forest.pkl")

# ---- Built-in feature importance ----
importances = pd.Series(model.feature_importances_, index=X_test.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
importances.head(12).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Random Forest — Feature Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=120)
print("Saved reports/feature_importance.png")
print(importances.head(10))

# ---- SHAP values ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.sample(min(500, len(X_test)), random_state=42))

# For binary classifiers, shap_values may be a list [class0, class1]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure()
shap.summary_plot(
    sv, X_test.sample(min(500, len(X_test)), random_state=42),
    show=False
)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=120)
print("Saved reports/shap_summary.png")
