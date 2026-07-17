"""
evaluate_final.py
------------------
Generates ROC curve and confusion matrix plots for the final chosen model
(Random Forest), and saves a health-score prediction helper used by the app.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
model = joblib.load(f"{MODELS_DIR}/random_forest.pkl")

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Random Forest (Final Model)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=120)
print("Saved reports/roc_curve.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Random Forest (Final Model)")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=120)
print("Saved reports/confusion_matrix.png")
