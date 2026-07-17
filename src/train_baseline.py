"""
train_baseline.py
------------------
Baseline model: Logistic Regression, with class-imbalance handling.

Why start here: Logistic Regression is fast, interpretable, and gives you
a floor to beat. Two imbalance-handling strategies are compared:
  1. class_weight="balanced" (built into sklearn, reweights the loss)
  2. SMOTE oversampling (synthesizes minority-class samples)

Both are worth knowing for an interview -- class_weight is simpler and
doesn't risk overfitting to synthetic points; SMOTE can help more when
the minority class is very small and complex.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(name, y_test, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1       :", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))


def main():
    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Approach 1: class_weight="balanced" ---
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    evaluate("Logistic Regression (class_weight=balanced)", y_test, y_pred, y_proba)

    joblib.dump(model, f"{MODELS_DIR}/logistic_baseline.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    print(f"\nSaved {MODELS_DIR}/logistic_baseline.pkl and scaler.pkl")


if __name__ == "__main__":
    main()
