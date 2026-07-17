"""
train_models.py
----------------
Trains Random Forest and XGBoost, tunes hyperparameters with
RandomizedSearchCV, and compares all models (including the saved
Logistic Regression baseline) side by side.

Why Random Forest + XGBoost specifically:
- Random Forest handles non-linear feature interactions well and gives
  clean feature importances -- good for explaining "why" a server is at risk.
- XGBoost typically squeezes out extra performance on tabular data and
  is the industry-standard choice for this kind of problem.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(name, y_test, y_pred, y_proba):
    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return metrics


def main():
    X_train, X_test, y_train, y_test = load_data()
    results = []

    # ---------------- Random Forest ----------------
    rf_param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_param_dist, n_iter=10, scoring="roc_auc", cv=3,
        random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print("Best RF params:", rf_search.best_params_)

    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest (tuned)", y_test, y_pred_rf, y_proba_rf))
    joblib.dump(best_rf, f"{MODELS_DIR}/random_forest.pkl")

    # ---------------- XGBoost ----------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # imbalance ratio

    xgb_param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42
        ),
        xgb_param_dist, n_iter=10, scoring="roc_auc", cv=3,
        random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print("\nBest XGB params:", xgb_search.best_params_)

    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost (tuned)", y_test, y_pred_xgb, y_proba_xgb))
    joblib.dump(best_xgb, f"{MODELS_DIR}/xgboost_model.pkl")

    # ---------------- Comparison table ----------------
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("reports/model_comparison.csv", index=False)

    best_model_name = results_df.loc[results_df["roc_auc"].idxmax(), "model"]
    print(f"\nBest model by ROC-AUC: {best_model_name}")


if __name__ == "__main__":
    main()
