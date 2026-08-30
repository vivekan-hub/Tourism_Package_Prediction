import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"
DEPLOY_DIR = "tourism_project/deployment"
MODEL_PATH = f"{DEPLOY_DIR}/best_model.pkl"
ENCODERS_PATH = f"{DEPLOY_DIR}/encoders.pkl"
METRICS_PATH = f"{DEPLOY_DIR}/metrics.json"
PARAMS_PATH = f"{DEPLOY_DIR}/best_params.json"
TARGET_COL = "ProdTaken"


def load_train_test():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("ERROR: train/test files not found. Run data_prep.py first.")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = test_df[col].astype(str).map(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
    print(f"Encoded categorical columns: {cat_cols}")
    return train_df, test_df, encoders


def split_X_y(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_train, y_train, X_test, y_test


def tune_model(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight (class imbalance): {scale_pos_weight:.3f}")

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5,
        scoring="f1", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("BEST PARAMETERS FOUND")
    print("=" * 60)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Test Set)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def get_feature_importance(model, feature_names):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names, "importance": importance
    }).sort_values("importance", ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


def save_artifacts(model, best_params, metrics, encoders):
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved best model to {MODEL_PATH}")

    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Saved label encoders to {ENCODERS_PATH}")

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params to {PARAMS_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    train_df, test_df = load_train_test()
    train_df, test_df, encoders = encode_categoricals(train_df, test_df)
    X_train, y_train, X_test, y_test = split_X_y(train_df, test_df)

    best_model, best_params = tune_model(X_train, y_train)
    metrics = evaluate_model(best_model, X_test, y_test)
    get_feature_importance(best_model, X_train.columns.tolist())
    save_artifacts(best_model, best_params, metrics, encoders)
