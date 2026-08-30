import pandas as pd
import numpy as np
import joblib
import json
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "models/metrics.json"
PARAMS_PATH = "models/best_params.json"
TARGET_COL = "ProdTaken"


def load_train_test():
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Label-encode categorical columns, fitting only on train to avoid leakage."""
    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # Handle unseen categories in test gracefully
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
    """Define model + param grid, run GridSearchCV."""
    model = RandomForestClassifier(random_state=42, class_weight="balanced")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",       # F1 is a good choice given likely class imbalance
        n_jobs=-1,
        verbose=1
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


def save_artifacts(model, best_params, metrics):
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved best model to {MODEL_PATH}")

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
    save_artifacts(best_model, best_params, metrics)