import pandas as pd
import joblib
import json
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

DEPLOY_DIR = "tourism_project/deployment"
MODEL_PATH = f"{DEPLOY_DIR}/best_model.pkl"
METRICS_PATH = f"{DEPLOY_DIR}/metrics.json"
PARAMS_PATH = f"{DEPLOY_DIR}/best_params.json"


def load_splits():
    for f in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run prep.py first.")
            sys.exit(1)
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze()
    y_test = pd.read_csv("ytest.csv").squeeze()
    print(f"Xtrain: {X_train.shape}, Xtest: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_pipeline(X_train):
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    print(f"Numeric: {numeric_cols}")
    print(f"Categorical: {categorical_cols}")

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    )

    scale_pos_weight = 1  # placeholder, set properly in main using y_train

    model = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)

    pipeline = make_pipeline(preprocessor, model)
    return pipeline


def tune_and_log(pipeline, X_train, y_train, X_test, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline.set_params(xgbclassifier__scale_pos_weight=scale_pos_weight)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    }

    mlflow.set_experiment("wellness-package-prediction")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("Best params:", best_params)
        mlflow.log_params(best_params)

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        print("Metrics:", metrics)
        mlflow.log_metrics(metrics)

        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        mlflow.sklearn.log_model(best_pipeline, "model")

        return best_pipeline, best_params, metrics


def save_artifacts(pipeline, best_params, metrics):
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved params/metrics to {DEPLOY_DIR}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_splits()
    pipeline = build_pipeline(X_train)
    best_pipeline, best_params, metrics = tune_and_log(pipeline, X_train, y_train, X_test, y_test)
    save_artifacts(best_pipeline, best_params, metrics)
retrain trigger
