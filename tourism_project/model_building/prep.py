import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

RAW_DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"
COLUMNS_TO_DROP = ["CustomerID"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], inplace=True)

    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    df.drop_duplicates(inplace=True)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "TypeofContact" in df.columns:
        df["TypeofContact"] = df["TypeofContact"].str.strip()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"Cleaned shape: {df.shape}, remaining missing values: {df.isnull().sum().sum()}")
    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Saved at repo root to match the GitHub Actions artifact paths
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print(f"Xtrain: {X_train.shape}, Xtest: {X_test.shape}")
    print(f"ytrain distribution:\n{y_train.value_counts(normalize=True).round(3)}")
    print(f"ytest distribution:\n{y_test.value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    split_and_save(df_clean)
