import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

RAW_DATA_PATH = "tourism_project/data/tourism.csv"
TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"

COLUMNS_TO_DROP = ["CustomerID"]
TARGET_COL = "ProdTaken"


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier columns
    df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], inplace=True)
    print(f"Dropped columns: {COLUMNS_TO_DROP}")

    # Drop stray unnamed index columns from CSV exports
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        print(f"Dropped stray index columns: {unnamed_cols}")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    # Fix inconsistent categorical labels
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "TypeofContact" in df.columns:
        df["TypeofContact"] = df["TypeofContact"].str.strip()

    # Impute missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_val}")

    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"\nTrain set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True).round(3)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True).round(3)}")

    return train_df, test_df


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"\nSaved train split to {TRAIN_PATH}")
    print(f"Saved test split to {TEST_PATH}")


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    train_df, test_df = split_data(df_clean)
    save_splits(train_df, test_df)
