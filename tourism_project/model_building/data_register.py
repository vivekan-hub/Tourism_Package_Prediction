import pandas as pd
import sys
import os

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]
TARGET_COL = "ProdTaken"

def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        print(f"\n⚠️  MISSING columns: {sorted(missing_cols)}")
    else:
        print("\n✅ All expected columns present.")

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")

    if TARGET_COL in df.columns:
        print(f"\nTarget ({TARGET_COL}) distribution:")
        print(df[TARGET_COL].value_counts())

    print("=" * 60)
    if missing_cols:
        sys.exit(1)
    return df

if __name__ == "__main__":
    register_dataset()
