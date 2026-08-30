import pandas as pd
import sys

# Expected columns from the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

DATA_PATH = "data/tourism.csv"

def validate_data(path: str = DATA_PATH) -> pd.DataFrame:
    # 1. Load the dataset
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}")
        sys.exit(1)

    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    # 2. Check expected columns are present
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        print(f"⚠️  MISSING columns: {missing_cols}")
    else:
        print("✅ All expected columns are present.")

    if extra_cols:
        print(f"ℹ️  Extra/unexpected columns found: {extra_cols}")

    # 3. Shape
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 4. Dtypes
    print("\nColumn dtypes:")
    print(df.dtypes)

    # 5. Missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    # 6. Target variable distribution
    if "ProdTaken" in df.columns:
        print("\nTarget variable (ProdTaken) distribution:")
        print(df["ProdTaken"].value_counts())
        print(df["ProdTaken"].value_counts(normalize=True).round(3))
    else:
        print("\n⚠️  Target column 'ProdTaken' not found!")

    print("=" * 60)

    # Fail the script (useful for CI/CD) if columns are missing
    if missing_cols:
        sys.exit(1)

    return df

if __name__ == "__main__":
    validate_data()