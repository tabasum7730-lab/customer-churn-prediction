"""
STEP 1: Load and clean the Telco Customer Churn dataset.

What this script does:
1. Loads the raw CSV into a Pandas DataFrame
2. Checks the shape, column types, and missing values
3. Fixes a known issue: 'TotalCharges' is stored as text instead of numbers
4. Drops the customerID column (it's just an ID, not useful for analysis)
5. Saves a clean version of the file for the next step (EDA)
"""

import pandas as pd

# 1. Load the raw data
df = pd.read_csv("telco_raw.csv")

print("Shape (rows, columns):", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))

print("\nColumn data types:")
print(df.dtypes)

# 2. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Fix TotalCharges — it looks numeric but Pandas read it as text (object)
# because some rows have a blank string "" instead of a number.
# errors='coerce' turns anything that can't be converted into NaN (Not a Number).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nRows where TotalCharges became NaN after conversion:")
print(df[df["TotalCharges"].isnull()][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

# These are all customers with tenure = 0 (brand new customers, haven't been
# billed yet) so it's correct to fill these with 0, not drop them.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 4. Drop customerID — it's a unique ID, has zero predictive value
df = df.drop(columns=["customerID"])

# 5. Confirm everything is clean now
print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values left")

print("\nFinal shape:", df.shape)

# 6. Save the cleaned file
df.to_csv("telco_clean.csv", index=False)
print("\nSaved cleaned file as telco_clean.csv")
