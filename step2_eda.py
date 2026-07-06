"""
STEP 2: Explore the data to find what actually drives churn.

What this script does:
1. Loads the cleaned data from step 1
2. Calculates the overall churn rate (our baseline)
3. Checks churn rate broken down by a few key columns
4. Makes 3 charts so you can SEE the patterns, not just read numbers
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_csv("telco_clean.csv")

# 2. Overall churn rate — this is our baseline.
# If a column's churn rate is way above or below this number, it matters.
overall_churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"Overall churn rate: {overall_churn_rate:.1f}%")

# Helper function: churn rate (%) for each category in a column
def churn_rate_by(column):
    rates = df.groupby(column)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    return rates.sort_values(ascending=False)

# 3. Check churn rate by Contract type
print("\n--- Churn rate by Contract type ---")
contract_rates = churn_rate_by("Contract")
print(contract_rates)

# 4. Check churn rate by Internet service type
print("\n--- Churn rate by Internet service ---")
internet_rates = churn_rate_by("InternetService")
print(internet_rates)

# 5. Check churn rate by Payment method
print("\n--- Churn rate by Payment method ---")
payment_rates = churn_rate_by("PaymentMethod")
print(payment_rates)

# 6. Check average tenure for customers who left vs stayed
print("\n--- Average tenure (months) ---")
print(df.groupby("Churn")["tenure"].mean())

# ===== CHARTS =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Chart 1: Churn rate by contract type
contract_rates.plot(kind="bar", ax=axes[0], color="#0F6E56")
axes[0].set_title("Churn rate by contract type")
axes[0].set_ylabel("Churn rate (%)")
axes[0].axhline(overall_churn_rate, color="red", linestyle="--", linewidth=1)

# Chart 2: Tenure distribution — stayed vs left
df[df["Churn"] == "No"]["tenure"].plot(kind="hist", bins=30, alpha=0.6, label="Stayed", ax=axes[1], color="#0F6E56")
df[df["Churn"] == "Yes"]["tenure"].plot(kind="hist", bins=30, alpha=0.6, label="Churned", ax=axes[1], color="#993C1D")
axes[1].set_title("Tenure: stayed vs churned")
axes[1].set_xlabel("Tenure (months)")
axes[1].legend()

# Chart 3: Churn rate by internet service
internet_rates.plot(kind="bar", ax=axes[2], color="#534AB7")
axes[2].set_title("Churn rate by internet service")
axes[2].set_ylabel("Churn rate (%)")
axes[2].axhline(overall_churn_rate, color="red", linestyle="--", linewidth=1)

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150)
print("\nSaved charts as eda_charts.png")
