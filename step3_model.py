"""
STEP 3: Build and evaluate a churn prediction model.

What this script does:
1. Converts text columns (like "Yes"/"No", "Month-to-month") into numbers,
   because ML models can only work with numbers
2. Splits data into a training set (80%) and a test set (20%)
3. Trains a Logistic Regression model on the training set
4. Tests it on data it has NEVER seen, to see how well it really predicts
5. Shows which factors the model thinks matter most (should match our EDA)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. Load cleaned data
df = pd.read_csv("telco_clean.csv")

# 2. Target column: what we're predicting (Yes/No -> 1/0)
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

# 3. One-hot encode all text columns into 0/1 columns
# e.g. "Contract" becomes 3 columns: Contract_One year, Contract_Two year, etc.
X = pd.get_dummies(X, drop_first=True)

print(f"Features after encoding: {X.shape[1]} columns")

# 4. Split into train (80%) and test (20%) sets
# stratify=y keeps the same churn ratio in both sets — important since only
# ~27% of customers churn, we don't want an unlucky split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training on {len(X_train)} customers, testing on {len(X_test)}")

# 5. Scale the features. This matters for two reasons:
#    a) Logistic Regression converges faster/cleaner on scaled data
#    b) Coefficients are only comparable to each other when features are on
#       the same scale — without this, "tenure" (0-72) would look artificially
#       weak next to a 0/1 dummy column purely due to scale, not importance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Predict on the unseen test set
y_pred = model.predict(X_test_scaled)

# 7. How good is it?
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.1%}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows=actual, cols=predicted):")
print(cm)

# 8. What does the model think matters most?
coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("\nTop 5 factors that REDUCE churn risk:")
print(coefficients.head(5))
print("\nTop 5 factors that INCREASE churn risk:")
print(coefficients.tail(5))

# ===== CHARTS =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix heatmap
im = axes[0].imshow(cm, cmap="Greens")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Stayed", "Churned"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Stayed", "Churned"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion matrix (accuracy: {acc:.1%})")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

# Top churn risk factors
top_factors = pd.concat([coefficients.head(5), coefficients.tail(5)])
colors = ["#0F6E56" if v < 0 else "#993C1D" for v in top_factors.values]
axes[1].barh(top_factors.index, top_factors.values, color=colors)
axes[1].set_title("Biggest churn risk factors (model weights)")
axes[1].axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
print("\nSaved chart as model_results.png")
