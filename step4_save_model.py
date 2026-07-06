"""
STEP 4 (Agent prep): Save the trained model so our agent can load it instantly,
instead of retraining every time someone asks a question.

We save 3 things together:
1. The trained Logistic Regression model
2. The StandardScaler (needed to transform new customer data the same way)
3. The exact list/order of feature columns (so new data lines up correctly)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load and prep data (same as step3)
df = pd.read_csv("telco_clean.csv")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])
X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 2. Save everything the agent will need, bundled into one file
joblib.dump({
    "model": model,
    "scaler": scaler,
    "feature_columns": feature_columns
}, "churn_model_bundle.pkl")

print("Saved model bundle as churn_model_bundle.pkl")
print(f"Model expects {len(feature_columns)} features")
