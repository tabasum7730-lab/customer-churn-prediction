"""
The "tool" our agent calls. This is just a regular Python function —
the agent (Gemini) decides WHEN to call it and WHAT to say with the result,
but the actual math/prediction happens here, not inside the LLM.

This separation is the core idea of agents: LLM = reasoning, tools = facts/actions.
"""

import pandas as pd
import joblib

# Load the saved model bundle once, when this file is imported
bundle = joblib.load("churn_model_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]


def predict_churn_risk(customer: dict) -> dict:
    """
    Takes a dictionary describing one customer and returns their churn risk.

    Example input:
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
        "TotalCharges": 140.0
    }

    Returns:
    {
        "churn_probability": 0.73,
        "risk_level": "High",
        "top_risk_factors": ["Month-to-month contract", "Fiber optic service", ...]
    }
    """
    # 1. Turn the single customer dict into a one-row DataFrame
    row = pd.DataFrame([customer])

    # 2. One-hot encode it the SAME way the training data was encoded
    row_encoded = pd.get_dummies(row)

    # 3. Make sure it has exactly the same columns as training data,
    # in the same order. Any missing column = 0 (that category wasn't present).
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4. Scale it using the SAME scaler fitted during training
    row_scaled = scaler.transform(row_encoded)

    # 5. Predict probability of churn (class 1 = "Yes")
    probability = model.predict_proba(row_scaled)[0][1]

    # 6. Translate into a simple risk level
    if probability >= 0.6:
        risk_level = "High"
    elif probability >= 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # 7. Identify which of THIS customer's traits push risk up the most,
    # using the model's known top risk-increasing factors
    risk_factors = []
    if customer.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract")
    if customer.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic internet service")
    if customer.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Pays by electronic check")
    if customer.get("tenure", 99) < 12:
        risk_factors.append("New customer (under 12 months tenure)")
    if customer.get("OnlineSecurity") == "No":
        risk_factors.append("No online security add-on")

    return {
        "churn_probability": round(float(probability), 3),
        "risk_level": risk_level,
        "top_risk_factors": risk_factors
    }


# Quick manual test when running this file directly
if __name__ == "__main__":
    test_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
        "TotalCharges": 140.0
    }
    result = predict_churn_risk(test_customer)
    print(result)
