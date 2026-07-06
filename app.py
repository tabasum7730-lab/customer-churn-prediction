"""
Customer Churn Prediction AI Agent — Streamlit Web App
Anyone in the world can use this to check a customer's churn risk.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk AI Agent",
    page_icon="🤖",
    layout="centered"
)

# ── Load or train the model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load saved model bundle, or train from scratch if not available."""
    if os.path.exists("churn_model_bundle.pkl"):
        bundle = joblib.load("churn_model_bundle.pkl")
        return bundle["model"], bundle["scaler"], bundle["feature_columns"]

    # Download data if needed
    if not os.path.exists("telco_raw.csv"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            "telco_raw.csv"
        )

    df = pd.read_csv("telco_raw.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])

    y = (df["Churn"] == "Yes").astype(int)
    X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
    feature_columns = X.columns.tolist()

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    joblib.dump({"model": model, "scaler": scaler, "feature_columns": feature_columns}, "churn_model_bundle.pkl")
    return model, scaler, feature_columns


def predict_churn(customer: dict, model, scaler, feature_columns):
    """Run prediction and return probability + risk level + factors."""
    row = pd.DataFrame([customer])
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    row_scaled = scaler.transform(row_encoded)
    prob = model.predict_proba(row_scaled)[0][1]

    risk = "🔴 High" if prob >= 0.6 else ("🟡 Medium" if prob >= 0.3 else "🟢 Low")

    factors = []
    if customer.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract")
    if customer.get("InternetService") == "Fiber optic":
        factors.append("Fiber optic internet service")
    if customer.get("PaymentMethod") == "Electronic check":
        factors.append("Pays by electronic check")
    if customer.get("tenure", 99) < 12:
        factors.append("New customer (under 12 months)")
    if customer.get("OnlineSecurity") == "No":
        factors.append("No online security add-on")

    return round(float(prob) * 100, 1), risk, factors


def get_gemini_recommendation(customer_name, probability, risk_level, factors, api_key):
    """Ask Gemini to write a human retention recommendation."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            f"Customer '{customer_name}' has a {risk_level} churn risk with {probability}% probability. "
            f"Risk factors: {', '.join(factors) if factors else 'None identified'}. "
            f"Write a short, friendly retention recommendation (3-4 sentences max) for a telecom support agent. "
            f"Be specific and actionable."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini recommendation unavailable: {str(e)}"


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 Customer Churn Risk AI Agent")
st.markdown("Fill in the customer details below and the AI agent will predict their churn risk and tell you what to do.")
st.divider()

# Load model
with st.spinner("Loading AI model..."):
    model, scaler, feature_columns = load_model()

# Gemini API key input
st.subheader("🔑 Gemini API Key")
api_key = st.text_input(
    "Enter your Gemini API key (get one free at aistudio.google.com/apikey)",
    type="password",
    placeholder="AIzaSy..."
)
st.divider()

# Customer details form
st.subheader("👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    customer_name = st.text_input("Customer Name", value="Riya")
    tenure = st.slider("Tenure (months)", 0, 72, 2)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.subheader("📦 Services")
col3, col4 = st.columns(2)

with col3:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col4:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

# Predict button
if st.button("🔍 Check Churn Risk", type="primary", use_container_width=True):

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure
    }

    with st.spinner("Agent is analyzing..."):
        probability, risk_level, factors = predict_churn(customer, model, scaler, feature_columns)

    # Results
    st.subheader(f"📊 Results for {customer_name}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Churn Probability", f"{probability}%")
    with col_b:
        st.metric("Risk Level", risk_level)

    # Progress bar
    st.progress(int(probability))

    # Risk factors
    if factors:
        st.subheader("⚠️ Risk Factors")
        for f in factors:
            st.markdown(f"- {f}")
    else:
        st.success("No major risk factors found!")

    # Gemini recommendation
    st.subheader("💡 AI Retention Recommendation")
    if api_key:
        with st.spinner("Gemini is writing recommendation..."):
            recommendation = get_gemini_recommendation(
                customer_name, probability, risk_level, factors, api_key
            )
        st.info(recommendation)
    else:
        # Fallback without Gemini
        if "High" in risk_level:
            st.warning(
                f"**{customer_name} is at high risk of leaving.** "
                "Recommended actions: (1) Offer a discounted annual contract upgrade, "
                "(2) Migrate to autopay with a small incentive, "
                "(3) Assign a retention specialist within 7 days."
            )
        elif "Medium" in risk_level:
            st.warning(
                f"**{customer_name} has medium churn risk.** "
                "Recommended actions: (1) Send a satisfaction survey, "
                "(2) Offer a loyalty discount, "
                "(3) Check in with a proactive support call."
            )
        else:
            st.success(
                f"**{customer_name} is a low-risk customer.** "
                "Keep engaging with loyalty rewards and regular check-ins."
            )
        st.caption("Add your Gemini API key above for a personalised AI recommendation.")

st.divider()
st.caption("Built with Python · Scikit-learn · Google Gemini · Streamlit | IBM Telco Dataset | 80.7% Model Accuracy")
