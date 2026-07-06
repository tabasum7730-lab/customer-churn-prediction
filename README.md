# 🤖 Customer Churn Prediction AI Agent

> An AI Agent that predicts which telecom customers are about to leave — and tells you exactly why and what to do about it.

---

## 📌 Problem Statement

Telecom companies lose thousands of customers every month. The problem is they have no way to know *which* customer is about to leave *before* they actually leave. By the time the customer cancels, it's too late. Companies are basically flying blind.

## ✅ Solution

We trained a machine learning model on 7,000+ real customer records that predicts churn risk for any new customer. Then we wrapped it inside an AI agent powered by Google Gemini — so instead of just getting a number, you describe a customer in plain English and the agent tells you their risk level, why they're at risk, and what retention action to take.

It's like having a **data scientist + retention specialist** in one tool.

---

## 🚀 How It Works

```
User describes a customer in plain English
        ↓
Gemini Agent receives the message
        ↓
Agent calls predict_churn_risk() tool (our trained ML model)
        ↓
Tool returns: churn probability + risk factors
        ↓
Gemini writes a human, actionable retention recommendation
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset | IBM Telco Customer Churn (7,043 customers) |
| Model | Logistic Regression |
| Accuracy | **80.7%** on unseen test data |
| Overall churn rate | 26.5% |
| Biggest churn driver | Month-to-month contracts (42.7% churn rate) |

---

## 🗂️ Project Structure

```
churn-prediction-project/
├── telco_raw.csv              # Original IBM Telco dataset
├── step1_clean_data.py        # Data loading and cleaning
├── telco_clean.csv            # Cleaned dataset output
├── step2_eda.py               # Exploratory Data Analysis
├── eda_charts.png             # EDA visualizations
├── step3_model.py             # Logistic Regression model + evaluation
├── model_results.png          # Confusion matrix + feature importance chart
├── step4_save_model.py        # Save trained model to disk
├── churn_model_bundle.pkl     # Saved model, scaler, and feature columns
├── churn_tool.py              # Prediction tool the agent calls
├── churn_agent.py             # Gemini AI Agent with function calling
├── step5_findings_report.md   # Written findings and business recommendations
├── requirements.txt           # All dependencies
└── README.md                  # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Gemini API key
Go to aistudio.google.com/apikey, sign in with Google, click Create API Key.

### 4. Set your API key
Mac/Linux:
```bash
export GEMINI_API_KEY="your_key_here"
```
Windows:
```bash
set GEMINI_API_KEY=your_key_here
```

### 5. Run in order
```bash
python step1_clean_data.py
python step2_eda.py
python step3_model.py
python step4_save_model.py
python churn_agent.py
```

---

## 💬 Example Agent Interaction

Input:
"I have a customer who has been with us 2 months, uses fiber optic internet, pays by electronic check, is on a month-to-month contract, and pays $70/month."

Agent Output:
"This customer has a High churn risk (70% probability). Key risk factors are their month-to-month contract, fiber optic service, electronic check payment, and being a new customer under 12 months. Recommended actions: (1) Offer a discounted annual plan upgrade, (2) Migrate them to autopay with a small incentive, (3) Assign a retention specialist in the next 7 days."

---

## 🔑 Top Churn Drivers Found

| Factor | Finding |
|---|---|
| Contract type | Month-to-month = 43% churn vs 3% for two-year |
| Tenure | Churned customers stayed only 18 months on average vs 38 months |
| Internet service | Fiber optic = 42% churn vs 7% for no internet |
| Payment method | Electronic check = 45% churn vs 15% for autopay |

---

## 🛠️ Technologies Used

- Python, Pandas, Scikit-learn, Matplotlib
- Google Gemini SDK + Function Calling
- Logistic Regression, StandardScaler
- Joblib (model saving)

---

## 🎯 Advantages

- Full AI Agent — reasons and acts, not just predicts
- Explains WHY a customer is at risk
- Anyone can use it in plain English
- 80.7% accuracy on real data
- Uses Google Gemini Function Calling — exactly what industry uses today

---

Built as a capstone for Google and Kaggle's 5-Day AI Agents Intensive Vibe Coding Course
