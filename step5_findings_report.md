# Customer Churn Prediction — Findings Report

## Objective
Identify which customers are likely to churn and what factors drive that risk, using the IBM Telco Customer Churn dataset (7,043 customers).

## Headline Numbers
- **Overall churn rate: 26.5%** — roughly 1 in 4 customers leave.
- **Model accuracy: 80.7%** using Logistic Regression on a held-out test set (20% of customers, never seen during training).

## Top Churn Drivers

### 1. Contract type is the single biggest factor
| Contract | Churn rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Month-to-month customers churn **15x more** than two-year customers. This is the clearest lever a business has: incentivizing longer contracts (discounts, perks) directly reduces churn risk.

### 2. Tenure compounds the effect
Customers who churn have stayed an average of **18 months**, versus **38 months** for those who remain. Churn is heavily front-loaded — the first few months are the highest-risk window, meaning onboarding and early engagement matter most.

### 3. Fiber optic customers churn at almost double the rate
| Internet service | Churn rate |
|---|---|
| Fiber optic | 41.9% |
| DSL | 19.0% |
| None | 7.4% |

This is counter-intuitive since fiber is the premium product — worth investigating pricing, reliability, or competitor pressure in this segment.

### 4. Payment method signals risk
| Payment method | Churn rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.1% |
| Bank transfer (auto) | 16.7% |
| Credit card (auto) | 15.2% |

Manual payment methods (especially electronic check) correlate with nearly 3x the churn rate of automatic payments — likely a proxy for lower engagement/commitment.

## Model Validation
The Logistic Regression model independently confirmed the EDA findings — its top weighted factors were:
- **Reduces churn risk:** tenure, two-year contracts, lower monthly charges
- **Increases churn risk:** fiber optic service, higher total charges, streaming add-ons

The fact that an unsupervised statistical model arrived at the same conclusions as manual EDA is strong evidence these patterns are real signal, not noise.

## Recommendations
1. **Push contract upgrades** — targeted offers to convert month-to-month customers to annual plans, especially during their first 90 days.
2. **Investigate fiber optic churn** — likely a pricing or service-quality issue worth a deeper dive.
3. **Reduce friction on electronic check payments** — nudge customers toward autopay with incentives.
4. **Prioritize early-tenure retention programs** — since churn risk is highest in month 1–12.

## Model Limitations
- Recall on churners is 57% — the model misses a meaningful share of customers who actually churn, so it should be used to *prioritize* outreach, not as a sole decision-maker.
- Logistic Regression assumes simple linear relationships; a tree-based model (Random Forest/XGBoost) could likely improve recall as a next iteration.
