# Data Directory

## Dataset

This project uses the **IBM Telco Customer Churn** dataset (~7,043 records, 21 feature columns).

### Download Instructions

1. Visit [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place the file in this directory: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Columns

| Column | Type | Description |
|---|---|---|
| customerID | string | Unique customer identifier (dropped during preprocessing) |
| gender | categorical | Customer gender (Male/Female) |
| SeniorCitizen | int | Whether the customer is a senior citizen (1/0) |
| Partner | categorical | Whether the customer has a partner (Yes/No) |
| Dependents | categorical | Whether the customer has dependents (Yes/No) |
| tenure | int | Months the customer has been with the company |
| PhoneService | categorical | Whether the customer has a phone service (Yes/No) |
| MultipleLines | categorical | Whether the customer has multiple lines |
| InternetService | categorical | Customer's internet service provider |
| OnlineSecurity | categorical | Whether the customer has online security |
| OnlineBackup | categorical | Whether the customer has online backup |
| DeviceProtection | categorical | Whether the customer has device protection |
| TechSupport | categorical | Whether the customer has tech support |
| StreamingTV | categorical | Whether the customer has streaming TV |
| StreamingMovies | categorical | Whether the customer has streaming movies |
| Contract | categorical | The contract term (Month-to-month, One year, Two year) |
| PaperlessBilling | categorical | Whether the customer has paperless billing |
| PaymentMethod | categorical | The customer's payment method |
| MonthlyCharges | float | Monthly amount charged to customer |
| TotalCharges | float | Total amount charged (has some missing values) |
| Churn | categorical | Whether the customer churned (Yes/No) — target variable |

### Synthetic Data Fallback

If the CSV file is not present, the project automatically generates a **synthetic dataset** with:
- The same column names, dtypes, and approximate distributions
- ~26% churn rate
- Realistic correlations between features (e.g., contract type → churn, tenure → total charges)

This allows all analysis and modelling code to run end-to-end without the real data.
