# Telecom Customer Churn Analysis and Prediction

**CS439: Intro to Data Science — Rutgers University**
**Shruti Shukla | NetID: scs250 | Section 05**

---

## Links

- **Slides:** https://docs.google.com/presentation/d/11lNJyzI0JRSZa4AcoyQ-16a37rGH2W5Qt03rAdlrEvU/edit?usp=sharing
- **Video:** https://drive.google.com/file/d/1Ri-eEW48S2MfihxOYHyFVe-qEHgpm-cq/view?usp=sharing
- **GitHub:** https://github.com/shuklacshruti/Telecom-Customer-Churn

---

## Research Question

Which customer features are most associated with telecom customer churn, and how effectively can machine learning models predict which customers are likely to leave?

---

## Project Overview

End-to-end machine learning pipeline for predicting customer churn using the IBM Telco Customer Churn dataset (7,032 records). The project covers data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

---

## File Structure

```
Telecom-Customer-Churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── churn_prediction.py
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repo and navigate into it:
   ```bash
   git clone https://github.com/shuklacshruti/Telecom-Customer-Churn.git
   cd Telecom-Customer-Churn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python3 churn_prediction.py
   ```

All 11 plots will be saved as PNG files in the project root.

---

## Models

| Model               | Recall | ROC-AUC |
|---------------------|--------|---------|
| Logistic Regression | 0.64   | 0.817   |
| Decision Tree       | 0.66   | 0.785   |
| Random Forest       | 0.65   | 0.816   |

Recall is the primary metric — missing a churner means losing that customer entirely.

---

## Key Findings

- Month-to-month customers churn at **42.7%** vs. 2.8% for two-year contracts
- Churned customers average **18.0 months** tenure vs. 37.7 months for loyal customers
- Electronic check payment is the strongest single predictor of churn
- A simple `riskFlag` feature (short tenure + high charges) identifies a segment with **51.3% churn rate**

---

## Dependencies

See `requirements.txt`. Key libraries: pandas, scikit-learn, imbalanced-learn, seaborn, matplotlib.
