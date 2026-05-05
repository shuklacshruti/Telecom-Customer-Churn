# Telecom Customer Churn Analysis and Prediction

**Author:** Shruti Shukla (NetID: scs250)  
**Date:** 16 March, 2026

---

## Research Question

> *Which customer features are most associated with telecom customer churn, and how effectively can machine learning models predict which customers are likely to leave?*

## Project Overview

This project investigates telecom customer churn using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It covers the full data science pipeline:

1. **Exploratory Data Analysis** – churn distribution, feature relationships, correlation analysis
2. **Data Preprocessing** – cleaning, encoding, feature engineering, SMOTE oversampling
3. **Model Development** – Logistic Regression, Decision Tree, Random Forest
4. **Model Evaluation** – Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Repository Structure

```
Telecom-Customer-Churn/
├── data/
│   └── README.md                         # Dataset download instructions
├── notebooks/
│   └── telecom_churn_analysis.ipynb      # Main analysis notebook
├── src/
│   ├── preprocessing.py                  # Data loading, cleaning, encoding, SMOTE
│   ├── eda.py                            # EDA visualisation functions
│   └── models.py                         # Model training and evaluation
├── outputs/
│   └── figures/                          # All saved plots (auto-created)
├── main.py                               # End-to-end script version of the analysis
└── requirements.txt                      # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add the Real Dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/`.

> **Without the real data:** A synthetic dataset with the same structure and distributions is generated automatically.

### 3a. Run via Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/telecom_churn_analysis.ipynb
```

### 3b. Run as a Script

```bash
python main.py
```

All figures are saved to `outputs/figures/`.

## Key Findings

| Feature | Association with Churn |
|---|---|
| Contract type (month-to-month) | Very high |
| Tenure (short) | Very high |
| Monthly charges (high) | High |
| Fiber optic internet | High |
| No add-on services | Moderate |
| Electronic check payment | Moderate |

## Model Performance (on Synthetic Data)

| Model | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| Decision Tree | Highest | Moderate | Moderate |
| Logistic Regression | Moderate | Moderate | Highest |
| Random Forest | Moderate | Moderate | Moderate |

> **Primary metric:** Recall – catching as many churners as possible is most important for a retention campaign.

## Tools & Libraries

- Python 3.12
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- imbalanced-learn (SMOTE)