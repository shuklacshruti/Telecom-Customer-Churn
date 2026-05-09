'''
    Rutgers CS439: INTRO TODATA SCIENCE 
    Prof: Dr. Ruixiang (Ryan) Tang
    By: Shurti Shukla (NetID: scs250)

    Research Question:
    Which customer features are most associated with telecom customer churn,
    and how effectively can machine learning models predict which customers
    are likely to leave?
'''

import pandas as pd
import sqlite3
import matplotlib.pyplot as plots
import matplotlib.patches as mpatches
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD & PREPROCESS
# Proposal: remove customerID, convert TotalCharges, handle
# missing values, binary + one-hot encoding, derived features
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: DATA LOADING & PREPROCESSING")
print("=" * 60)

dataSet = pd.read_csv("data/Telco-Customer-Churn.csv")

# Drop identifier — not useful for prediction
dataSet.drop("customerID", axis=1, inplace=True)

# Convert to numeric (some TotalCharges are empty strings)
dataSet["TotalCharges"]   = pd.to_numeric(dataSet["TotalCharges"],   errors="coerce")
dataSet["MonthlyCharges"] = pd.to_numeric(dataSet["MonthlyCharges"], errors="coerce")
dataSet.dropna(inplace=True)

print(f"\nDataset shape after cleaning: {dataSet.shape}")
print(f"Missing values remaining: {dataSet.isnull().sum().sum()}")

# Binary encoding: Yes->1, No->0, Male->0, Female->1
encode_toBinary = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in encode_toBinary:
    dataSet[col] = dataSet[col].map({'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1})

# Internet-related encoding: service present->1, not present->0
encode_ForInternet = [
    'InternetService', 'MultipleLines', 'OnlineSecurity',
    'DeviceProtection', 'OnlineBackup', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]
for col in encode_ForInternet:
    dataSet[col] = dataSet[col].map({
        'No': 0, 'Yes': 1, 'Fiber optic': 1, 'DSL': 1,
        'No phone service': 0, 'No internet service': 0
    })

# Feature engineering: total number of services subscribed (proposal: "number of services")
serviceCol = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
dataSet['totalServices'] = dataSet[serviceCol].sum(axis=1)

# Persist to SQLite and reload
conn = sqlite3.connect("telecom_dbase.db")
dataSet.to_sql("customerInfo", conn, if_exists="replace", index=False)
df = pd.read_sql_query("SELECT * FROM customerInfo", conn)

print("\nSample rows:")
print(df.head(3).to_string())
print("\nData types:")
print(df.dtypes.to_string())


# ─────────────────────────────────────────────────────────────
# SECTION 2: EXPLORATORY DATA ANALYSIS
# Proposal: churn distribution, feature relationships, trends
# in tenure, charges, and contract type
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 2.1 Churn distribution ---
plots.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plots.title('Churn Distribution (0 = Stay, 1 = Churn)')
plots.tight_layout()
plots.savefig("plot_01_churn_distribution.png", dpi=150)
plots.show()

print(f"\nChurn rate : {df['Churn'].mean():.4f}  ({df['Churn'].mean()*100:.1f}%)")
print(f"Stay  rate : {1 - df['Churn'].mean():.4f}  ({(1-df['Churn'].mean())*100:.1f}%)")
print("-> Dataset is imbalanced (~27% churn). SMOTE will be applied before training.")

# --- 2.2 Feature histograms ---
df.hist(bins=30, figsize=(14, 10))
plots.suptitle("Feature Distributions", y=1.01)
plots.tight_layout()
plots.savefig("plot_02_feature_histograms.png", dpi=150)
plots.show()

# --- 2.3 Correlation with churn ---
numeric_Data = df.select_dtypes(include=['int64', 'float64'])
corr_churn   = numeric_Data.drop('Churn', axis=1).corrwith(numeric_Data['Churn'])
plots.figure(figsize=(8, 4))
corr_churn.sort_values().plot(kind='bar', grid=True, color='steelblue')
plots.title("Feature Correlation with Churn")
plots.ylabel("Pearson r")
plots.tight_layout()
plots.savefig("plot_03_correlation_churn.png", dpi=150)
plots.show()

print("\nTop correlations with Churn:")
print(corr_churn.sort_values(ascending=False).to_string())
print("-> Tenure is negatively correlated (longer customers less likely to churn).")
print("-> MonthlyCharges positively correlated (higher bill = more likely to leave).")

# --- 2.4 Churn by senior citizen status ---
senior_churn = df.groupby('SeniorCitizen')['Churn'].value_counts().unstack()
labels = ['Stayed', 'Churned']
fig, axes = plots.subplots(1, 2, figsize=(10, 5))
axes[0].pie(senior_churn.loc[1], labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
axes[0].set_title('Senior Citizens')
axes[1].pie(senior_churn.loc[0], labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
axes[1].set_title('Non-Seniors')
fig.suptitle('Churn by Senior Citizen Status')
plots.tight_layout()
plots.savefig("plot_04_senior_churn.png", dpi=150)
plots.show()

senior_rate = df.groupby('SeniorCitizen')['Churn'].mean()
print(f"\nChurn rate -- Seniors: {senior_rate[1]:.2%}  |  Non-seniors: {senior_rate[0]:.2%}")
print("-> Senior citizens churn at a notably higher rate.")

# --- 2.5 Contract type vs churn (proposal highlight) ---
plots.figure(figsize=(7, 4))
sns.countplot(x='Contract', hue='Churn', data=df)
plots.title('Churn by Contract Type')
plots.tight_layout()
plots.savefig("plot_05_contract_churn.png", dpi=150)
plots.show()

contract_churn = df.groupby('Contract')['Churn'].mean()
print("\nChurn rate by Contract Type:")
print(contract_churn.to_string())
print("-> Month-to-month customers churn the most -- no long-term commitment.")

# --- 2.6 Tenure distribution by churn ---
plots.figure(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='dodge', bins=30, palette='Set2')
stay_patch  = mpatches.Patch(color='#66c2a5', label='Stay (0)')
churn_patch = mpatches.Patch(color='#fc8d62', label='Churn (1)')
plots.legend(handles=[stay_patch, churn_patch])
plots.title('Tenure Distribution by Churn Status')
plots.xlabel('Tenure (Months)')
plots.ylabel('Count')
plots.grid(True)
plots.tight_layout()
plots.savefig("plot_06_tenure_churn.png", dpi=150)
plots.show()

tenure_mean = df.groupby('Churn')['tenure'].mean()
print(f"\nAvg tenure -- Stay: {tenure_mean[0]:.1f} months  |  Churn: {tenure_mean[1]:.1f} months")
print("-> Churned customers had significantly shorter tenure on average.")

# --- 2.7 Tenure vs Monthly Charges scatter ---
plots.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn', alpha=0.5)
plots.title('Tenure vs Monthly Charges by Churn')
plots.grid(True)
plots.tight_layout()
plots.savefig("plot_07_tenure_vs_charges.png", dpi=150)
plots.show()

# --- 2.8 Risk flag: short tenure + high charges ---
df['riskFlag'] = ((df['tenure'] < 37) & (df['MonthlyCharges'] > 61)).astype(int)
df.to_sql("customerInfo", conn, if_exists="replace", index=False)
churn_by_risk = df.groupby('riskFlag')['Churn'].mean()
print(f"\nChurn rate -- Low risk: {churn_by_risk[0]:.2%}  |  High risk: {churn_by_risk[1]:.2%}")
print("-> Customers with short tenure AND high monthly charges churn at 51%+.")

# --- 2.9 Correlation heatmap ---
plots.figure(figsize=(10, 8))
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'totalServices', 'riskFlag', 'Churn']
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plots.title('Correlation Heatmap')
plots.tight_layout()
plots.savefig("plot_08_heatmap.png", dpi=150)
plots.show()


# ─────────────────────────────────────────────────────────────
# SECTION 3: DATA PREPARATION
# Proposal: 80/20 split, SMOTE for class imbalance, encoding
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: DATA PREPARATION")
print("=" * 60)

X = df.drop('Churn', axis=1)
y = df['Churn']

# One-hot encode remaining categorical columns (Contract, PaymentMethod)
X = pd.get_dummies(X, columns=['Contract', 'PaymentMethod'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set    : {X_test.shape[0]} samples")
print(f"\nClass balance before SMOTE:\n{y_train.value_counts(normalize=True).to_string()}")

smote = SMOTE(random_state=42)
X_train_s, y_train_s = smote.fit_resample(X_train, y_train)

print(f"\nClass balance after SMOTE:\n{pd.Series(y_train_s).value_counts(normalize=True).to_string()}")
print("-> SMOTE balances training set so models don't just predict Stay every time.")


# ─────────────────────────────────────────────────────────────
# SECTION 4: MODEL TRAINING & EVALUATION
# Proposal: Logistic Regression, Decision Tree, Random Forest
#           + XGBoost; metrics: accuracy, precision, recall,
#           F1, confusion matrix, ROC-AUC
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: MODEL TRAINING & EVALUATION")
print("=" * 60)

# --- Model 1: Logistic Regression (baseline) ---
print("\n-- Logistic Regression --")
logreg = LogisticRegression(max_iter=3000, random_state=42)
logreg.fit(X_train_s, y_train_s)
y_pred_lr = logreg.predict(X_test)
y_prob_lr = logreg.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# --- Model 2: Decision Tree ---
print("\n-- Decision Tree --")
dtree = DecisionTreeClassifier(max_depth=6, random_state=42)
dtree.fit(X_train_s, y_train_s)
y_pred_dt = dtree.predict(X_test)
y_prob_dt = dtree.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# --- Model 3: Random Forest ---
print("\n-- Random Forest --")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_s, y_train_s)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

'''
# --- Model 4: XGBoost (extension from proposal) ---
print("\n-- XGBoost --")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=(y_train_s == 0).sum() / (y_train_s == 1).sum(),
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train_s, y_train_s)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
 
'''

# ─────────────────────────────────────────────────────────────
# SECTION 5: RESULTS VISUALIZATION
# Proposal: recall emphasized, ROC-AUC, confusion matrices,
#           feature importance
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: RESULTS & VISUALIZATIONS")
print("=" * 60)

# --- 5.1 ROC-AUC curves ---
fig, ax = plots.subplots(figsize=(8, 6))
models_roc = [
    ("Logistic Regression", y_prob_lr),
    ("Decision Tree",       y_prob_dt),
    ("Random Forest",       y_prob_rf),
    #("XGBoost",             y_prob_xgb),
]
for name, y_prob in models_roc:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name}  (AUC = {roc_auc:.3f})", lw=2)
    print(f"{name:25s}  AUC = {roc_auc:.3f}")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC-AUC Curves -- All Models")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plots.tight_layout()
plots.savefig("plot_09_roc_auc.png", dpi=150)
plots.show()
print("-> Higher AUC = better ability to distinguish churners from non-churners.")

# --- 5.2 Confusion matrices side by side ---
#fig, axes = plots.subplots(1, 4, figsize=(20, 4))
fig, axes = plots.subplots(1, 3, figsize=(15, 4))
model_preds = [
    ("Logistic Regression", y_pred_lr),
    ("Decision Tree",       y_pred_dt),
    ("Random Forest",       y_pred_rf),
    #("XGBoost",             y_pred_xgb),
]
for ax, (name, y_pred) in zip(axes, model_preds):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Stay', 'Churn'],
                yticklabels=['Stay', 'Churn'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plots.suptitle('Confusion Matrices -- All Models', fontsize=14)
plots.tight_layout()
plots.savefig("plot_10_confusion_matrices.png", dpi=150)
plots.show()

# --- 5.3 Feature importance across all models ---
lr_coef = pd.Series(logreg.coef_[0],            index=X_train.columns).sort_values()
dt_imp  = pd.Series(dtree.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_imp  = pd.Series(rf.feature_importances_,    index=X_train.columns).sort_values(ascending=False)
# xgb_imp = pd.Series(xgb.feature_importances_,   index=X_train.columns).sort_values(ascending=False)

#fig, axes = plots.subplots(1, 4, figsize=(24, 6))
fig, axes = plots.subplots(1, 3, figsize=(18, 6))
lr_coef.plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Logistic Regression\n(Coefficients)')
axes[0].axvline(0, color='black', lw=0.8)

dt_imp.head(15).plot(kind='barh', ax=axes[1], color='lightgreen')
axes[1].set_title('Decision Tree\n(Top-15)')

rf_imp.head(15).plot(kind='barh', ax=axes[2], color='coral')
axes[2].set_title('Random Forest\n(Top-15)')

# xgb_imp.head(15).plot(kind='barh', ax=axes[3], color='orchid')
#axes[3].set_title('XGBoost\n(Top-15)')

fig.suptitle('Feature Importances Across All Models', fontsize=16)
plots.tight_layout()
plots.savefig("plot_11_feature_importance.png", dpi=150)
plots.show()
'''
print("\nTop-10 XGBoost feature importances:")
print(xgb_imp.head(10).to_string())
print("-> Contract type, payment method, and online security are the strongest predictors.")
print("-> This aligns with EDA: month-to-month + electronic check = highest churn.")
'''
conn.close()
print("\n" + "=" * 60)
print("Done. All 11 plots saved as PNG files.")
print("=" * 60)