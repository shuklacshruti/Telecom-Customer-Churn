"""
Data Preprocessing Module
=========================
Handles loading, cleaning, encoding, feature engineering,
train/test splitting, and SMOTE oversampling for the
Telco Customer Churn dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# Data loading / synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the structure and approximate
    distributions of the IBM Telco Customer Churn dataset.

    Parameters
    ----------
    n : int
        Number of customer records to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the same columns as the real Telco dataset.
    """
    rng = np.random.default_rng(random_state)

    # --- Customer identifiers ---
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]

    # --- Demographics ---
    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # --- Tenure (months) with realistic distribution ---
    tenure = np.clip(rng.exponential(scale=32, size=n).astype(int), 1, 72)

    # --- Phone & internet services ---
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def internet_addon(internet, p_yes=0.28):
        return np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes]),
        )

    online_security = internet_addon(internet_service, 0.29)
    online_backup = internet_addon(internet_service, 0.34)
    device_protection = internet_addon(internet_service, 0.34)
    tech_support = internet_addon(internet_service, 0.29)
    streaming_tv = internet_addon(internet_service, 0.38)
    streaming_movies = internet_addon(internet_service, 0.39)

    # --- Contract & billing ---
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # --- Charges correlated with internet service and tenure ---
    base_monthly = np.where(
        internet_service == "No",
        rng.uniform(18, 30, size=n),
        np.where(
            internet_service == "DSL",
            rng.uniform(25, 65, size=n),
            rng.uniform(50, 110, size=n),
        ),
    )
    monthly_charges = np.round(base_monthly, 2)
    total_charges_raw = np.round(monthly_charges * tenure + rng.normal(0, 30, size=n), 2)
    total_charges_raw = np.clip(total_charges_raw, 18.0, None)

    # Introduce ~1% missing values in TotalCharges (as in the real dataset)
    missing_mask = rng.random(n) < 0.011
    total_charges = np.where(missing_mask, "", total_charges_raw.astype(str))

    # --- Churn label (approx 26% churn rate with realistic correlations) ---
    churn_prob = 0.05
    churn_prob = np.where(contract == "Month-to-month", churn_prob + 0.25, churn_prob)
    churn_prob = np.where(internet_service == "Fiber optic", churn_prob + 0.08, churn_prob)
    churn_prob = np.where(tenure < 12, churn_prob + 0.10, churn_prob)
    churn_prob = np.where(tenure > 48, churn_prob - 0.10, churn_prob)
    churn_prob = np.where(paperless_billing == "Yes", churn_prob + 0.03, churn_prob)
    churn_prob = np.clip(churn_prob, 0.02, 0.92)
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


def load_data(filepath: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.

    Falls back to synthetic data if the CSV is not found at *filepath*.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset ready for preprocessing.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Dataset loaded from {filepath}  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
    else:
        print(
            f"[INFO] Dataset not found at '{filepath}'. "
            "Using synthetic data. Download the real dataset from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it at data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
        df = generate_synthetic_data()
        print(f"Synthetic dataset generated  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Cleaning & feature engineering
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco dataset.

    Steps:
    - Drop customerID (non-informative)
    - Convert TotalCharges to numeric; impute missing with median
    - Ensure SeniorCitizen is integer

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.copy()

    # Drop non-informative customer identifier
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges: coerce blanks / strings to NaN, then impute
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    missing_count = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    if missing_count > 0:
        print(f"  Imputed {missing_count} missing TotalCharges values with median ({median_tc:.2f})")

    # Ensure SeniorCitizen is integer
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    print(f"Cleaned dataset shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features.

    New features:
    - num_services : count of add-on services subscribed (0–8)
    - avg_monthly_per_tenure : MonthlyCharges / (tenure + 1)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (before encoding).

    Returns
    -------
    pd.DataFrame
        Dataset with additional feature columns.
    """
    df = df.copy()

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )

    df["avg_monthly_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical features and separate target.

    - Binary Yes/No columns → 1/0
    - Multi-category columns → one-hot encoding (drop first to avoid multicollinearity)
    - Target 'Churn' → 1 (Yes) / 0 (No)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X contains encoded features and y is the binary target.
    """
    df = df.copy()

    # Encode target
    y = (df.pop("Churn") == "Yes").astype(int)

    # Binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int) if col != "gender" else (df[col] == "Male").astype(int)

    # Columns where "No internet service" / "No phone service" map to 0 same as "No"
    service_binary_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_binary_cols:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)

    # Multi-category columns → one-hot
    multi_cat_cols = ["InternetService", "Contract", "PaymentMethod"]
    existing_multi = [c for c in multi_cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_multi, drop_first=True)

    # Ensure all remaining object columns are handled
    remaining_obj = df.select_dtypes(include="object").columns.tolist()
    if remaining_obj:
        df = pd.get_dummies(df, columns=remaining_obj, drop_first=True)

    return df, y


# ---------------------------------------------------------------------------
# Train / test split and SMOTE
# ---------------------------------------------------------------------------

def split_and_resample(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20,
                        random_state: int = 42):
    """
    Split into train/test sets and apply SMOTE to the training set.

    Parameters
    ----------
    X : pd.DataFrame
        Encoded feature matrix.
    y : pd.Series
        Binary target.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train_res, X_test, y_train_res, y_test, scaler)
        where the training data has been SMOTE-resampled and all splits are
        scaled with StandardScaler fitted on the resampled training data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"Train churn rate before SMOTE: {y_train.mean():.2%}")

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Train size after SMOTE: {len(X_train_res):,}")
    print(f"Train churn rate after SMOTE: {y_train_res.mean():.2%}")

    scaler = StandardScaler()
    X_train_res = pd.DataFrame(
        scaler.fit_transform(X_train_res), columns=X_train.columns
    )
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_res, X_test, y_train_res, y_test, scaler


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(filepath: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Execute the full preprocessing pipeline and return model-ready data.

    Parameters
    ----------
    filepath : str
        Path to raw CSV (optional; falls back to synthetic data).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, df_clean)
        where df_clean is the cleaned + feature-engineered (pre-encoded) frame
        useful for EDA.
    """
    df_raw = load_data(filepath)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    X, y = encode_features(df_feat)
    X_train, X_test, y_train, y_test, scaler = split_and_resample(X, y)
    return X_train, X_test, y_train, y_test, scaler, df_feat
