"""
Telecom Customer Churn Analysis and Prediction
===============================================
Shruti Shukla (NetID: scs250)
16 March, 2026

End-to-end script that:
  1. Loads / generates data
  2. Runs Exploratory Data Analysis
  3. Pre-processes data (cleaning, encoding, SMOTE)
  4. Trains Logistic Regression, Decision Tree, and Random Forest
  5. Evaluates all models and saves figures to outputs/figures/
"""

import sys
import os

# Make src importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution

from src.preprocessing import run_preprocessing_pipeline, load_data, clean_data, engineer_features
from src.eda import run_eda
from src.models import run_modelling_pipeline


def main():
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load raw data (EDA uses pre-encoded frame with string labels)    #
    # ------------------------------------------------------------------ #
    df_raw = load_data(data_path)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    # ------------------------------------------------------------------ #
    # 2. Exploratory Data Analysis                                        #
    # ------------------------------------------------------------------ #
    run_eda(df_feat, output_dir=output_dir)

    # ------------------------------------------------------------------ #
    # 3. Pre-processing pipeline (encode + split + SMOTE + scale)        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    from src.preprocessing import encode_features, split_and_resample
    X, y = encode_features(df_feat)
    X_train, X_test, y_train, y_test, scaler = split_and_resample(X, y)

    # ------------------------------------------------------------------ #
    # 4 & 5. Modelling and evaluation                                     #
    # ------------------------------------------------------------------ #
    trained, summary = run_modelling_pipeline(
        X_train, X_test, y_train, y_test, output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All figures saved to: {os.path.abspath(output_dir)}")
    print("\nFinal model comparison (sorted by Recall):")
    print(summary.sort_values("Recall", ascending=False).to_string())


if __name__ == "__main__":
    main()
