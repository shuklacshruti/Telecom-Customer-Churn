"""
Model Training & Evaluation Module
====================================
Trains Logistic Regression, Decision Tree, and Random Forest classifiers
on the pre-processed Telco Customer Churn data and evaluates them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

CHURN_COLORS = ["#4C72B0", "#DD8452"]


def _save_fig(fig, name: str, output_dir: str = "outputs/figures") -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(random_state: int = 42) -> dict:
    """
    Return a dictionary of model name → untrained estimator.

    Models
    ------
    - Logistic Regression  (max_iter=1000 for convergence)
    - Decision Tree        (max_depth=10 to reduce overfitting)
    - Random Forest        (100 estimators)
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit each model on the training data.

    Parameters
    ----------
    models : dict
        {name: estimator} from :func:`get_models`.
    X_train, y_train : training data.

    Returns
    -------
    dict
        {name: fitted_estimator}
    """
    trained = {}
    for name, model in models.items():
        print(f"Training {name} …", end=" ")
        model.fit(X_train, y_train)
        trained[name] = model
        print("done")
    return trained


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute evaluation metrics for a single fitted model.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, y_pred, y_prob
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else np.zeros(len(y_test))
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_all_models(trained: dict, X_test: pd.DataFrame,
                        y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate every trained model and return a summary DataFrame.

    Parameters
    ----------
    trained : dict
        {name: fitted_estimator}
    X_test, y_test : test data.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = accuracy / precision / recall / f1 / roc_auc.
    """
    results = {}
    for name, model in trained.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
        }
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(classification_report(y_test, metrics["y_pred"],
                                    target_names=["No Churn", "Churn"]))

    summary = pd.DataFrame(results).T
    return summary


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrices(trained: dict, X_test: pd.DataFrame, y_test: pd.Series,
                             output_dir: str = "outputs/figures") -> None:
    """Plot a confusion matrix for each model."""
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (name, model) in zip(axes, trained.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax, linewidths=0.5,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    _save_fig(fig, "09_confusion_matrices", output_dir)
    plt.show()


def plot_roc_curves(trained: dict, X_test: pd.DataFrame, y_test: pd.Series,
                    output_dir: str = "outputs/figures") -> None:
    """Plot ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    for name, model in trained.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_fig(fig, "10_roc_curves", output_dir)
    plt.show()


def plot_metrics_comparison(summary: pd.DataFrame,
                             output_dir: str = "outputs/figures") -> None:
    """Grouped bar chart comparing all metrics across models."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    existing = [m for m in metrics_to_plot if m in summary.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    summary[existing].plot(kind="bar", ax=ax, edgecolor="white", rot=0)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)

    plt.tight_layout()
    _save_fig(fig, "11_metrics_comparison", output_dir)
    plt.show()


def plot_feature_importance(trained: dict, feature_names: list,
                             output_dir: str = "outputs/figures",
                             top_n: int = 20) -> None:
    """
    Plot feature importances for tree-based models and coefficients for
    Logistic Regression.
    """
    for name, model in trained.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            continue

        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        fi_top = fi.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        fi_top[::-1].plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Feature Importance – {name} (Top {top_n})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        _save_fig(fig, f"12_feature_importance_{safe_name}", output_dir)
        plt.show()


# ---------------------------------------------------------------------------
# Full modelling pipeline
# ---------------------------------------------------------------------------

def run_modelling_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series,
                            output_dir: str = "outputs/figures",
                            random_state: int = 42):
    """
    Train all models, evaluate, and generate all plots.

    Returns
    -------
    tuple
        (trained_models, summary_df)
    """
    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    models = get_models(random_state=random_state)
    trained = train_models(models, X_train, y_train)
    summary = evaluate_all_models(trained, X_test, y_test)

    print("\n--- Performance Summary ---")
    print(summary.to_string())

    plot_confusion_matrices(trained, X_test, y_test, output_dir)
    plot_roc_curves(trained, X_test, y_test, output_dir)
    plot_metrics_comparison(summary, output_dir)
    plot_feature_importance(trained, list(X_train.columns), output_dir)

    return trained, summary
