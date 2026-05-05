"""
Exploratory Data Analysis (EDA) Module
=======================================
Generates visualisations and summary statistics for the
Telco Customer Churn dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent colour palette
CHURN_PALETTE = {"No": "#4C72B0", "Yes": "#DD8452"}
CHURN_COLORS = ["#4C72B0", "#DD8452"]


def _save_fig(fig, name: str, output_dir: str = "outputs/figures") -> None:
    """Save a matplotlib figure to *output_dir* as a tight-layout PNG."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Churn distribution
# ---------------------------------------------------------------------------

def plot_churn_distribution(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Bar + pie chart of overall churn rate."""
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(churn_counts.index, churn_counts.values,
                       color=CHURN_COLORS, edgecolor="white", width=0.5)
    axes[0].set_title("Count")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Customers")
    for bar, pct in zip(bars, churn_pct.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=11,
        )

    # Pie chart
    axes[1].pie(
        churn_counts.values,
        labels=churn_counts.index,
        colors=CHURN_COLORS,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Proportion")

    plt.tight_layout()
    _save_fig(fig, "01_churn_distribution", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 2. Demographic features vs churn
# ---------------------------------------------------------------------------

def plot_demographic_features(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Bar plots for gender, SeniorCitizen, Partner, and Dependents vs churn."""
    demo_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Demographic Features vs Churn", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, demo_cols):
        ct = (
            df.groupby([col, "Churn"])
            .size()
            .reset_index(name="count")
        )
        sns.barplot(data=ct, x=col, y="count", hue="Churn",
                    palette=CHURN_PALETTE, ax=ax, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(title="Churn")

    plt.tight_layout()
    _save_fig(fig, "02_demographic_features", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Service features vs churn
# ---------------------------------------------------------------------------

def plot_service_features(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Stacked-percentage bar charts for service-related features."""
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle("Service Features vs Churn", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, existing):
        ct_pct = (
            df.groupby([col, "Churn"])
            .size()
            .unstack(fill_value=0)
            .apply(lambda r: r / r.sum() * 100, axis=1)
        )
        ct_pct.plot(kind="bar", stacked=True, ax=ax,
                    color=CHURN_COLORS, edgecolor="white", rot=30)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("% Customers")
        ax.legend(title="Churn", loc="upper right")

    for ax in axes[len(existing):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save_fig(fig, "03_service_features", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Contract & billing features vs churn
# ---------------------------------------------------------------------------

def plot_contract_billing(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Bar charts for Contract, PaperlessBilling, and PaymentMethod vs churn."""
    cols = ["Contract", "PaperlessBilling", "PaymentMethod"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Contract & Billing Features vs Churn", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, cols):
        ct = (
            df.groupby([col, "Churn"])
            .size()
            .reset_index(name="count")
        )
        sns.barplot(data=ct, x=col, y="count", hue="Churn",
                    palette=CHURN_PALETTE, ax=ax, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Churn")

    plt.tight_layout()
    _save_fig(fig, "04_contract_billing", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Numeric features distribution
# ---------------------------------------------------------------------------

def plot_numeric_distributions(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """KDE plots for tenure, MonthlyCharges, and TotalCharges by churn."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    existing = [c for c in num_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(existing), figsize=(6 * len(existing), 5))
    if len(existing) == 1:
        axes = [axes]
    fig.suptitle("Numeric Feature Distributions by Churn", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, existing):
        for churn_val, color in zip(["No", "Yes"], CHURN_COLORS):
            subset = df[df["Churn"] == churn_val][col].dropna()
            ax.hist(subset, bins=40, alpha=0.55, color=color, label=f"Churn={churn_val}",
                    density=True, edgecolor="white")
            subset.plot.kde(ax=ax, color=color, linewidth=2)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(title="Churn")

    plt.tight_layout()
    _save_fig(fig, "05_numeric_distributions", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Heatmap of correlations among numeric features."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "Churn" in df.columns:
        num_df["Churn"] = (df["Churn"] == "Yes").astype(int)

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "06_correlation_heatmap", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Number of services vs churn
# ---------------------------------------------------------------------------

def plot_num_services(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Box plot and count plot for num_services vs churn."""
    if "num_services" not in df.columns:
        print("num_services column not found – skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Number of Services vs Churn", fontsize=15, fontweight="bold")

    sns.boxplot(data=df, x="Churn", y="num_services",
                hue="Churn", palette=CHURN_PALETTE, legend=False, ax=axes[0])
    axes[0].set_title("Distribution")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Services")

    ct = df.groupby(["num_services", "Churn"]).size().reset_index(name="count")
    sns.barplot(data=ct, x="num_services", y="count", hue="Churn",
                palette=CHURN_PALETTE, ax=axes[1], edgecolor="white")
    axes[1].set_title("Count by Service Level")
    axes[1].set_xlabel("Number of Services")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Churn")

    plt.tight_layout()
    _save_fig(fig, "07_num_services", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# 8. Tenure vs monthly charges scatter
# ---------------------------------------------------------------------------

def plot_tenure_charges_scatter(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Scatter plot of tenure vs MonthlyCharges coloured by churn."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for churn_val, color in zip(["No", "Yes"], CHURN_COLORS):
        subset = df[df["Churn"] == churn_val]
        ax.scatter(
            subset["tenure"], subset["MonthlyCharges"],
            alpha=0.3, s=15, color=color, label=f"Churn={churn_val}",
        )
    ax.set_title("Tenure vs Monthly Charges by Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges ($)")
    ax.legend(title="Churn")
    plt.tight_layout()
    _save_fig(fig, "08_tenure_charges_scatter", output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# Full EDA pipeline
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """
    Run all EDA visualisations in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned + feature-engineered dataset (with Churn as Yes/No strings).
    output_dir : str
        Directory where figures are saved.
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print(f"\nChurn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  None")

    print("\n--- Numeric Summary ---")
    print(df.describe().T.to_string())

    print("\n--- Generating Plots ---")
    plot_churn_distribution(df, output_dir)
    plot_demographic_features(df, output_dir)
    plot_service_features(df, output_dir)
    plot_contract_billing(df, output_dir)
    plot_numeric_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_num_services(df, output_dir)
    plot_tenure_charges_scatter(df, output_dir)
    print("\nEDA complete.")
