"""
Visualization Module for Digital Bank Customer Churn Prediction.
Provides visual diagnostics:
  - Feature Importance Bar Charts
  - Model Comparison Multi-Metric Charts
  - Optuna Hyperparameter Optimization History & Parameter Importance Plots
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Set headless backend for server/script execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Set refined aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_feature_importance(
    df_imp: pd.DataFrame,
    output_path: Union[str, Path],
    model_name: str = "XGBoost",
    top_k: int = 15,
    metric_col: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
) -> str:
    """
    Renders an executive-quality horizontal bar chart of the Top-K feature importances.
    
    Args:
        df_imp: DataFrame containing 'Feature' column and an importance metric column.
        output_path: File path to save the generated PNG.
        model_name: Name of the model (e.g. 'XGBoost', 'Random Forest').
        top_k: Number of top features to display.
        metric_col: Name of column to sort and plot (auto-detected if None).
        figsize: Figure dimensions (width, height).
        
    Returns:
        The resolved output filepath.
    """
    if df_imp is None or df_imp.empty or "Feature" not in df_imp.columns:
        return ""

    df = df_imp.copy()
    
    # Auto-detect importance metric column
    if metric_col is None:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Rank"]
        metric_col = numeric_cols[0] if numeric_cols else "Importance"

    df = df.sort_values(by=metric_col, ascending=False).head(top_k).copy()
    df = df.sort_values(by=metric_col, ascending=True)  # Ascending for horizontal bars

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Palette gradient from dark teal to deep navy
    colors = sns.color_palette("viridis", n_colors=len(df))

    bars = ax.barh(
        df["Feature"],
        df[metric_col],
        color=colors,
        edgecolor="none",
        height=0.65,
        alpha=0.88,
    )

    # Annotate values on bars
    max_val = df[metric_col].max()
    offset = max_val * 0.015
    for bar in bars:
        width = bar.get_width()
        val_str = f"{width:.4f}" if width < 1.0 else f"{width:.1f}"
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2.0,
            val_str,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
            color="#2c3e50",
        )

    ax.set_xlim(0, max_val * 1.15)
    ax.set_title(
        f"Top {len(df)} Features by {metric_col.replace('_', ' ')} — {model_name}",
        fontweight="bold",
        pad=15,
        color="#1a252f",
    )
    ax.set_xlabel(f"Importance ({metric_col.replace('_', ' ')})", labelpad=10)
    ax.set_ylabel("Predictor Feature", labelpad=10)
    sns.despine(left=True, bottom=True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.yaxis.grid(False)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_model_comparison(
    df_results: pd.DataFrame,
    output_path: Union[str, Path],
    figsize: Tuple[int, int] = (11, 6),
) -> str:
    """
    Renders a multi-metric comparative bar chart for models (LR vs RF vs XGBoost).
    """
    if df_results is None or df_results.empty:
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["ROC_AUC", "PR_AUC", "F1_Score", "Recall", "Balanced_Accuracy"]
    avail_metrics = [m for m in metrics if m in df_results.columns]

    model_col = "Model" if "Model" in df_results.columns else "Experiment"

    # Melt for grouped bar chart
    melted = df_results.melt(
        id_vars=[model_col],
        value_vars=avail_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("muted", n_colors=df_results[model_col].nunique())

    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue=model_col,
        palette=palette,
        ax=ax,
        edgecolor="none",
        alpha=0.9,
    )

    ax.set_title("Model Benchmark Comparison Across Critical Churn Metrics", fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Evaluation Score (0 - 1.0)", labelpad=10)
    ax.set_xlabel("")
    ax.legend(title="Model Family", frameon=True, facecolor="white", loc="lower right")
    sns.despine(top=True, right=True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    # Annotate values above bars
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.015),
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=0,
            )

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_optuna_tuning_history(
    study: Any,
    output_path: Union[str, Path],
    metric_name: str = "PR_AUC",
    figsize: Tuple[int, int] = (9, 5),
) -> str:
    """
    Renders Optuna optimization history: trial index vs CV metric objective with running best curve.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trials = [t for t in study.trials if t.value is not None]
    if not trials:
        return ""

    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    
    # Running maximum
    best_values = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(trial_numbers, values, color="#3498db", alpha=0.7, label="Trial Objective Value", s=40)
    ax.plot(trial_numbers, best_values, color="#e74c3c", linewidth=2.2, label=f"Best {metric_name} So Far")

    ax.set_title(f"Optuna Hyperparameter Optimization Progress ({metric_name})", fontweight="bold", pad=15)
    ax.set_xlabel("Trial Number", labelpad=10)
    ax.set_ylabel(f"CV Score ({metric_name})", labelpad=10)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    sns.despine(top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_lift_and_gains_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    output_path: Union[str, Path],
    model_name: str = "XGBoost",
    figsize: Tuple[int, int] = (14, 6),
) -> str:
    """
    Renders a 2-panel business diagnostic chart:
      Panel 1: Cumulative Gains Chart (% of Churners Captured vs % of Customers Contacted)
      Panel 2: Decile Lift Chart (Relative Lift vs Random Guessing per decile)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)

    df_eval = pd.DataFrame({"actual": y_t, "prob": y_p})
    df_eval = df_eval.sort_values(by="prob", ascending=False).reset_index(drop=True)
    df_eval["decile"] = pd.qcut(df_eval.index, q=10, labels=[f"D{i}" for i in range(1, 11)])

    base_rate = float(df_eval["actual"].mean())
    total_churners = float(df_eval["actual"].sum())

    deciles = []
    decile_lifts = []
    cum_gains = [0.0]
    pct_pop = [0.0]

    cum_churn = 0.0
    cum_pop = 0.0
    n_total = len(df_eval)

    for dec, grp in df_eval.groupby("decile", observed=False):
        n_cust = len(grp)
        n_churn = float(grp["actual"].sum())
        cum_churn += n_churn
        cum_pop += n_cust
        
        churn_rate = n_churn / max(n_cust, 1)
        lift = churn_rate / max(base_rate, 1e-5)
        
        deciles.append(str(dec))
        decile_lifts.append(lift)
        cum_gains.append((cum_churn / max(total_churners, 1e-5)) * 100.0)
        pct_pop.append((cum_pop / max(n_total, 1)) * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Cumulative Gains Chart
    ax1.plot(pct_pop, cum_gains, marker="o", color="#2980b9", linewidth=2.5, label=f"{model_name} Model")
    ax1.plot([0, 100], [0, 100], linestyle="--", color="#7f8c8d", linewidth=1.8, label="Random Guess Baseline (1.0x)")
    ax1.fill_between(pct_pop, cum_gains, pct_pop, color="#3498db", alpha=0.15, label="Model Advantage Zone")

    # Annotate Top 10% and Top 20%
    if len(cum_gains) > 2:
        top10_gains = cum_gains[1]
        top20_gains = cum_gains[2]
        ax1.annotate(
            f"Top 10%: {top10_gains:.1f}% Churners ({decile_lifts[0]:.2f}x Lift)",
            xy=(pct_pop[1], top10_gains),
            xytext=(pct_pop[1] + 8, top10_gains - 8),
            arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold",
            fontsize=9.5,
            color="#c0392b",
        )
        ax1.annotate(
            f"Top 20%: {top20_gains:.1f}% Churners",
            xy=(pct_pop[2], top20_gains),
            xytext=(pct_pop[2] + 8, top20_gains - 8),
            arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold",
            fontsize=9.5,
            color="#27ae60",
        )

    ax1.set_title(f"Cumulative Gains Chart ({model_name})", fontweight="bold", pad=12)
    ax1.set_xlabel("% Customers Contacted (Ranked by Risk)", labelpad=10)
    ax1.set_ylabel("% Churners Captured (Cumulative Recall)", labelpad=10)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 105])
    ax1.legend(loc="lower right", frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Decile Lift Chart
    bar_colors = ["#e74c3c" if l >= 2.0 else "#3498db" if l >= 1.0 else "#95a5a6" for l in decile_lifts]
    bars = ax2.bar(deciles, decile_lifts, color=bar_colors, width=0.6, alpha=0.88)
    ax2.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.8, label="Random Guess Baseline (1.0x)")

    for bar, l in zip(bars, decile_lifts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            f"{l:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
        )

    ax2.set_title(f"Decile Lift Chart vs Random Baseline ({model_name})", fontweight="bold", pad=12)
    ax2.set_xlabel("Risk Decile (D1 = Highest Risk, D10 = Safest)", labelpad=10)
    ax2.set_ylabel("Lift Factor (x Baseline)", labelpad=10)
    ax2.set_ylim([0, max(decile_lifts) * 1.25])
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
