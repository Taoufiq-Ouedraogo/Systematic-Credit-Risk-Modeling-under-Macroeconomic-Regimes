"""
backtesting.py

Walk-forward backtesting framework for macro credit risk models
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED = 42
np.random.seed(SEED)





def walk_forward_backtest_TS(
    framework_df, model, n_splits=5,
):
    df = framework_df.sort_values("date").reset_index(drop=True)

    X = df.drop(columns=["default", "date"])
    y = df["default"]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        # Skip invalid folds
        if len(y_test) < 10 or y_test.nunique() < 2:
            continue

        # Train
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
        
        results.append({
            "fold": fold_idx + 1,
            "auc": roc_auc_score(y_test, y_prob),
            "avg_precision": average_precision_score(y_test, y_prob),
            "brier": brier_score_loss(y_test, y_prob)
        })
    return pd.DataFrame(results)




def plot_backtest_results(summary, title="Model (Framework)"):
    folds = summary["fold"]
    # AUC
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Walk-Forward Backtest - {title}", fontweight="bold")
    axes[0].plot(folds, summary["auc"], marker="o",
        linewidth=2, color="steelblue", label="AUC")
    axes[0].axhline(summary["auc"].mean(), linestyle="--",
        color="black", alpha=0.6, label=f"Mean = {summary['auc'].mean():.3f}")
    axes[0].set_title("AUC")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend()
    # Average Precision
    axes[1].plot(folds, summary["avg_precision"], marker="o", 
        linewidth=2, color="darkorange", label="Avg Precision")
    axes[1].axhline(summary["avg_precision"].mean(), linestyle="--", color="black",
        alpha=0.6, label=f"Mean = {summary['avg_precision'].mean():.3f}")
    axes[1].set_title("Precision (Rare Event Focus)")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Avg Precision")
    axes[1].legend()
    # Brier Score
    axes[2].plot(folds, summary["brier"], marker="o",
        linewidth=2, color="seagreen", label="Brier Score")
    axes[2].axhline(summary["brier"].mean(), linestyle="--", color="black",
        alpha=0.6, label=f"Mean = {summary['brier'].mean():.4f}")
    axes[2].set_title("PD Calibration Quality (Brier Score)")
    axes[2].set_xlabel("Fold")
    axes[2].set_ylabel("Brier Score")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(folds)
    plt.tight_layout()
    plt.show()



def collect_model_across_frameworks(backtest_result, model_name):
    data = {}
    for framework, models_dict in backtest_result.items():
        if model_name in models_dict:
            data[framework] = models_dict[model_name]
    return data



def plot_model_across_frameworks(backtest_result, model_name):
    data = collect_model_across_frameworks(backtest_result, model_name)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Framework Comparison - {model_name}", fontweight="bold")
    metrics = ["auc", "avg_precision", "brier"]
    titles = ["AUC", "Average Precision", "Brier Score"]
    colors = ["steelblue", "darkorange", "seagreen"]
    frameworks = list(data.keys())
    for i, metric in enumerate(metrics):
        for fw in frameworks:
            df = data[fw]
            axes[i].plot(df["fold"], df[metric], marker="o", label=fw)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Fold")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    plt.tight_layout()
    plt.show()