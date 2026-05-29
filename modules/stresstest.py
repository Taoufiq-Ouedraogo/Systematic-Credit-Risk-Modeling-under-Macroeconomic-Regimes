"""
stress_test.py

Macro stress testing + sensitivity analysis for credit risk models
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



plt.style.use("seaborn-v0_8-whitegrid")


# SCENARIOS
STRESS_SCENARIOS = {
    # BASELINE
    "Baseline": {
        "fed_funds": 0.0,
        "treasury_2y": 0.0,
        "treasury_10y": 0.0,
        "unemp_rate": 0.0,
        "cpi": 0.0,
        "real_gdp": 0.0,
        "ind_prod": 0.0,
        "credit_spread": 0.0,
    },

    # 2008 GLOBAL FINANCIAL CRISIS  
    "2008 Financial Crisis": {
        "fed_funds": -4.0,
        "treasury_2y": -3.0,
        "treasury_10y": -2.5,
        "unemp_rate": +4.0,
        "cpi": -1.5,
        "real_gdp": -4.5,
        "ind_prod": -5.0,
        "credit_spread": +6.0,
    },

    # COVID-19 SHOCK  
    "COVID Shock": {
        "fed_funds": -2.0,
        "treasury_2y": -1.5,
        "treasury_10y": -1.0,
        "unemp_rate": +3.5,
        "cpi": -1.0,
        "real_gdp": -6.0,
        "ind_prod": -6.5,
        "credit_spread": +5.0,
    },

    # EUROPEAN SOVEREIGN DEBT CRISIS  
    "Sovereign Debt Crisis": {
        "fed_funds": -1.0,
        "treasury_2y": -1.5,
        "treasury_10y": -1.8,
        "unemp_rate": +2.5,
        "cpi": +0.5,
        "real_gdp": -2.5,
        "ind_prod": -3.0,
        "credit_spread": +7.0,
    },

    # 1970s OIL SHOCK / STAGFLATION REGIME
     "Oil Shock Stagflation": {
        "fed_funds": +3.0,
        "treasury_2y": +2.0,
        "treasury_10y": +1.5,
        "unemp_rate": +3.0,
        "cpi": +6.0,
        "real_gdp": -2.0,
        "ind_prod": -3.5,
        "credit_spread": +3.0,
    },

     # EMERGING MARKET / RISK-OFF CRISIS
     "EM Crisis (Risk-Off)": {
        "fed_funds": -1.0,
        "treasury_2y": -2.0,
        "treasury_10y": -2.5,
        "unemp_rate": +2.0,
        "cpi": -0.5,
        "real_gdp": -3.0,
        "ind_prod": -3.0,
        "credit_spread": +5.5,
    },

    # GENERIC MACRO REGIMES  
    "Mild Recession": {
        "fed_funds": -1.0,
        "treasury_2y": -0.8,
        "treasury_10y": -0.5,
        "unemp_rate": +1.5,
        "cpi": +0.3,
        "real_gdp": -1.0,
        "ind_prod": -1.2,
        "credit_spread": +1.2,
    },

    "Severe Recession": {
        "fed_funds": -2.5,
        "treasury_2y": -2.0,
        "treasury_10y": -1.5,
        "unemp_rate": +3.5,
        "cpi": -0.5,
        "real_gdp": -3.0,
        "ind_prod": -3.5,
        "credit_spread": +3.5,
    },

    "Rate Hike Cycle": {
        "fed_funds": +2.5,
        "treasury_2y": +2.8,
        "treasury_10y": +1.8,
        "unemp_rate": +0.5,
        "cpi": +1.2,
        "real_gdp": -0.5,
        "ind_prod": -0.3,
        "credit_spread": +0.8,
    },

    "V-Shaped Recovery": {
        "fed_funds": +0.5,
        "treasury_2y": +0.3,
        "treasury_10y": +0.5,
        "unemp_rate": -1.5,
        "cpi": +0.5,
        "real_gdp": +2.5,
        "ind_prod": +2.0,
        "credit_spread": -1.0,
    },
}



# STRESS TESTING
def stress_test_macro_scenarios(model, X_test, scenarios=STRESS_SCENARIOS):
    base_probs = model.predict_proba(X_test)[:, 1]
    baseline_mean_pd = np.mean(base_probs)
    baseline_median_pd = np.median(base_probs)
    baseline_p95_pd = np.percentile(base_probs, 95)
    results = {}
    for scenario_name, shocks in scenarios.items():
        X_stressed = X_test.copy()
        # apply macro shocks
        for col, delta in shocks.items():
            if col in X_stressed.columns:
                X_stressed[col] = X_stressed[col] + delta
        stressed_probs = model.predict_proba(X_stressed)[:, 1]
        # stressed distribution
        stressed_mean_pd = np.mean(stressed_probs)
        stressed_median_pd = np.median(stressed_probs)
        stressed_p95_pd = np.percentile(stressed_probs, 95)
        # shifts vs baseline
        mean_pd_change = stressed_mean_pd - baseline_mean_pd
        median_pd_change = stressed_median_pd - baseline_median_pd
        p95_change = stressed_p95_pd - baseline_p95_pd
        mean_pd_change_pct = (
            (stressed_mean_pd / (baseline_mean_pd + 1e-12)) - 1
        ) * 100
        # distribution distance (shape shift proxy)
        mean_absolute_distribution_shift = np.mean(
            np.abs(stressed_probs - base_probs)
        )
        results[scenario_name] = {
            # central tendency
            "stressed_mean_probability_of_default": stressed_mean_pd,
            "stressed_median_probability_of_default": stressed_median_pd,
            # tail risk
            "stressed_95th_percentile_probability_of_default": stressed_p95_pd,
            # absolute changes
            "mean_PD_shift_from_baseline": mean_pd_change,
            "median_PD_shift_from_baseline": median_pd_change,
            "95th_percentile_PD_shift_from_baseline": p95_change,
            # relative change
            "relative_change_in_mean_PD_from_baseline": mean_pd_change_pct,
            # distribution shape shift
            "mean_absolute_PD_shift_from_baseline": mean_absolute_distribution_shift,
        }
    return pd.DataFrame(results).T



# SENSITIVITY ANALYSIS
def stress_sensitivity_analysis(
    model, X_test, macro_cols, shocks=np.arange(-3, 3.5, 0.5)
):
    base_pd_mean = model.predict_proba(X_test)[:, 1].mean()
    records = []
    for feature in macro_cols:
        if feature not in X_test.columns:
            continue

        for shock_value in shocks:
            X_tmp = X_test.copy()
            X_tmp[feature] = X_tmp[feature] + shock_value
            stressed_pd_mean = model.predict_proba(X_tmp)[:, 1].mean()
            pd_change_from_baseline = stressed_pd_mean - base_pd_mean
            pd_relative_change_from_baseline_pct = (
                (stressed_pd_mean / (base_pd_mean + 1e-12)) - 1
            ) * 100
            records.append({
                "macro_variable": feature,
                "shock_size": shock_value,
                # level
                "stressed_mean_probability_of_default": stressed_pd_mean,
                # absolute sensitivity
                "change_in_mean_pd_from_baseline": pd_change_from_baseline,
                # relative sensitivity
                "relative_change_in_mean_pd_from_baseline_percent": pd_relative_change_from_baseline_pct,
            })
    return pd.DataFrame(records)



def plot_stress_test_scenarios(scenarios_df):
    plt.figure(figsize=(12, 5))
    sns.heatmap(scenarios_df, cmap="RdBu_r", annot=True, fmt=".1f", linewidths=0.5)
    plt.title("Macro Stress Test Scenarios (Heatmap)")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_stress_heatmap(stress_df, title=None):
    baseline = stress_df.loc["Baseline"]
    color_df = (stress_df - baseline) / (baseline + 1e-12)
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        color_df, cmap="RdBu_r", center=0, annot=stress_df, fmt=".2f", linewidths=0.5
    )
    plt.title(title or "Stress Test Results")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_sensitivizty_curves(
    sensitivity_df,
    value_col="change_in_mean_pd_from_baseline",
    title="Macro Sensitivity Analysis (PD Response Curves)"
):
    fig, ax = plt.subplots(figsize=(12, 5))
    for var in sensitivity_df["macro_variable"].unique():
        df_var = sensitivity_df[sensitivity_df["macro_variable"] == var]
        df_var = df_var.sort_values("shock_size")
        ax.plot(df_var["shock_size"], df_var[value_col], label=var, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Shock Size")
    ax.set_ylabel("Change in Mean PD vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_sensitivity_curves(
    sensitivity_df, ax,
    value_col="change_in_mean_pd_from_baseline",
    title="Macro Sensitivity Analysis (PD Response Curves)"
):
    for var in sensitivity_df["macro_variable"].unique():
        df_var = sensitivity_df[sensitivity_df["macro_variable"] == var]
        df_var = df_var.sort_values("shock_size")
        ax.plot(df_var["shock_size"], df_var[value_col], label=var, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Shock Size")
    ax.set_ylabel("Change in Mean PD vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
