# Quantitative-Macro-Credit-Risk-Modeling
---------------


This project explores how incorporating **macroeconomic variables** and **market regimes** can improve the prediction of **Probability of Default (PD)** in credit risk models.

Traditional credit scoring models rely mainly on borrower-specific features and assume stable relationships over time. However, real-world evidence shows that credit risk is highly sensitive to economic conditions and behaves differently across market regimes.

This project builds a **macro-conditional and regime-aware machine learning framework** to better capture these dynamics.

---

## 🎯 Objectives

The main goal is to test the hypothesis:

> **Macroeconomic and market regime information significantly improves predictive accuracy of PD models**

To do this, we compare three types of models:

1. **Baseline Model**
   - Uses only borrower-level features

2. **Macro-Conditional Model**
   - Adds macroeconomic variables

3. **Regime-Aware Model**
   - Incorporates both macro variables and latent market regimes

---

## 🧠 Methodology

### 1. Data Sources

- **Borrower Data**
  - Lending Club dataset (2007–2018)

- **Macroeconomic Data (FRED API)**
  - Federal Funds Rate (FEDFUNDS)
  - Unemployment Rate (UNRATE)
  - CPI (CPIAUCSL)
  - High Yield Spread (BAMLH0A0HYM2)
  - GDP (GDPC1)

---

### 2. Feature Engineering

- Financial ratios (leverage, liquidity, etc.)
- Time alignment (monthly frequency)
- Macro transformations (lags, differences)
- Spread and volatility features

---

### 3. Modeling

#### Baseline Models
- Logistic Regression
- Gradient Boosting (XGBoost / LightGBM)

#### Macro-Conditional Models
- Same models + macro variables

#### Regime Detection
- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- K-Means Clustering

#### Regime-Aware Models
- Models conditioned on detected regimes

---

## 📈 Evaluation Framework

We evaluate models across multiple dimensions:

| Dimension       | Metric            | Purpose |
|----------------|------------------|--------|
| Discrimination | ROC-AUC, PR-AUC  | Ranking performance |
| Calibration    | Brier Score      | Probability accuracy |
| Calibration    | Calibration Curve| Predicted vs actual PD |
| Robustness     | Per-regime AUC   | Stability across regimes |
| Generalization | Time-based OOS   | Out-of-sample performance |
| Stress Testing | Scenario Analysis| Behavior under shocks |

---

## ⚠️ Key Challenges

- Aligning borrower and macro data over time  
- Handling non-stationary economic variables  
- Interpreting regime detection outputs  
- Balancing model performance and interpretability  

---

## 🚀 Key Contributions

- Integrates **macro variables directly into ML models** (not as overlays)
- Uses **unsupervised learning** to detect market regimes
- Evaluates models across **different economic states**
- Provides a **reproducible, open-source framework**
- Includes **stress testing under crisis scenarios**

