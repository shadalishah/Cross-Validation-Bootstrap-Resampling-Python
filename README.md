# 🔁 Resampling Methods — Cross-Validation & Bootstrap for Model Reliability

> **Skills Demonstrated:** Cross-Validation · Bootstrap · LOOCV · K-Fold CV · Validation Set Approach · Standard Error Estimation · Logistic Regression · Model Selection · Python · Scikit-learn · Statsmodels

---

## 🎯 Project Overview

This project applies **Resampling Methods** to evaluate and validate machine learning models without overfitting. It answers a critical production question every ML engineer faces:

> *"How do we honestly estimate how well our model will perform on unseen data?"*

Four exercises are covered across three real-world datasets:

1. **Default Dataset** — Estimating test error of credit default prediction via Validation Set & Bootstrap
2. **Weekly Dataset** — Computing LOOCV error for stock market direction classification
3. **Simulated Dataset** — Selecting the best polynomial degree using LOOCV
4. **Boston Dataset** — Bootstrap estimation of population statistics (mean, median, percentile)

---

## 📁 Datasets Used

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| **Default** | Simulated credit card data (ISLP) | 10,000 rows, 4 features | Credit default prediction |
| **Weekly** | S&P 500 weekly returns 1990–2010 (Real) | 1,089 rows, 9 features | Market direction classification |
| **Boston** | U.S. Census Bureau (Real) | 506 rows, 13 features | Population mean/median estimation |
| **Simulated** | Generated via `numpy` | 100 observations | Polynomial model selection |

---

## 🔧 Techniques & Tools Applied

| Technique | Purpose |
|-----------|---------|
| Validation Set Approach | Simple train/test split error estimation |
| Leave-One-Out Cross-Validation (LOOCV) | Low-bias error estimate, no randomness |
| K-Fold Cross-Validation | Bias-variance balanced error estimation |
| Bootstrap (B=1000, B=10000) | SE estimation when no formula exists |
| Logistic Regression (GLM Binomial) | Default & market direction prediction |
| Polynomial Model Comparison | Selecting optimal model complexity |
| `cross_validate()` + `sklearn_sm` | Scikit-learn & Statsmodels integration |

**Libraries:** `numpy` · `pandas` · `statsmodels` · `scikit-learn` · `matplotlib` · `ISLP`

---

## 📊 Key Results

### Exercise 5 — Credit Default Test Error (Validation Set Approach)

**Dataset:** Default (n=10,000) | **Predictors:** income + balance | **Split:** 50/50

| Split | Validation Set Error |
|-------|---------------------|
| Split 1 (random_state=0) | **3.98%** |

**With student dummy variable added:**

| Model | Predictors | Validation Error |
|-------|-----------|-----------------|
| Without student | income + balance | 3.98% |
| With student | income + balance + student | Similar — student does **not** reduce error |

> **Finding:** Adding the student dummy variable does **not** meaningfully reduce test error. Income and balance alone are sufficient predictors of credit default risk — student status adds no additional signal.

---

### Exercise 6 — Bootstrap vs Formula Standard Errors (Default Dataset)

**Full model coefficients:** income + balance → default

| Coefficient | GLM Formula SE | Bootstrap SE (B=1000) | Match? |
|-------------|---------------|----------------------|--------|
| Intercept | **0.4348** | **0.4253** | ✅ ~Same |
| Balance | **0.000227** | **0.000227** | ✅ Identical |
| Income | **0.000005** | **0.000005** | ✅ Identical |

> **Key Finding:** Bootstrap and formula-based standard errors are virtually identical, confirming the GLM assumptions are well-satisfied for this dataset. Bootstrap is valuable when these assumptions cannot be verified.

---

### Exercise 7 — LOOCV Error for Stock Market Direction (Weekly Dataset)

**Model:** Logistic Regression with Lag1 + Lag2 → Direction (Up/Down)

**Full model coefficients (all 1,089 observations):**

| Predictor | Coefficient | p-value | Significant? |
|-----------|-------------|---------|--------------|
| Intercept | 0.2212 | 0.000 | ✅ |
| Lag1 | -0.0387 | 0.140 | ❌ |
| **Lag2** | **0.0602** | **0.023** | ✅ |

**First observation prediction (leave-one-out):**
- Predicted direction: **"Up"** ✅ (correctly classified)

| Metric | Value |
|--------|-------|
| **LOOCV Test Error** | **44.99%** |
| LOOCV Accuracy | ~55% |

> **Finding:** LOOCV error of ~45% confirms logistic regression with Lag1+Lag2 only slightly outperforms random guessing (50%) for stock market direction — consistent with the Efficient Market Hypothesis. The model lacks sufficient flexibility for the true Bayes decision boundary.

---

### Exercise 8 — Polynomial Model Selection with LOOCV (Simulated Data)

**True model:** Y = X − 2X² + ε &nbsp;&nbsp; (n=100, p=2)

**LOOCV Errors by Polynomial Degree:**

| Model | Polynomial Degree | LOOCV Error | Best? |
|-------|------------------|-------------|-------|
| Model i | Degree 1 (Linear) | **6.633** | ❌ High — underfits |
| **Model ii** | **Degree 2 (Quadratic)** | **1.123** | **🏆 Best** |
| Model iii | Degree 3 (Cubic) | 1.302 | ❌ Slight overfit |
| Model iv | Degree 4 (Quartic) | 1.332 | ❌ Overfit |

**Seed independence confirmed:** LOOCV gives identical results with different random seeds (6.633, 1.123, 1.302, 1.332) — because LOOCV is deterministic (no randomness in observation selection).

**Statistical significance of polynomial terms (Degree-4 model):**

| Term | Coefficient | p-value | Significant? |
|------|-------------|---------|--------------|
| x¹ | 16.60 | 0.000 | ✅ |
| **x²** | **-22.22** | **0.000** | ✅ |
| x³ | -1.08 | 0.575 | ❌ |
| x⁴ | 2.27 | 0.237 | ❌ |

> **Key Finding:** LOOCV correctly identifies the quadratic model (degree 2) as optimal — matching the true data generating process. Only x¹ and x² are statistically significant, fully agreeing with cross-validation results.

---

### Exercise 9 — Bootstrap Estimation of Boston Housing Statistics

**Target variable:** `medv` (median home value, 506 Boston suburbs)

#### Population Mean Estimation

| Method | Estimate | Standard Error |
|--------|----------|---------------|
| **Sample Mean (µ̂)** | **22.533** | — |
| Formula SE | — | **0.4089** |
| **Bootstrap SE (B=1,000)** | — | **0.4048** |

**95% Bootstrap Confidence Interval for µ:**
```
[21.723, 23.342]
```

#### Median & Percentile Estimation (Bootstrap)

| Statistic | Estimate | Bootstrap SE (B iterations) |
|-----------|----------|----------------------------|
| **Sample Median (µ̂_med)** | **21.200** | **0.3807** (B=10,000) |
| **10th Percentile (µ̂_0.1)** | **12.750** | **0.5004** (B=5,000) |

> **Key Finding:** Bootstrap SE for the median (0.381) is **smaller** than for the mean (0.405), indicating the `medv` distribution is non-normal (right-skewed). Bootstrap provides reliable SE estimates for statistics like the median and percentiles, where no closed-form formula exists.

---

## 💡 Business Insights

1. **Model Validation Prevents Overconfidence:** The Default model shows 3.98% error on validation — but repeating with different splits reveals variability. In production credit scoring, using K-Fold CV over a single validation split gives a more reliable risk estimate.

2. **Bootstrap Validates Model Assumptions:** When Bootstrap SEs match GLM formula SEs exactly (as in Exercise 6), it confirms the model assumptions hold. Divergence would signal model misspecification — a critical check before deployment.

3. **LOOCV Catches Overfitting Automatically:** Polynomial degree 2 was selected by LOOCV with error 1.123, versus degree 4 at 1.332 — cross-validation correctly penalizes unnecessary complexity without human intervention.

4. **Uncertainty Quantification for Business Decisions:** The 95% bootstrap CI for Boston median home value [21.72, 23.34] gives decision-makers a range, not just a point estimate — essential for risk-aware real estate investment models.

---

## 🗂️ File Structure

```
Chapter_5_Applied_Exercise_Solutions/
│
├── Chapter_5.ipynb          ← Main analysis notebook (all exercises)
├── Chapter_5.html           ← Rendered HTML version (easy browser viewing)
├── Chapter_5.qmd            ← Quarto source file
└── README.md                ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP scikit-learn statsmodels pandas numpy matplotlib

# Launch notebook
jupyter notebook Chapter_5.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 5: Resampling Methods — Applied Exercises 5–9.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM) provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
