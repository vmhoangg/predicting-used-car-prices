# Predicting Used Car Prices  
**MSc Business Analytics – Data Mining & Visualisation (BNM868J)**  
Candidate: 194507  

---

## 1) Project Overview

This project develops machine learning models to predict used car prices using structured vehicle data (2,230 records, 32 variables).

The objective is to build a robust regression model that improves pricing transparency and reduces estimation bias in the second-hand car market.

The problem is formulated as a **supervised regression task**, where the target variable is continuous (Price in USD).

---

## 2) Dataset Summary

- **Observations:** 2,230 vehicles  
- **Features:** 31 input variables + 1 target variable (Price)  
- **Key predictors:**
  - Estimated_Mileage  
  - Car_Age (engineered feature)  
  - Model / Brand  
  - Engine_Size  
  - Service_History  
  - Insurance  
  - Owners  

### Data Preprocessing Steps

- Cleaning numerical formatting (units, commas, special characters)
- Standardizing categorical inconsistencies (e.g., "Petroll" → "Petrol")
- Missing value imputation:
  - Mode (categorical)
  - Median (numerical)
- Feature engineering:
  - Created `Car_Age` from Year
- Outlier detection:
  - Isolation Forest (3% contamination)
  - Removed 67 observations  
  - Final dataset: **2,163 records**

---

## 3) Exploratory Data Analysis (EDA)

Key insights:

- **Strong negative correlation:**  
  Estimated_Mileage → Price (–0.95)

- **Car_Age is a strong nonlinear predictor**

- **Brand significantly affects pricing**

- Service history and insurance status show meaningful price separation.

---

## 4) Models Implemented

| Model | Test RMSE |
|-------|-----------|
| Dummy Regressor | 8703.13 |
| Linear Regression | 2422.91 |
| Decision Tree (tuned) | 2187.01 |
| Random Forest (best) | **2002.15** |
| CatBoost (best) | 2004.18 |
| Bagging Neural Network | 3087.03 |

### Best Model: Random Forest

- Balanced bias–variance trade-off
- Strong generalization
- Lowest test RMSE
- Stable across parameter tuning

---

## 5) Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

RMSE was prioritized due to interpretability in real-world pricing context.

---

## 6) Error Cost Consideration

Two types of prediction error:

**Overestimation**
- Longer listing duration
- Reduced buyer engagement

**Underestimation**
- Direct seller financial loss
- Reduced platform trust

Underestimation is considered more costly and should be minimized in deployment scenarios.

---

## 7) Key Takeaways

- Ensemble tree-based models outperform linear models.
- Feature engineering (Car_Age) significantly improved performance.
- Overfitting observed in deep CatBoost models.
- Neural networks underperformed without extensive tuning.

---

## 8) How to Run

### 8.1 Clone repository

```bash
git clone https://github.com/vmhoangg/used-car-price-ml.git
cd used-car-price-ml
```

### 8.2 Create virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 8.3 Install dependencies
```
pip install -r requirements.txt
```

### 8.4 Run notebook
```
jupyter notebook
```

## 9) Project Structure

```text
used-car-price-ml/
│
├── data/                # Raw and processed dataset files
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── report/              # Final academic report (PDF)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 10) Future improvement
- Implement k-fold cross-validation instead of a single train/test split to improve model robustness.
- Apply hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
- Integrate SHAP (SHapley Additive exPlanations) for model interpretability.
- Add model calibration analysis to assess prediction reliability.
- Deploy the best-performing model using a Streamlit web application for interactive price estimation.
