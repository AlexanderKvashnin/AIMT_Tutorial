# 🔬 Data Analysis and Statistics in Materials Science
## Predicting Material Hardness with Machine Learning

Welcome to this laboratory tutorial! In this lab, you will walk through a complete data science pipeline applied to a materials science dataset. The dataset contains physical, mechanical, and electronic descriptors of various materials, along with their measured **hardness**. Your goal is to learn how to analyze the data and build machine learning (ML) models that predict hardness from these descriptors.

---

## 📚 Learning Objectives

By the end of this tutorial, you will be able to:

1. Perform **primary data analysis**: load data, inspect structure, identify feature types, check for missing values and outliers.
2. Perform **correlation analysis** using Pearson and Spearman coefficients.
3. Apply **statistical tests** and interpret distributions of features.
4. Prepare data for **machine learning models** (encoding, scaling, train/test split).
5. Build and evaluate several ML models (Ridge, Decision Tree, KNN, Random Forest, XGBoost).
6. Interpret model results with **feature importance** and **SHAP**.

The dataset (`train_hardness.txt`) is tab-separated and contains **54 materials** described by the following columns:

| Column | Meaning |
|---|---|
| `materials` | Material identifier (e.g., `NaCl`, `SiC`, `TiN`) |
| `Hardness` | **Target variable** — Vickers hardness (GPa) |
| `Density` | Density (g/cm³) |
| `Bulk_voigt`, `Bulk_reuss`, `Bulk_vrh` | Bulk modulus (Voigt, Reuss, Voigt–Reuss–Hill averages, GPa) |
| `Shear_voig`, `Shear_reuss`, `Shear_vrh` | Shear modulus (three averaging schemes, GPa) |
| `elasticity` | Elastic anisotropy factor |
| `poisson` | Poisson's ratio |
| `Youngs` | Young's modulus (GPa) |
| `Fr`, `Rx`, `Rn`, `RW` | Electronegativity-based descriptors |
| `Ax`, `An`, `AW` | Electron-affinity-based descriptors |
| `IEx`, `IEn`, `IEW` | Ionization-energy-based descriptors |

---

## 0. Installing the Required Libraries

Before we start, we install the libraries we will need. If you already have them, `pip` will simply confirm they are installed.

```python
! pip install pandas numpy tabulate seaborn matplotlib scikit-learn xgboost shap
```

| Library | Purpose |
|---|---|
| `pandas` | Tabular data manipulation |
| `numpy` | Numerical computations |
| `seaborn`, `matplotlib` | Data visualization |
| `scikit-learn` | Machine-learning algorithms and utilities |
| `xgboost` | Gradient boosting model |
| `shap` | Model interpretability |

---

## 1. Importing Libraries

We start by importing all the modules that we will use throughout the notebook.

```python
import pandas as pd
import numpy as np
from tabulate import tabulate
import re

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from matplotlib.colors import LinearSegmentedColormap

from xgboost import XGBRegressor
import shap

# Nicer plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
```

---

## 2. Loading the Dataset `train_hardness.txt`

The file is **tab-separated** (`sep='\t'`). We load it into a `pandas` DataFrame called `amo_set`.

```python
# Load the data
amo_set = pd.read_csv('train_hardness.txt', sep='\t')

# Quick check of what we loaded
amo_set.head()
```

> 💡 **Tip:** Because the file uses tab separators and material names may contain trailing spaces, the columns look clean. If you ever see a column named `'materials '` (with a space), call `amo_set.columns = amo_set.columns.str.strip()`.

---

## 3. Basic Dataset Analysis

In this block we get to know the dataset: its size, its structure, the data types, and its cleanliness.

### **Task 1.** Output the number of rows and columns using `df.shape`.

```python
# Number of rows (objects) and columns (features)
n_rows, n_cols = amo_set.shape
print(f"Rows (materials): {n_rows}")
print(f"Columns (features): {n_cols}")
```

### **Task 2.** Show the first rows of the dataset.

```python
amo_set.head(10)
```

### **Task 3.** What types of data are found?

We distinguish:
- **Numeric** — `Hardness`, `Density`, `Bulk_voigt`, `Bulk_reuss`, `Bulk_vrh`, `Shear_voig`, `Shear_reuss`, `Shear_vrh`, `elasticity`, `poisson`, `Youngs`, `Fr`, `Rx`, `Rn`, `RW`, `Ax`, `An`, `AW`, `IEx`, `IEn`, `IEW`
- **Categorical / text / identifier** — `materials`

There are no inherently binary features in this dataset.

### **Task 4.** Print the names of all columns and rename the target.

The target for this lab is **hardness**, so we rename the column `Hardness` to a shorter, more convenient name `Hardness` (it already has a good name) — but for consistency with the tutorial, we will additionally create a short alias `H` used in some plots.

```python
# List all column names
print(list(amo_set.columns))

# Keep a short alias for the target (do NOT drop the original)
amo_set = amo_set.rename(columns={'Hardness': 'H'})
amo_set.head()
```

> 💡 From this point on, the target variable is called **`H`** (hardness).

### **Task 5.** Identify feature groups.

| Group | Features |
|---|---|
| Identifier (name) | `materials` |
| Target feature | `H` (hardness) |
| Elastic / mechanical properties | `Bulk_voigt`, `Bulk_reuss`, `Bulk_vrh`, `Shear_voig`, `Shear_reuss`, `Shear_vrh`, `elasticity`, `poisson`, `Youngs` |
| Structural / mass properties | `Density` |
| Electronegativity descriptors | `Fr`, `Rx`, `Rn`, `RW` |
| Electron-affinity descriptors | `Ax`, `An`, `AW` |
| Ionization-energy descriptors | `IEx`, `IEn`, `IEW` |

### **Task 6.** Define data types using `df.info()`.

```python
amo_set.info()
```

### **Task 7.** Count the number of `NaN` values in each column.

```python
amo_set.isna().sum()
```

> 📝 If the sum is `0` for every column, the dataset is **complete**: no missing values. Real-world datasets are rarely this clean, so always check.

### **Task 8.** Analyse the zero values.

Some entries — for example, `Fr`, `Rx`, `Rn`, `RW` and `Ax`, `An`, `AW` — could legitimately be exactly `0` for certain materials. For descriptors that are physically positive (mass, moduli, ionization energy), a `0` would indicate an error or missing marker.

```python
# Count zeros per column
(amo_set.drop(columns=['materials']) == 0).sum()
```

> 📝 For this dataset, zeros, when they appear, are **legitimate physical values** — they mostly appear in elemental descriptor columns (e.g., `An` for materials that contain no atoms whose electron affinity is tabulated as non-zero). They should NOT be treated as missing.

### **Task 9.** Check the uniqueness of the categorical column.

```python
print("Number of unique materials:", amo_set['materials'].nunique())
print("Total rows:", len(amo_set))
```

> 💡 If `nunique()` equals the number of rows, then the identifier column is a **primary key** — every row corresponds to a different material. Such a column must **not** be used as a model feature.

### **Task 10.** General conclusion on data quality.

- ✅ No missing values.
- ✅ All descriptors are numerical.
- ✅ `materials` is an identifier (to be dropped before modelling).
- ✅ The dataset is small (≈ 50 materials) — this constrains the choice of ML models and calls for careful cross-validation.
- ✅ Data is **ready for EDA** and modelling.

---

## 4. Exploratory Data Analysis (EDA)

**EDA** is the stage of research during which we:

- examine the structure of the dataset;
- discover hidden relationships between features;
- detect patterns and anomalies;
- test hypotheses (normality, distributions, outliers);
- identify highly correlated features;
- determine which variables influence the target value.

**Why EDA matters:** it helps pick the appropriate statistical methods, build correct ML models, and avoid errors related to noise, outliers or false assumptions.

Before the analysis, we drop the identifier column (it is not informative for statistics or for the model).

```python
# Drop the identifier column
amo_set = amo_set.drop(['materials'], axis=1)
amo_set.head()
```

---

## 4.1 Correlation Analysis (Pearson & Spearman)

Correlation analysis quantifies the **strength** and **direction** of the relationship between two features. In materials science this is crucial: many physical properties (moduli, density, electronic descriptors) are mutually correlated, and they jointly affect hardness.

### **Task 1.** Build a **full** Pearson correlation matrix and visualise it with `seaborn.heatmap`.

```python
plt.figure(figsize=(14, 10))

# Pearson correlation matrix for all numerical features
corr_matrix = amo_set.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # hide the upper triangle

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Correlation coefficient"}
)

plt.title('Features correlation', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

> 🔍 **What to look for:**
> - The three bulk moduli (`Bulk_voigt`, `Bulk_reuss`, `Bulk_vrh`) should be almost perfectly correlated with each other (they are three approximations of the same quantity). The same is true for `Shear_voig`, `Shear_reuss`, `Shear_vrh`.
> - `Youngs` will be strongly correlated with the shear and bulk moduli (they are physically linked: `E ≈ 9KG / (3K + G)`).
> - Hardness `H` typically correlates positively with `Youngs`, `Bulk_vrh` and `Shear_vrh`.

### **Task 2.** Identify groups of strongly correlated features (|r| ≥ 0.25).

```python
corr_matrix = amo_set.corr()
mask = (corr_matrix >= 0.25) | (corr_matrix <= -0.25)
filtered_corr = corr_matrix.where(mask)

# Drop the rows where all values are NaN
significant_corr = filtered_corr.dropna(how='all', axis=0)

# Sort features by their correlation with H
if 'H' in significant_corr.columns:
    target_corr_series = significant_corr['H'].dropna().sort_values(ascending=False)

    significant_corr = significant_corr.reindex(
        index=target_corr_series.index,
        columns=target_corr_series.index
    )

plt.figure(figsize=(12, 8))

sns.heatmap(
    significant_corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    linecolor='lightgray',
    annot_kws={"size": 9, "weight": "bold"},
    cbar_kws={'shrink': 0.8}
)

plt.title('Significant correlations between features (|r| ≥ 0.25)',
          pad=20, fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()
```

### **Task 3.** Find the features most strongly related to hardness.

```python
plt.figure(figsize=(5, 5))

target_corr = amo_set.corr()[['H']].sort_values(by='H', ascending=False)

sns.heatmap(
    target_corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    cbar_kws={'shrink': 0.8}
)

plt.title('Correlation of features with H', pad=20)
plt.tight_layout()
plt.show()
```

**Questions to answer:**

- Which features have the strongest **positive** correlation with `H`?
- Which features have the strongest **negative** correlation with `H`?
- Is there a physical meaning behind these correlations?
  - For instance, `Youngs` and `Bulk_vrh` being strongly positive means stiffer materials tend to be harder.
  - Features such as `An` (electron affinity) or `poisson` (Poisson's ratio) often correlate **negatively** with hardness.
- Are there features with **|r| < 0.1**? They may still be useful in non-linear models (tree-based algorithms will find interactions that Pearson cannot see).
- Which features should be included first in an ML model? → The ones with the highest |correlation|.
- Could they cause **overfitting** due to intercorrelation? → Yes, e.g., `Bulk_voigt`, `Bulk_reuss`, `Bulk_vrh` carry almost the same information. Regularised models (Ridge, Lasso) help here.

---

## 4.2 Statistical Analysis

### **Task 1.** Summary table based on the data.

- Select numerical features.
- Build a descriptive-statistics table using `describe()`.
- Round to three decimals.
- Format with **pandas Styler** (light background, outlined cells, dark header with white text).
- Display the table.

```python
print("\nDESCRIPTIVE STATISTICS amo_set")
print("=" * 60)

# Formation of descriptive statistics (round to 3 decimals)
stats = amo_set.describe().round(3)

# Formatting with pandas Styler
styled_stats = stats.style \
    .set_properties(**{
        'background-color': '#f8f9fa',
        'border': '1px solid #dee2e6'
    }) \
    .set_table_styles([{
        'selector': 'th',
        'props': [
            ('background-color', '#440154'),
            ('color', 'white'),
            ('font-weight', 'bold')
        ]
    }]) \
    .format("{:.3f}")

# Display the styled table
display(styled_stats)
```

### **Task 2.** Compare Pearson vs Spearman correlations with the target.

We build a table where for each feature we show:

1. Pearson's correlation with `H`;
2. Spearman's correlation with `H`;
3. The absolute difference `|Spearman − Pearson|`.

A large difference indicates a **non-linear** relation, presence of **outliers**, or a **monotone-but-not-linear** relationship. Spearman is rank-based and robust to outliers.

```python
# Pearson correlation vector
pearson_corr = amo_set.corr(numeric_only=True)['H']

# Spearman correlation vector
spearman_corr = amo_set.corr(method='spearman', numeric_only=True)['H']

# Build a comparison table
corr_compare = pd.DataFrame({
    'feature': pearson_corr.index,
    'Pearson': pearson_corr.values,
    'Spearman': spearman_corr.values
})

# Absolute difference
corr_compare['Difference = |Pearson - Spearman|'] = (
    corr_compare['Pearson'] - corr_compare['Spearman']
).abs()

# Drop the row for H itself
corr_compare = corr_compare[corr_compare['feature'] != 'H']

# Sort by the difference, descending
corr_compare = corr_compare.sort_values(
    by='Difference = |Pearson - Spearman|',
    ascending=False
)

corr_compare
```

**Interpretation:**

- Features at the top of the table have the largest mismatch between Pearson and Spearman.
- If `|Pearson − Spearman|` is large while `|Spearman|` itself is high → the relation is **monotone but non-linear**.
- If `|Pearson|` is much smaller than `|Spearman|` → likely **outliers**.
- If both are close to zero → probably no linear nor monotone relationship.

---

## 4.3 Scatter Plots of Top Features vs Hardness

Visual inspection is essential for detecting the **shape** of the relationship (linear, non-linear, clusters, outliers).

```python
# Correlation of each feature with the target
target_corr_series = amo_set.corr()['H'].drop('H').sort_values(key=abs, ascending=False)

# Top-4 features by |correlation| with H
top_features = target_corr_series.head(4).index.tolist()
print("Top features:", top_features)

# 2×2 grid of scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    axes[i].scatter(amo_set[feature], amo_set['H'], alpha=0.6, edgecolor='k')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('H (Hardness)')
    axes[i].set_title(f'{feature} vs H (corr: {target_corr_series[feature]:.2f})')

plt.tight_layout()
plt.show()
```

**Analyse the graphs:**

- Does the relationship appear linear or non-linear?
- Does the plot match the numerical value of the correlation?

---

## 5. Machine-Learning Models for Predicting Hardness

We now build predictive models. Our goal: given physical/electronic descriptors, predict the hardness of a new material.

### 5.1 Preparing the Data

Steps:

1. Rename the target to a shorter name (`Hardness` is fine, but we already use `H`).
2. Separate features (`X`) from the target (`y`).
3. There are no categorical features left (the identifier has been dropped), so no encoding is needed.
4. Split into train / test (80/20).
5. Standardise features for distance- and gradient-based models (Ridge, KNN).

```python
# Features and target
X = amo_set.drop(columns=['H'])
y = amo_set['H']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)
```

### 5.2 Training Several Models

We compare five algorithms:

| Model | Rationale |
|---|---|
| **Ridge** | Simple linear baseline with regularisation |
| **Decision Tree** | Non-linear, interpretable |
| **KNN** | Non-parametric, distance-based |
| **Random Forest** | Robust ensemble of trees |
| **XGBoost** | State-of-the-art gradient boosting |

```python
models = {
    'Ridge'         : Ridge(alpha=1.0, random_state=42),
    'DecisionTree'  : DecisionTreeRegressor(random_state=42),
    'KNN'           : KNeighborsRegressor(n_neighbors=5),
    'RandomForest'  : RandomForestRegressor(n_estimators=300, random_state=42),
    'XGBoost'       : XGBRegressor(
                          n_estimators=400,
                          learning_rate=0.05,
                          max_depth=4,
                          random_state=42,
                          n_jobs=-1
                      )
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results.append({
        'Model': name,
        'MAE'  : mean_absolute_error(y_test, y_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²'   : r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
results_df
```

### 5.3 Cross-Validation

With only ~50 materials, a single train/test split is unreliable. We use **K-Fold cross-validation** to obtain a more robust estimate.

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=cv, scoring='r2')
    cv_results.append({
        'Model'     : name,
        'CV R² mean': scores.mean(),
        'CV R² std' : scores.std()
    })

cv_results_df = pd.DataFrame(cv_results).sort_values('CV R² mean', ascending=False)
cv_results_df
```

> 📝 A large `CV R² std` means the model is unstable; expect this on small datasets.

### 5.4 Hyperparameter Tuning (Example with XGBoost)

We illustrate how to tune the most promising model.

```python
param_grid = {
    'n_estimators' : [200, 400, 800],
    'max_depth'    : [3, 4, 6],
    'learning_rate': [0.03, 0.05, 0.1]
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    xgb, param_grid,
    cv=5, scoring='r2', n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

print("Best params:", grid.best_params_)
print("Best CV R² :", round(grid.best_score_, 3))
```

### 5.5 Feature Importance

We analyse which features contribute most to the predictions of the best model (here, the tuned XGBoost).

```python
best_model = grid.best_estimator_

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Feature importances (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

importances
```

> 🔍 Expect to see `Youngs`, `Bulk_vrh`, `Shear_vrh`, `Density`, and possibly some of the `*n`/`*W` electronic descriptors at the top.

### 5.6 Model Interpretation with SHAP

SHAP gives per-prediction attributions: not only *which* features matter, but also *how* (positively or negatively) they push the prediction.

```python
explainer   = shap.Explainer(best_model, X_train_scaled, feature_names=X.columns)
shap_values = explainer(X_test_scaled)

# Global summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
```

```python
# Bar plot of mean |SHAP|
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")
```

**Interpretation:**

- Features at the top have the largest impact on the predicted hardness.
- The colour of each point indicates the feature value (red = high, blue = low).
- If red points cluster on the right (positive SHAP), a high feature value **increases** hardness. If red points cluster on the left, a high value **decreases** hardness.

---

## 6. Conclusions

- The dataset is small but clean: 54 materials, no missing values, all features numeric except the identifier.
- Correlation analysis shows that the moduli (`Youngs`, `Bulk_vrh`, `Shear_vrh`) are the strongest predictors of hardness.
- The three bulk/shear-modulus variants are almost interchangeable — a single representative per group is enough for a minimal model.
- Spearman vs Pearson comparison highlights monotone-but-non-linear relations (e.g. with `elasticity` or electronic descriptors).
- Tree-based models (Random Forest, XGBoost) perform notably better than Ridge on this dataset, confirming non-linearity.
- SHAP summarises feature contributions clearly, providing physically interpretable insights.

---

## 📝 Exercises

1. **Feature selection.** Train XGBoost using only the top-5 features from the SHAP analysis. Does the test R² degrade? Why?
2. **Model reduction.** Remove `Bulk_voigt`, `Bulk_reuss`, `Shear_voig`, `Shear_reuss` and re-run. What happens?
3. **Log target.** Hardness is strictly positive; try `log(H)` as target. Do the metrics improve?
4. **Outliers.** Use `IsolationForest` to detect outliers and retrain.
5. **Feature engineering.** Add `Bulk_vrh / Shear_vrh` (Pugh's ratio `K/G`) as a new feature and see if it improves the model.

---

## 📎 References

- Pugh, S. F. (1954). *Relations between the elastic moduli and the plastic properties of polycrystalline pure metals.*
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* (SHAP)
- scikit-learn documentation: <https://scikit-learn.org>
