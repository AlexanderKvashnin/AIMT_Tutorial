# 🧪 Python for Data Science in Materials Science

> **A complete beginner-friendly tutorial**
> From basic statistics to neural networks on Materials Project data

---

## 📋 Table of Contents

- [Introduction and Setup](#-introduction-and-setup)
- [1. Statistical Methods in Data Processing](#-1-statistical-methods-in-data-processing)
- [2. Regression Methods: Linear, Random Forest, XGBoost, Neural Networks](#-2-regression-methods-linear-random-forest-xgboost-neural-networks)
- [3. Working with Data: Cleaning, Outliers, Correlations](#-3-working-with-data-cleaning-outliers-correlations)
- [4. Loading Data from Materials Project](#-4-loading-data-from-materials-project)
- [5. Training ML Models: Complete Pipeline](#-5-training-ml-models-complete-pipeline)
- [6. Regression Metrics](#-6-regression-metrics)
- [7. Simple Neural Networks](#-7-simple-neural-networks)
- [8. Full Workflow on Materials Project Data](#-8-full-workflow-on-materials-project-data)
- [9. Practical Recommendations](#-9-practical-recommendations)

---

## 🔧 Introduction and Setup

### What is a Python Notebook?

A **Jupyter Notebook** is an interactive environment where code is organized into "cells". You run a cell with `Shift + Enter`, and the output (a number, a table, a plot) appears right below the code. This makes it perfect for learning: you can change one line and immediately see how the result changes.

### Installing Libraries

Open a terminal (or command prompt) and run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost mp-api matminer pymatgen scipy
```

| Library | What it's used for |
|---|---|
| `numpy` | Working with numerical arrays and math |
| `pandas` | Data tables (like Excel, but in code) |
| `matplotlib` | Basic plotting |
| `seaborn` | Beautiful statistical plots |
| `scikit-learn` | Main ML library (models, metrics, data prep) |
| `xgboost` | Gradient boosting algorithm |
| `mp-api` | Official API for Materials Project |
| `matminer` | Feature generation for materials |
| `pymatgen` | Working with crystal structures |
| `scipy` | Scientific computing, statistics |

> **💡 Tip**
> If you use **Anaconda**, many libraries are already installed. Just add `!pip install xgboost mp-api matminer` directly inside a notebook cell.

---

## 📊 1. Statistical Methods in Data Processing

Statistics is the foundation of data analysis. Before training any ML model, you need to understand:

- How values are **distributed** (histogram)
- Where the **center** of the data is (mean, median)
- How **spread out** the data is (standard deviation)
- Whether there are **outliers** or asymmetry

### 1.1 Importing Libraries

```python
# Import numpy — a library for working with arrays and math.
# np is the conventional alias so we don't have to type "numpy" every time.
import numpy as np

# Import pandas — a library for working with tables (DataFrames).
# pd is the standard alias.
import pandas as pd

# Import pyplot from matplotlib for creating plots.
# plt is the standard alias for matplotlib.pyplot.
import matplotlib.pyplot as plt

# Import seaborn — a wrapper over matplotlib for beautiful statistical plots.
# sns is the standard alias.
import seaborn as sns
```

**What's happening here:**

- `import X` is a Python command that loads an external library
- `as Y` gives the library a short alias so we write less code
- All libraries must be installed via `pip install` (see above)

> **📝 Note**
> The aliases `np`, `pd`, `plt`, `sns` are a **community convention**. The entire world uses these shortcuts. If you name them differently, the code will still work, but other programmers won't understand it.

### 1.2 Creating a Synthetic Dataset

```python
# np.random.seed(42) fixes the "seed" of the random number generator.
# This means: every time you run the code, it will generate the SAME "random" numbers.
# This is critical for reproducibility in scientific experiments!
np.random.seed(42)

# Create three groups of values simulating the bulk modulus of different materials.
# np.random.normal(loc, scale, size) generates normally distributed numbers:
#   loc (mean) = average value
#   scale (std) = standard deviation
#   size = how many numbers to generate
# np.concatenate merges the three arrays into one.
bulk_modulus = np.concatenate([
    np.random.normal(loc=100, scale=15, size=200),   # Ceramics: mean 100 GPa
    np.random.normal(loc=50, scale=10, size=300),    # Metals: mean 50 GPa
    np.random.normal(loc=180, scale=25, size=100)    # Hard materials: mean 180 GPa
])

# Create a DataFrame (table) from a single column.
# pandas.DataFrame is a data structure similar to an Excel table.
# The curly braces {} create a dictionary: key "bulk_modulus_GPa" -> array of values.
df_stats = pd.DataFrame({'bulk_modulus_GPa': bulk_modulus})
```

**Line by line:**

1. `np.random.seed(42)` — fixes the generator. Without it, data would be different on every run.
2. `np.random.normal(100, 15, 200)` — generates 200 numbers distributed like a "bell" around 100 with spread 15.
3. `np.concatenate([...])` — glues three arrays into one of length 200 + 300 + 100 = 600.
4. `pd.DataFrame({...})` — wraps the array in a table with a column name.

### 1.3 Descriptive Statistics

```python
# .mean() computes the arithmetic mean of all values in a column
print("Mean:", df_stats['bulk_modulus_GPa'].mean())

# .median() — the median: the value that splits sorted data in half
print("Median:", df_stats['bulk_modulus_GPa'].median())

# .std() — standard deviation: a measure of spread around the mean
print("Std deviation:", df_stats['bulk_modulus_GPa'].std())

# .skew() — skewness:
#   0 -> symmetric distribution
#   > 0 -> "tail" on the right (a few very large values)
#   < 0 -> "tail" on the left
print("Skewness:", df_stats['bulk_modulus_GPa'].skew())

# .kurtosis() — "peakedness":
#   > 0 -> sharper peak than a normal distribution
#   < 0 -> flatter than normal
print("Kurtosis:", df_stats['bulk_modulus_GPa'].kurtosis())
```

**How to read the output:**

| Metric | What it means for materials | Example interpretation |
|---|---|---|
| **Mean** | Average elastic modulus across the sample | "Average bulk modulus = 93 GPa" |
| **Median** | Typical modulus (robust to outliers) | If mean > median -> data skewed right |
| **Std** | Spread of properties | "+/-22 GPa around the mean" |
| **Skewness** | Asymmetry | > 0 -> some materials have extremely high stiffness |
| **Kurtosis** | Frequency of extreme values | High kurtosis -> many "outliers" |

### 1.4 Visualizing the Distribution

```python
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Creates a figure (fig) with a grid of plots: 1 row x 2 columns.
# axes is an array of two Axes objects, one per subplot.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# --- Plot 1: Histogram ---
# axes[0] — first (left) axis
# .hist() draws a histogram: it splits data into bins and counts how many values fall in each.
axes[0].hist(df_stats['bulk_modulus_GPa'], bins=40, edgecolor='black', alpha=0.7)

# .set_xlabel() and .set_ylabel() — axis labels
axes[0].set_xlabel('Bulk Modulus (GPa)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of Bulk Modulus')

# --- Plot 2: Boxplot ---
# sns.boxplot() draws a "box-and-whiskers" plot
sns.boxplot(y=df_stats['bulk_modulus_GPa'], ax=axes[1])
axes[1].set_title('Boxplot of Bulk Modulus')

# plt.tight_layout() automatically adjusts spacing so plots don't overlap
plt.tight_layout()

# plt.show() displays the figure (in Jupyter — right in the cell output)
plt.show()
```

### 1.5 Correlation Analysis

```python
# Fix the seed again for reproducibility
np.random.seed(1)

# n — number of samples (materials) in our synthetic dataset
n = 500

# Atomic radius: uniform distribution from 1.0 to 2.5 Angstrom
atomic_radius = np.random.uniform(1.0, 2.5, n)

# Electronegativity: smaller radius means higher electronegativity (inverse relationship)
electronegativity = 4.0 - atomic_radius + np.random.normal(0, 0.3, n)

# Melting temperature: linearly related to electronegativity + noise
melting_temp = 1000 + 800 * electronegativity + np.random.normal(0, 200, n)

# Bulk modulus: linearly related to melting temperature + noise
bulk_modulus = 20 + 40 * melting_temp / 1000 + np.random.normal(0, 15, n)

# Shear modulus: correlated with bulk modulus (roughly 0.6 * bulk) + noise
shear_modulus = 0.6 * bulk_modulus + np.random.normal(0, 8, n)

# Collect all arrays into one DataFrame
df_corr = pd.DataFrame({
    'atomic_radius': atomic_radius,
    'electronegativity': electronegativity,
    'melting_T': melting_temp,
    'bulk_modulus': bulk_modulus,
    'shear_modulus': shear_modulus
})

# .corr(method='pearson') computes the Pearson correlation matrix.
# Value at cell [i, j] = correlation coefficient between column i and column j:
#   +1 -> perfect positive correlation
#    0 -> no linear relationship
#   -1 -> perfect negative correlation
corr_matrix = df_corr.corr(method='pearson')

# Create a figure 8x6 inches
plt.figure(figsize=(8, 6))

# sns.heatmap() draws a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, linewidths=0.5)

plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.show()
```

> **⚠️ Important**
> **Correlation is not causation.** High correlation only means two quantities change together; it does not explain why.

**How to read the correlation matrix:**

| Value | Interpretation |
|---|---|
| 0.9 – 1.0 | Very strong relationship |
| 0.7 – 0.9 | Strong relationship |
| 0.5 – 0.7 | Moderate relationship |
| 0.3 – 0.5 | Weak relationship |
| 0 – 0.3 | Essentially no linear relationship |

---

## 🤖 2. Regression Methods: Linear, Random Forest, XGBoost, Neural Networks

**Regression** is the task of predicting a **numeric** value (in contrast to **classification**, where you predict a class). In materials science, regression is used to predict elastic moduli, melting temperatures, band gaps, and more.

### 2.1 Linear Regression

**Idea:** find the line (or hyperplane) that minimizes the sum of squared differences between predictions and actual values.

Mathematically: `y_hat = w1*x1 + w2*x2 + ... + wn*xn + b`, where `wi` are coefficients and `b` is the intercept.

```python
# Import the LinearRegression class from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Import train_test_split — it splits data into training and test sets
from sklearn.model_selection import train_test_split

# Import metrics:
#   mean_absolute_error (MAE) — mean absolute error
#   mean_squared_error (MSE) — mean squared error
#   r2_score (R2) — coefficient of determination
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# X — feature matrix. We select 3 columns.
X = df_corr[['atomic_radius', 'electronegativity', 'melting_T']]

# y — target vector. This is what we want to predict.
y = df_corr['bulk_modulus']

# train_test_split splits the data into two parts:
#   test_size=0.2 -> 20% goes to the test set, 80% to training
#   random_state=42 -> fixes the "randomness" of the split for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model object (not yet trained)
lin_reg = LinearRegression()

# .fit(X_train, y_train) — train the model.
lin_reg.fit(X_train, y_train)

# .predict(X_test) — make predictions on test data.
y_pred_lr = lin_reg.predict(X_test)

print("Linear Regression Results:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_lr):.2f} GPa")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f} GPa")
print(f"  R2:   {r2_score(y_test, y_pred_lr):.4f}")

# lin_reg.coef_ — array of coefficients (one per feature)
# lin_reg.intercept_ — the bias term (b)
print(f"  Coefficients: {dict(zip(X.columns, lin_reg.coef_.round(3)))}")
print(f"  Intercept: {lin_reg.intercept_:.3f}")
```

**Line-by-line explanation of key lines:**

| Line | What it does |
|---|---|
| `X = df_corr[['a','b','c']]` | Creates the feature matrix (2D: rows x columns) |
| `y = df_corr['bulk_modulus']` | Creates the target vector (1D: only values) |
| `train_test_split(X, y, test_size=0.2)` | Splits X and y **consistently** |
| `lin_reg.fit(X_train, y_train)` | **Training**: the model finds coefficients on training data |
| `y_pred_lr = lin_reg.predict(X_test)` | **Prediction**: applies the learned formula to new data |
| `lin_reg.coef_` | Coefficients showing the contribution of each feature |

### 2.2 Visualizing Predictions

```python
# plt.figure(figsize=(6, 6)) creates a new figure 6x6 inches
plt.figure(figsize=(6, 6))

# .scatter(x, y) draws points
plt.scatter(y_test, y_pred_lr, alpha=0.5, edgecolors='k', linewidth=0.5)

# Draw the ideal y = x line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', label='Perfect prediction')

plt.xlabel('Actual Bulk Modulus (GPa)')
plt.ylabel('Predicted Bulk Modulus (GPa)')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.tight_layout()
plt.show()
```

> **💡 Tip**
> **How to read a parity plot:** the closer the points are to the red dashed line, the better the model.

### 2.3 Random Forest Regressor

**Idea:** train many decision trees, each on a random subset of data, and average their predictions.

```python
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Create the model
rf_reg = RandomForestRegressor(
    n_estimators=200,        # number of trees in the forest
    max_depth=10,            # maximum depth of each tree
    min_samples_leaf=5,      # each leaf must contain at least 5 samples
    random_state=42
)

# Train on training data
rf_reg.fit(X_train, y_train)

# Make predictions on test data
y_pred_rf = rf_reg.predict(X_test)

print("Random Forest Results:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_rf):.2f} GPa")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f} GPa")
print(f"  R2:   {r2_score(y_test, y_pred_rf):.4f}")

# rf_reg.feature_importances_ — array of "importance" for each feature (sum = 1.0)
importances = pd.Series(rf_reg.feature_importances_, index=X.columns)

# .sort_values() — sort ascending
# .plot(kind='barh') — horizontal bar chart
importances.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

**Random Forest parameters:**

| Parameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | Number of trees | More -> more stable, but slower. Usually 100–500 |
| `max_depth` | Maximum tree depth | Larger -> more complex model, risk of overfitting |
| `min_samples_leaf` | Minimum objects per leaf | Larger -> simpler model, better generalization |
| `random_state` | Random seed | Fixes reproducibility |

### 2.4 XGBoost Regressor

**Idea:** build trees **sequentially**, where each new tree corrects the errors of the previous one. This is called **gradient boosting**.

```python
# Import XGBRegressor from the xgboost library
from xgboost import XGBRegressor

# Create the XGBoost model
xgb_reg = XGBRegressor(
    n_estimators=300,        # number of sequential trees
    learning_rate=0.05,      # learning rate (small = slower, but more accurate)
    max_depth=4,             # shallow trees
    subsample=0.8,           # each tree trains on 80% of random samples
    colsample_bytree=0.8,    # each tree uses 80% of random features
    random_state=42
)

# Train the model
xgb_reg.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_reg.predict(X_test)

print("XGBoost Results:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_xgb):.2f} GPa")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f} GPa")
print(f"  R2:   {r2_score(y_test, y_pred_xgb):.4f}")

# Plot feature importances
xgb_importances = pd.Series(xgb_reg.feature_importances_, index=X.columns)
xgb_importances.sort_values().plot(kind='barh', color='darkorange')
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
```

**Key XGBoost parameters:**

| Parameter | Meaning | Typical values |
|---|---|---|
| `n_estimators` | How many trees to build | 100–1000 |
| `learning_rate` | Contribution of each tree | 0.01–0.3 |
| `max_depth` | Tree depth | 3–8 |
| `subsample` | Fraction of samples per tree | 0.6–1.0 |
| `colsample_bytree` | Fraction of features per tree | 0.6–1.0 |

> **📝 Note**
> **Random Forest vs XGBoost:**
> - RF builds trees **in parallel**, independent of each other.
> - XGBoost builds trees **sequentially**, each correcting the errors of the previous one.
> - On tabular data XGBoost is usually more accurate, but needs more careful tuning.

### 2.5 Neural Network (MLPRegressor)

**Idea:** a Multi-Layer Perceptron (MLP) is a sequence of neuron layers.

```python
# Import MLPRegressor (Multi-Layer Perceptron Regressor)
from sklearn.neural_network import MLPRegressor

# StandardScaler brings all features to the same scale (mean=0, std=1).
# Neural networks are VERY sensitive to feature scale!
from sklearn.preprocessing import StandardScaler

# Create scaler and fit it on training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# IMPORTANT: for test data use ONLY transform (no fit)!
X_test_scaled = scaler.transform(X_test)

# Create the neural network
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # two hidden layers: 64 neurons and 32 neurons
    activation='relu',              # ReLU activation function
    solver='adam',                  # optimization algorithm
    max_iter=2000,                  # maximum of 2000 training iterations
    early_stopping=True,            # stop training if validation error grows
    random_state=42
)

# Train the model on scaled data
mlp_reg.fit(X_train_scaled, y_train)

# Predict (also on scaled data!)
y_pred_mlp = mlp_reg.predict(X_test_scaled)

print("Neural Network (MLP) Results:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_mlp):.2f} GPa")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_mlp)):.2f} GPa")
print(f"  R2:   {r2_score(y_test, y_pred_mlp):.4f}")
```

**How a neuron works:**

```
Input: x1, x2, x3
      |   |   |
   [w1] [w2] [w3]   <- weights (trainable parameters)
      |   |   |
   z = w1*x1 + w2*x2 + w3*x3 + b   <- linear combination
      |
   a = ReLU(z) = max(0, z)         <- nonlinearity
      |
   output
```

> **⚠️ Warning**
> **Always scale your data for neural networks!** Without `StandardScaler`, the network trains very slowly or doesn't converge at all, because different features have different scales.

---

## 🧹 3. Working with Data: Cleaning, Outliers, Correlations

### 3.1 Outlier Detection with IQR

**IQR (Interquartile Range)** = Q3 - Q1, where Q1 is the 25th percentile and Q3 is the 75th percentile.

Rule: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR are considered outliers.

```python
# Fix the seed
np.random.seed(7)

# Create "clean" data: 480 normal values
n_clean = 480
bulk_modulus_clean = np.random.normal(120, 20, n_clean)

# Add 20 obvious outliers (extremely hard and soft materials)
outliers = np.array([400, 380, -50, 450, 420, 390, 410, 430, 370, 440,
                     460, 350, 470, 360, 480, 340, 490, 330, 500, 320])

# Combine clean data and outliers
bulk_modulus_all = np.concatenate([bulk_modulus_clean, outliers])

# Create a DataFrame
df_out = pd.DataFrame({'bulk_modulus': bulk_modulus_all})

# --- IQR method ---
Q1 = df_out['bulk_modulus'].quantile(0.25)   # 25th percentile
Q3 = df_out['bulk_modulus'].quantile(0.75)   # 75th percentile
IQR = Q3 - Q1                                # interquartile range

lower_bound = Q1 - 1.5 * IQR   # lower outlier bound
upper_bound = Q3 + 1.5 * IQR   # upper outlier bound

# Filter outliers using logical OR (|)
outliers_iqr = df_out[(df_out['bulk_modulus'] < lower_bound) |
                       (df_out['bulk_modulus'] > upper_bound)]

print(f"IQR bounds: [{lower_bound:.1f}, {upper_bound:.1f}] GPa")
print(f"Outliers detected by IQR: {len(outliers_iqr)}")
```

### 3.2 Outlier Detection with Z-score

**Z-score** shows how many standard deviations a value is from the mean: `z = (x - mean) / std`.

Rule: if `|z| > 3`, the value is considered an outlier.

```python
# Import scipy.stats for statistical functions
from scipy import stats

# Compute z-scores and take the absolute value
z_scores = np.abs(stats.zscore(df_out['bulk_modulus']))

# Filter: keep rows where |z| > 3
outliers_z = df_out[z_scores > 3]
print(f"Outliers detected by Z-score (|z| > 3): {len(outliers_z)}")
```

### 3.3 Visualizing Outliers

```python
# Create a figure 10x4 inches
plt.figure(figsize=(10, 4))

# First subplot: data with outliers
plt.subplot(1, 2, 1)
plt.hist(df_out['bulk_modulus'], bins=50, edgecolor='black', alpha=0.7)

# Draw vertical dashed lines at the IQR bounds
plt.axvline(lower_bound, color='red', linestyle='--', label='IQR bounds')
plt.axvline(upper_bound, color='red', linestyle='--')
plt.xlabel('Bulk Modulus (GPa)')
plt.title('With Outliers')
plt.legend()

# Filter data: keep only values within the bounds
df_clean = df_out[(df_out['bulk_modulus'] >= lower_bound) &
                  (df_out['bulk_modulus'] <= upper_bound)]

# Second subplot: data after removing outliers
plt.subplot(1, 2, 2)
plt.hist(df_clean['bulk_modulus'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Bulk Modulus (GPa)')
plt.title('After Removing Outliers (IQR)')
plt.tight_layout()
plt.show()

print(f"Original size: {len(df_out)}, Cleaned size: {len(df_clean)}")
```

### 3.4 Handling Missing Values

```python
# Create a dataset with missing values (np.nan = Not a Number)
df_missing = pd.DataFrame({
    'bulk_modulus': [120, np.nan, 95, 110, np.nan, 130, 105],
    'shear_modulus': [80, 75, np.nan, 70, 85, np.nan, 72],
    'melting_T': [1500, 1200, 1800, np.nan, 1400, 1600, 1300]
})

# .isnull() returns True/False for each cell (NaN -> True).
# .sum() counts True values in each column.
print("Missing values per column:")
print(df_missing.isnull().sum())

# .isnull().mean() — fraction of missing values
print(f"\nPercentage missing: {df_missing.isnull().mean().round(2) * 100}")

# --- Option 1: drop rows with any missing value ---
df_dropped = df_missing.dropna()
print(f"\nAfter dropna(): {len(df_dropped)} rows (from {len(df_missing)})")

# --- Option 2: fill missing values with the mean ---
df_filled = df_missing.fillna(df_missing.mean())
print("\nAfter filling with column means:")
print(df_filled.round(1))
```

> **⚠️ Warning**
> In materials science, filling missing values with the mean is **dangerous**: properties of different material classes differ greatly (ceramics vs. metals). Better to drop missing values or use smarter methods.

### 3.5 Feature Selection

```python
# .corr() — correlation matrix, .abs() — absolute value
corr_matrix = df_corr.corr().abs()

# np.triu(...) — upper triangle of the matrix
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# List of features with correlation > 0.9 (strongly redundant)
high_corr_features = [col for col in upper_triangle.columns
                       if any(upper_triangle[col] > 0.9)]
print(f"Features to consider dropping: {high_corr_features}")
```

### 3.6 Recursive Feature Elimination (RFECV)

**RFECV** — Recursive Feature Elimination with Cross-Validation.

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

# Create a base model to estimate feature importance
rf_selector = RandomForestRegressor(n_estimators=50, random_state=42)

# RFECV:
#   step=1 -> remove one feature at a time
#   cv=5 -> 5-fold cross-validation
#   scoring='neg_mean_absolute_error' -> maximize negative MAE (= minimize MAE)
rfecv = RFECV(rf_selector, step=1, cv=5, scoring='neg_mean_absolute_error')

# Train on training data
rfecv.fit(X_train, y_train)

# .n_features_ — optimal number of features
print(f"Optimal number of features: {rfecv.n_features_}")

# .support_ — boolean array: True if the feature is kept
print(f"Selected features: {list(X.columns[rfecv.support_])}")
```

---

## 🔬 4. Loading Data from Materials Project

**Materials Project** (https://materialsproject.org) is the largest open database of materials properties computed with DFT.

### 4.1 Getting an API Key

1. Register at [materialsproject.org](https://materialsproject.org)
2. Go to the **API** section: https://materialsproject.org/api
3. Copy your key
4. Store it as an environment variable:

```bash
# Linux/macOS
export MP_API_KEY="your_key_here"

# Windows (cmd)
set MP_API_KEY=your_key_here
```

Or directly in Python:

```python
import os
os.environ["MP_API_KEY"] = "your_key_here"
```

### 4.2 Querying Elastic Properties

```python
# Import MPRester — the main class for working with the API
from mp_api.client import MPRester

import pandas as pd

# with MPRester() as mpr: opens a connection to the API.
# It automatically reads the key from the MP_API_KEY environment variable.
with MPRester() as mpr:

    # .materials.elasticity.search(...) — searches for materials with elastic properties.
    # k_vrh=(50, 300) — bulk modulus (Voigt-Reuss-Hill average) in 50-300 GPa range
    # g_vrh=(20, 200) — shear modulus in 20-200 GPa range
    docs = mpr.materials.elasticity.search(
        k_vrh=(50, 300),
        g_vrh=(20, 200),
        num_chunks=1,
        chunk_size=200
    )

print(f"Retrieved {len(docs)} elasticity records")

# Create an empty list to accumulate data
records = []

# Loop over each document in the results
for doc in docs:
    records.append({
        'material_id': doc.material_id,           # Material ID, e.g., "mp-149"
        'formula': doc.formula_pretty,             # Formula, e.g., "SiO2"
        'bulk_modulus_vrh': doc.k_vrh,             # Bulk modulus (VRH average)
        'shear_modulus_vrh': doc.g_vrh,            # Shear modulus (VRH)
        'poisson_ratio': doc.poisson_ratio,        # Poisson's ratio
    })

# Convert the list of dictionaries into a DataFrame
df_mp = pd.DataFrame(records)

# .head() prints the first 5 rows
print(df_mp.head())

# .shape — table dimensions (rows, columns)
print(f"\nDataset shape: {df_mp.shape}")

# .describe() — descriptive statistics for all numeric columns
print(df_mp.describe())
```

### 4.3 Available Elastic Fields in Materials Project

| Field | Description |
|---|---|
| `k_vrh` | Bulk modulus (Voigt-Reuss-Hill average) |
| `k_voigt` | Bulk modulus by Voigt (upper bound) |
| `k_reuss` | Bulk modulus by Reuss (lower bound) |
| `g_vrh` | Shear modulus (VRH average) |
| `g_voigt` | Shear modulus by Voigt |
| `g_reuss` | Shear modulus by Reuss |
| `poisson_ratio` | Poisson's ratio |
| `universal_anisotropy` | Universal anisotropy |
| `homogeneous_poisson` | Homogeneous Poisson's ratio |
| `elastic_tensor` | Full elastic tensor (6x6) |

### 4.4 Alternative: Matbench

**Matbench** is a standardized set of benchmarks for ML in materials science.

```bash
pip install matbench
```

```python
from matbench.bench import MatbenchBenchmark

# autoload=False — do not load all datasets immediately (they are large)
mb = MatbenchBenchmark(autoload=False)

# Loop over all tasks
for task in mb.tasks:
    task.load()   # loads the task data
    print(f"Task: {task.metadata.name}")
    print(f"  Target: {task.metadata.target}")
    print(f"  Input type: {task.metadata.input_type}")
    print(f"  Number of samples: {task.metadata.n_samples}")
    print()
```

---

## 🚀 5. Training ML Models: Complete Pipeline

### 5.1 Splitting into train / validation / test

**Why three sets?**

| Set | Purpose | Analogy |
|---|---|---|
| **Train** | The model learns from this data | Textbook |
| **Validation** | Hyperparameter tuning, early stopping | Practice exam |
| **Test** | Final performance evaluation | Real exam |

```python
from sklearn.model_selection import train_test_split

# Convert to numpy arrays (.values)
X = df_corr[['atomic_radius', 'electronegativity', 'melting_T']].values
y = df_corr['bulk_modulus'].values

# Step 1: separate test (20% of all data)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: split the remaining 80% to get validation (25% of 80% = 20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# Result: 60% train, 20% val, 20% test
print(f"Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
```

### 5.2 Training Multiple Models and Comparing Them

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline is a "conveyor": scaler -> model.
# Benefit: when calling .fit(X_train), the scaler is automatically fitted and applied,
# and .predict(X_test) applies the same scaler to the test set.
# This protects against data leakage!
models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),                # step 1: scaling
        ('model', LinearRegression())                # step 2: model
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=4, random_state=42))
    ]),
    'Neural Network (MLP)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                early_stopping=True, random_state=42))
    ])
}

# Dictionary for results
results = {}

# .items() returns (key, value) pairs from the dictionary
for name, pipeline in models.items():
    # Train the pipeline on training data
    pipeline.fit(X_train, y_train)

    # Predict on validation
    y_val_pred = pipeline.predict(X_val)

    # Save the metrics
    results[name] = {
        'MAE': mean_absolute_error(y_val, y_val_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'R2': r2_score(y_val, y_val_pred)
    }

# Convert the dictionary to a DataFrame and transpose (.T) for readability
results_df = pd.DataFrame(results).T
print("Validation Set Performance:")
print(results_df.round(4))
```

### 5.3 Final Evaluation on the Test Set

```python
# Find the model with the highest R2 on validation
best_model_name = results_df['R2'].idxmax()

# Get the best model from the dictionary
best_model = models[best_model_name]

# IMPORTANT: before the final test, retrain on train + val
# np.vstack() stacks arrays vertically (row-wise).
# np.ravel() flattens a 2D array into 1D.
best_model.fit(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]).ravel())

# Test prediction
y_test_pred = best_model.predict(X_test)

print(f"\nBest model: {best_model_name}")
print("Test Set Performance:")
print(f"  MAE:  {mean_absolute_error(y_test, y_test_pred):.2f} GPa")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f} GPa")
print(f"  R2:   {r2_score(y_test, y_test_pred):.4f}")

# --- Parity plots for all models ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# .ravel() turns the 2x2 matrix into a 1D array of 4 elements.
# zip(...) iterates over pairs (axis, (name, pipeline)).
for ax, (name, pipeline) in zip(axes.ravel(), models.items()):
    y_pred = pipeline.predict(X_test)

    # Points: actual vs predicted
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.3)

    # Ideal line
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', linewidth=1.5)

    ax.set_xlabel('Actual (GPa)')
    ax.set_ylabel('Predicted (GPa)')
    ax.set_title(f'{name}\nR2 = {r2_score(y_test, y_pred):.3f}')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

---

## 📐 6. Regression Metrics

### Definitions

| Metric | Formula | Meaning | Units |
|---|---|---|---|
| **MAE** | `(1/n) * sum(abs(y_i - y_hat_i))` | Mean absolute error | Same as target (GPa) |
| **MSE** | `(1/n) * sum((y_i - y_hat_i)^2)` | Mean squared error | GPa^2 |
| **RMSE** | `sqrt(MSE)` | Root of MSE | GPa |
| **R2** | `1 - SS_res / SS_tot` | Fraction of explained variance | Unitless |

### How to Interpret

- **MAE = 5 GPa** -> on average the model is off by 5 GPa
- **RMSE > MAE** -> some large errors exist (RMSE is sensitive to outliers)
- **R2 = 0.9** -> the model explains 90% of the data variance
- **R2 = 0** -> the model is no better than just predicting the mean
- **R2 < 0** -> the model is worse than predicting the mean

> **💡 Tip**
> **When to use which metric:**
> - **MAE** — when the "typical" error matters and outliers are not critical
> - **RMSE** — when large errors are especially undesirable
> - **R2** — for comparing models on the same dataset

---

## 🧠 7. Simple Neural Networks

### 7.1 Anatomy of a Perceptron

A **perceptron** is the basic "building block" of a neural network. A single neuron:

1. Receives inputs `x1, x2, ..., xn`
2. Computes a weighted sum: `z = sum(wi * xi) + b`
3. Applies an activation: `a = sigma(z)`
4. Passes the result forward

**MLP (Multi-Layer Perceptron)** is many such neurons organized in layers:

- **Input layer** — receives features
- **Hidden layers** — extract nonlinear combinations
- **Output layer** — produces the prediction

### 7.2 Visualizing the Architecture

```python
# Create a figure
fig, ax = plt.subplots(figsize=(10, 4))

# Names of input features
inputs = ['atomic\nradius', 'electro-\nnegativity', 'melting\nT']

# Coordinates of input neurons
input_x = [0, 0, 0]
input_y = [3, 2, 1]

# Draw input neurons
for x, y, label in zip(input_x, input_y, inputs):
    circle = plt.Circle((x, y), 0.25, color='lightblue', ec='black', zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, zorder=4)

# Hidden layer
hidden_y = [2.8, 2.0, 1.2]
for y in hidden_y:
    circle = plt.Circle((2, y), 0.25, color='lightgreen', ec='black', zorder=3)
    ax.add_patch(circle)
    ax.text(2, y, 'S', ha='center', va='center', fontsize=14, zorder=4)

# Output neuron
circle = plt.Circle((4, 2.0), 0.3, color='salmon', ec='black', zorder=3)
ax.add_patch(circle)
ax.text(4, 2.0, 'bulk\nmodulus', ha='center', va='center', fontsize=9, zorder=4)

# Draw connections between layers
for y1 in input_y:
    for y2 in hidden_y:
        ax.plot([0.25, 1.75], [y1, y2], 'gray', alpha=0.4, linewidth=0.8)

for y2 in hidden_y:
    ax.plot([2.25, 3.7], [y2, 2.0], 'gray', alpha=0.4, linewidth=0.8)

# Axis settings
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(0.5, 3.5)
ax.axis('off')
ax.set_title('Simple Neural Network (Multilayer Perceptron)', fontsize=13)
plt.tight_layout()
plt.show()
```

### 7.3 How a Network Learns: Backpropagation

1. **Forward pass:** data flows through the network, producing a prediction
2. **Loss:** compute the error `L = (y - y_hat)^2`
3. **Backward pass:** using the chain rule, compute how each weight affects the loss
4. **Update:** adjust weights: `w <- w - eta * dL/dw`
5. Repeat for all batches many times

### 7.4 Effect of Model Complexity

```python
# Different architectures: from simple to complex
hidden_sizes = [(), (10,), (50,), (50, 25), (100, 50, 25)]
train_losses = []
val_losses = []

for hs in hidden_sizes:
    # () means a network without hidden layers (only input -> output)
    mlp = MLPRegressor(hidden_layer_sizes=hs, max_iter=1000,
                       early_stopping=True, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    # MSE on training
    train_losses.append(mean_squared_error(y_train, mlp.predict(X_train_scaled)))

    # MSE on validation
    val_losses.append(mean_squared_error(y_val, mlp.predict(scaler.transform(X_val))))

# Labels for the X-axis
x_labels = [str(hs) if hs else 'No hidden' for hs in hidden_sizes]
x_pos = np.arange(len(hidden_sizes))

# Build a bar chart
plt.figure(figsize=(10, 5))
plt.bar(x_pos - 0.15, train_losses, width=0.3, label='Train MSE')
plt.bar(x_pos + 0.15, val_losses, width=0.3, label='Validation MSE')
plt.xticks(x_pos, x_labels, rotation=15)
plt.ylabel('MSE (GPa^2)')
plt.title('Model Complexity vs Error')
plt.legend()
plt.tight_layout()
plt.show()
```

**What to watch for:**

- A too-simple network -> **underfitting** (high bias): both train and val errors are large
- A too-complex network -> **overfitting** (high variance): train error is low, val error is high
- The optimum is where the val error is minimal

---

## 🔬 8. Full Workflow on Materials Project Data

### 8.1 Downloading and Preparing Data

```python
from mp_api.client import MPRester
import pandas as pd
import numpy as np

with MPRester() as mpr:
    # Search for materials with reasonable elastic properties
    docs = mpr.materials.elasticity.search(
        k_vrh=(10, 400),      # bulk modulus 10-400 GPa
        g_vrh=(5, 250),       # shear modulus 5-250 GPa
        num_chunks=5,         # 5 requests
        chunk_size=500        # 500 results each
    )

# Collect the required fields
data = []
for doc in docs:
    data.append({
        'material_id': doc.material_id,
        'formula': doc.formula_pretty,
        'bulk_modulus': doc.k_vrh,
        'shear_modulus': doc.g_vrh,
        'poisson_ratio': doc.poisson_ratio,
    })

df_mp = pd.DataFrame(data)
print(f"Downloaded {len(df_mp)} materials")
print(df_mp[['bulk_modulus', 'shear_modulus', 'poisson_ratio']].describe())

# Drop rows with missing target values
df_mp = df_mp.dropna(subset=['bulk_modulus', 'shear_modulus'])
print(f"After cleaning: {len(df_mp)} materials")
```

### 8.2 Feature Generation from Composition

> **📝 Note**
> In a real workflow you would use `matminer` to automatically generate hundreds of descriptors from chemical formulas. Here, for simplicity, we use synthetic numeric features.

```python
np.random.seed(42)
n = len(df_mp)

# Synthetic features (in reality — from matminer)
df_mp['avg_atomic_radius'] = np.random.uniform(1.0, 2.2, n)
df_mp['avg_electronegativity'] = np.random.uniform(1.5, 3.5, n)
df_mp['n_elements'] = np.random.randint(1, 4, n)

# Define features and target
feature_cols = ['avg_atomic_radius', 'avg_electronegativity', 'n_elements']
X_mp = df_mp[feature_cols].values      # feature matrix
y_mp = df_mp['shear_modulus'].values   # target vector (shear modulus)

print(f"Feature matrix shape: {X_mp.shape}")
print(f"Target shape: {y_mp.shape}")
```

### 8.3 Full ML Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Split: 60% train, 20% val, 20% test ---
X_train_mp, X_temp_mp, y_train_mp, y_temp_mp = train_test_split(
    X_mp, y_mp, test_size=0.4, random_state=42
)
X_val_mp, X_test_mp, y_val_mp, y_test_mp = train_test_split(
    X_temp_mp, y_temp_mp, test_size=0.5, random_state=42
)

print(f"Train: {len(X_train_mp)}, Val: {len(X_val_mp)}, Test: {len(X_test_mp)}")

# --- Define 5 models inside Pipelines ---
mp_models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge Regression': Pipeline([
        ('scaler', StandardScaler()),
        # Ridge — linear regression with L2 regularization
        ('model', Ridge(alpha=1.0))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=300, max_depth=12,
                                         min_samples_leaf=3, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=400, learning_rate=0.05,
                                max_depth=5, subsample=0.8, random_state=42))
    ]),
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                activation='relu', solver='adam',
                                max_iter=3000, early_stopping=True,
                                random_state=42))
    ])
}

# --- Training and validation evaluation ---
mp_results = {}
for name, pipe in mp_models.items():
    pipe.fit(X_train_mp, y_train_mp)
    y_val_pred = pipe.predict(X_val_mp)

    mp_results[name] = {
        'Val MAE': mean_absolute_error(y_val_mp, y_val_pred),
        'Val RMSE': np.sqrt(mean_squared_error(y_val_mp, y_val_pred)),
        'Val R2': r2_score(y_val_mp, y_val_pred)
    }

mp_results_df = pd.DataFrame(mp_results).T
print("\n=== Validation Set Performance ===")
print(mp_results_df.round(3))

# --- Select the best model by validation R2 ---
best_name = mp_results_df['Val R2'].idxmax()
best_pipe = mp_models[best_name]

# Final training on train + val
best_pipe.fit(np.vstack([X_train_mp, X_val_mp]),
              np.concatenate([y_train_mp, y_val_mp]))

# Test evaluation
y_test_pred_mp = best_pipe.predict(X_test_mp)

print(f"\n=== Test Set Performance ({best_name}) ===")
print(f"MAE:  {mean_absolute_error(y_test_mp, y_test_pred_mp):.3f} GPa")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_mp, y_test_pred_mp)):.3f} GPa")
print(f"R2:   {r2_score(y_test_mp, y_test_pred_mp):.4f}")
```

### 8.4 Comparing All Models on the Test Set

```python
# Train ALL models on train+val and evaluate on test
test_results = []
for name, pipe in mp_models.items():
    pipe.fit(np.vstack([X_train_mp, X_val_mp]),
             np.concatenate([y_train_mp, y_val_mp]))
    y_pred = pipe.predict(X_test_mp)

    test_results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test_mp, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_mp, y_pred)),
        'R2': r2_score(y_test_mp, y_pred)
    })

# .set_index('Model') — makes the Model column the index for convenience
test_df = pd.DataFrame(test_results).set_index('Model')
print("\n=== Final Test Set Comparison ===")
print(test_df.round(4))

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE and RMSE — bar chart
test_df[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
axes[0].set_ylabel('Error (GPa)')
axes[0].set_title('Error Metrics by Model')
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(loc='upper right')

# R2 — bar chart with a horizontal line at 0.5
test_df['R2'].plot(kind='bar', ax=axes[1], color='seagreen')
axes[1].set_ylabel('R2')
axes[1].set_title('R2 Score by Model')
axes[1].tick_params(axis='x', rotation=15)
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='R2=0.5')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 💡 9. Practical Recommendations

### Which Model to Choose?

| Situation | Recommendation |
|---|---|
| First run, need a baseline | **Linear Regression** — fast, interpretable |
| Many features, risk of overfitting | **Ridge / Lasso** — regularization helps |
| Tabular data, medium size (~1000+) | **Random Forest** — reliable default |
| Need maximum accuracy | **XGBoost / LightGBM** — usually win |
| Lots of data (10,000+), complex patterns | **Neural Networks** — can beat tree models |
| Little data (< 500) | **Ridge / Random Forest** — neural nets overfit |

### Common Mistakes

> **⚠️ Warning 1: Data leakage**
> Scaling BEFORE splitting into train/test is a common mistake. Always split first, then apply `fit` only on train.

> **⚠️ Warning 2: Extrapolation**
> ML models cannot predict outside the training distribution. If trained on oxides, don't expect good predictions for intermetallics.

> **⚠️ Warning 3: Ignoring physics**
> A negative bulk modulus is physically impossible. Poisson's ratio must be between -1 and 0.5 for stable materials. Consider post-processing.

> **⚠️ Warning 4: Overfitting**
> If train R2 = 0.99 but val R2 = 0.5, the model memorized training data instead of learning a pattern. Simplify the model or collect more data.

### What to Learn Next

| Topic | Library / Tool | Why |
|---|---|---|
| Automatic descriptor generation | `matminer` | Hundreds of features from composition and structure |
| Graph neural networks | `CGCNN`, `MEGNet`, `SchNet` | Work directly with crystal structures |
| Benchmarking | `Matbench` | Compare against published results |
| Model interpretation | `SHAP`, `LIME` | Understand what the model "thinks" |
| Bayesian optimization | `scikit-optimize`, `optuna` | Search for new materials |
| Active learning | `modAL` | Select the most informative experiments |

### Useful Links

- [Materials Project](https://materialsproject.org)
- [Matbench](https://matbench.materialsproject.org)
- [Matminer documentation](https://hackingmaterials.lbl.gov/matminer/)
- [Scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [Awesome Materials Informatics](https://github.com/topics/materials-informatics)

---

## 🎯 Conclusion

This tutorial covers the full cycle of working with data in materials science:

1. ✅ **Statistics** — understand your data before modeling
2. ✅ **Regression** — 4 basic ML algorithms
3. ✅ **Data cleaning** — outliers, missing values, correlations
4. ✅ **Materials Project** — real data from an open database
5. ✅ **Pipeline** — proper splitting and training
6. ✅ **Metrics** — how to measure quality
7. ✅ **Neural networks** — simple MLPs from scratch
8. ✅ **Full workflow** — from API to final comparison

**Key takeaway:** in materials science, ML does not replace physics — it complements it. A good model is one that generalizes to new materials, not one that memorizes training data.
