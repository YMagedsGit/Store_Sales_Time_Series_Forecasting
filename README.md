# Store Sales Time Series Forecasting — Multivariate XGBoost Pipeline

A multivariate time series forecasting pipeline predicting daily retail sales for a grocery chain in Ecuador, using calendar features, lag features, rolling statistics, and oil price integration — built with a strict time-based split and Tweedie-objective XGBoost.

---

## Project Overview

The goal is to forecast daily store-level sales across 54 stores and 33 product families using historical transaction data, external oil price data, and engineered time features. Ecuador's economy is oil-dependent, meaning consumer spending fluctuates with oil prices — making this a real-world multivariate forecasting problem rather than a pure time series task.

The pipeline emphasizes correct time series methodology: no random splits, no future data leaking into lag features, and a diagnostic iteration loop where residual analysis drove additional feature engineering.

---

## Dataset

**Source:** [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)  
**Files used:** `train.csv`, `stores.csv`, `oil.csv`  
**Size:** ~3 million rows across 54 stores × 33 product families × 1,684 days  
**Target:** `sales` — daily unit sales per store-family combination

---

## Approach

### 1. Time-Based Train/Test Split
Data was split chronologically — everything before 2017 for training, 2017 onward for testing. A random split was explicitly avoided because:
- Lag features computed from future rows would leak into the training set
- The model would learn a relationship between features and targets that cannot exist at real inference time
- Evaluation metrics would be inflated and not representative of real deployment performance

### 2. Oil Price Integration & Imputation
Oil price data had significant missing values due to non-trading days (weekends, holidays). A layered imputation strategy was used:
- Primary: 7-day rolling mean to smooth short gaps
- Fallback: Global median for longer gaps

Oil price was computed at the **date level** before merging into the store-family dataframe, since oil is a market-level signal shared across all stores — not a per-store feature.

### 3. Feature Engineering
Six feature categories were engineered with documented reasoning:

| Feature | Type | Reasoning |
|---|---|---|
| `day_of_week_sin/cos` | Calendar (cyclic) | Captures weekly seasonality; sin/cos encoding prevents the model treating Monday and Sunday as maximally different |
| `month_sin/cos` | Calendar (cyclic) | Captures annual seasonality without the discontinuity of raw month integers |
| `lag_7` | Lag | Sales 7 days ago — same weekday last week, strong autocorrelation signal |
| `lag_14` | Lag | Two weeks ago — reinforces weekly pattern with more history |
| `rolling_mean_7` | Rolling window | Smoothed recent trend; `min_periods=1` used to preserve early rows instead of dropping them |
| `oil_price` | External | Oil price on that date; forward-filled then median-filled for missing days |
| `oil_lag_7` | External lag | Oil price 7 days prior — consumer spending reacts to oil changes with a lag |
| `is_holiday` | Calendar | Public holidays suppress or shift sales patterns |
| `is_payday` | Calendar | End-of-month paydays (15th and last day) drive spending spikes visible in residual analysis |

**Lag features were computed with `groupby(['store_nbr', 'family'])`** before shifting — ensuring that the lag from the last row of one store-family group never bleeds into the first row of the next group.

### 4. Modeling

Two models were trained and compared:

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression (baseline) | 104,075 | 4,999 |
| **XGBoost (Tweedie, no log transform)** | **253.9** | **72.6** |

**Why XGBoost dominates by such a large margin:**
Linear regression assumes additive, linear relationships and cannot model the interactions between `store_nbr`, `family`, `day_of_week`, and `lag_7` that drive most of the variance. With 54 stores × 33 families = 1,782 category combinations, tree-based models learn store-family-specific decision boundaries that linear models cannot represent.

**Why Tweedie objective without log transform:**
Sales data is zero-inflated and right-skewed — exactly the distribution Tweedie regression is designed for. Applying `log1p` on top of a Tweedie objective double-handles the skewness: Tweedie accounts for it internally, so adding a log transform distorts the loss function. Using `reg:squarederror` with `log1p` is a valid alternative pairing, but mixing Tweedie with log-transformed targets is not.

### 5. Residual Diagnostics & Iterative Feature Engineering
After the initial model, residuals were plotted over time. Two patterns were identified:
- **Monthly spikes** — consistent with payday behavior on the 15th and last day of each month → `is_payday` feature added
- **Holiday dips** — sudden drops on specific dates → `is_holiday` feature added

This diagnostic loop — train → evaluate → inspect residuals → engineer → retrain — is the core of the workflow.

---

## Project Structure

```
store-sales-forecasting/
│
├── Store_Sales_Time_Series_Forecasting.ipynb   # Main notebook
├── train.csv                                    # Transaction data
├── oil.csv                                      # Daily oil prices
├── stores.csv                                   # Store metadata
└── README.md
```

---

## Key Libraries

```
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

---

## Key Takeaways

- A random train/test split is incorrect for time series — lag features computed on a randomly split dataset leak future values into training, producing metrics that cannot be reproduced in deployment
- `min_periods=1` on rolling windows preserves all rows including early cold-start periods, at the cost of less stable estimates for the first few windows — a deliberate tradeoff
- The Tweedie objective and target transformation must be chosen as a consistent pair; mixing Tweedie loss with a log-transformed target produces undefined optimization behavior
- Residual analysis over time is not optional — it reveals systematic patterns the model is missing that metrics alone will not surface
- Oil price features should be computed at the date level before merging, not at the store-family level, since oil is a shared market signal

---

## Results

| Metric | Linear Regression | XGBoost (Tweedie) |
|---|---|---|
| RMSE | 104,075 | **253.9** |
| MAE | 4,999 | **72.6** |
| Objective | MSE | Tweedie |
| Log Transform | No | No |
| Engineered Features | 9 | 9 |
| Split Type | Time-based | Time-based |
