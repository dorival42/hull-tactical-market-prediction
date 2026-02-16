# 🎯 HULL TACTICAL - MARKET PREDICTION
## Complete Kaggle Challenge Analysis

**Analysis Date:** November 7, 2025  
**Participants:**
- Pierre Chrislin DORIVAL
- Emile STEEVENSON
- Jobed FELIMA
- Sebastien Witchmen ESTANIS

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Data Description](#data-description)
4. [API Architecture](#api-architecture)
5. [Submission Methodology](#submission-methodology)
6. [Recommended Strategy](#recommended-strategy)

---

## 🎯 OVERVIEW

### Challenge Objective

Predict **S&P 500 excess returns** (`market_forward_excess_returns`) while respecting a **120% volatility constraint**.

### Intellectual Challenge

Challenge the **Efficient Market Hypothesis (EMH)**, which states that it is impossible to systematically beat the market.

### Unique Feature

Unlike most Kaggle competitions, our models will be **run in real time** on the market for 6 months after the submission deadline.

---

## 📊 DATA DESCRIPTION

### TRAIN.CSV (8,991 rows × 98 columns)

#### 🔑 Identifier
- `date_id`: Unique identifier for each trading day (0 to 8990)

#### 📈 FEATURES (95 predictive variable columns)

| Category | Prefix | Count | Description |
|----------|--------|-------|-------------|
| **Dummy/Binary** | `D*` | 9 | Binary/categorical variables (D1-D9) |
| **Macro Economic** | `E*` | 20 | Macroeconomic indicators (E1-E20) |
| **Interest Rate** | `I*` | 9 | Interest rates (I1-I9) |
| **Market Dynamics** | `M*` | 18 | Market dynamics (M1-M18) |
| **Price/Valuation** | `P*` | 13 | Price and valuation (P1-P13) |
| **Sentiment** | `S*` | 12 | Sentiment indicators (S1-S12) |
| **Volatility** | `V*` | 13 | Volatility indicators (V1-V13) |
| **Momentum** | `MOM*` | 1 | Momentum indicator |

**⚠️ IMPORTANT**: The earliest years contain **many missing values** (incomplete coverage in older data).

#### 🎯 TARGETS (3 columns - TRAIN ONLY)

1. **`forward_returns`**
   - Returns obtained by buying the S&P 500 and selling it the next day
   - Formula: `(Price_t+1 - Price_t) / Price_t`

2. **`risk_free_rate`**
   - Federal Funds Rate
   - Used to calculate excess returns

3. **`market_forward_excess_returns`** ⭐ **MAIN TARGET**
   - Excess returns relative to expectations
   - **Calculation**:
     ```
     1. excess_returns = forward_returns - risk_free_rate
     2. mean_5y = 5-year rolling average of excess_returns
     3. deviation = excess_returns - mean_5y
     4. MAD = Median Absolute Deviation of deviation
     5. market_forward_excess_returns = winsorize(deviation, MAD × 4)
     ```
   - **This is the value we need to predict**

---

### TEST.CSV (11 rows × 99 columns)

#### Structure during the training phase
- **Mock test set**: Copy of the **last 180 `date_id`s** from the train set (8811-8990)
- **Only 10 rows** in the provided mock file

#### Additional columns (compared to train)

| Column | Description |
|--------|-------------|
| **`is_scored`** | Indicates whether the row is included in the evaluation |
| **`lagged_forward_returns`** | `forward_returns` with a 1-day lag |
| **`lagged_risk_free_rate`** | `risk_free_rate` with a 1-day lag |
| **`lagged_market_forward_excess_returns`** | `market_forward_excess_returns` with a 1-day lag |

**⚠️ Why the lag?**  
Simulates reality: we only know returns **after market close**. This prevents "look-ahead bias".

---

## 🔄 COMPETITION PHASES

### Phase 1: Model Training (Sep 16 - Dec 15, 2025)

```
TRAIN SET
├── Date IDs: 0 → 8990
├── Features: D*, E*, I*, M*, P*, S*, V*, MOM*
└── Targets: forward_returns, risk_free_rate, market_forward_excess_returns

TEST SET (Mock)
├── Date IDs: 8811 → 8990 (copy of last 180 days from train)
├── Features: Identical to train
├── Lagged targets: Available with 1-day lag
└── is_scored: True for all days
```

**Public Leaderboard**: ⚠️ **NOT MEANINGFUL** (data already seen in the train set)

---

### Phase 2: Forecasting (Dec 15, 2025 - Jun 16, 2026)

```
TEST SET (Real-time)
├── New market data served progressively
├── Your notebooks run AUTOMATICALLY every day
├── is_scored: True only for new trading days
└── Duration: ~6 months = ~180 trading days
```

**Private Leaderboard**: Calculated on real market predictions in real time.

---

## 🏗️ API ARCHITECTURE

### Main Components

#### 1. **Gateway** (`default_gateway.py`)
- **Role**: Coordinates the evaluation
- **Responsibilities**:
  - Load test data
  - Send batches to the inference server
  - Validate predictions
  - Generate the submission file

```python
class DefaultGateway(kaggle_evaluation.core.templates.Gateway):
    def generate_data_batches(self):
        # Reads test.csv
        # Generates batches by date_id
        # Yield (test_batch, batch_id)
    
    def competition_specific_validation(self, prediction, row_ids, data_batch):
        # Challenge-specific validation
        pass
```

#### 2. **InferenceServer** (`default_inference_server.py`)
- **Role**: Our prediction code
- **Responsibilities**:
  - Receive data batches
  - Generate predictions
  - Return allocations (0.0 to 2.0)

```python
class DefaultInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def predict(self, test_batch):
        # OUR CODE HERE
        # Return an allocation between 0.0 and 2.0
        return allocation
```

#### 3. **gRPC Communication**
- Uses Protocol Buffers for communication
- Enables DataFrame exchange between Gateway and InferenceServer

---

## 📤 SUBMISSION METHODOLOGY

### What We Need to Submit

**A NOTEBOOK** that:
1. Defines a `predict(test_batch)` function
2. Creates an `InferenceServer` with that function
3. Starts the server with `server.serve()`

### Minimal Example

```python
from kaggle_evaluation import default_inference_server

def predict(test_batch):
    """
    Args:
        test_batch: DataFrame with features for one date_id
    
    Returns:
        float or Series: Allocation between 0.0 and 2.0
    """
    # Our prediction model
    prediction = model.predict(test_batch)

    
    return prediction

# Create the server with our predict function
inference_server = default_inference_server.DefaultInferenceServer(predict)

# Test locally
inference_server.run_local_gateway()

# For Kaggle submission
inference_server.serve()
```

---

## 🎯 RECOMMENDED STRATEGY

### 1. Exploratory Data Analysis (EDA)

#### A. Temporal Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('train.csv')

# Analyze feature coverage over time
missing_by_date = train.isnull().sum(axis=1)
plt.plot(train['date_id'], missing_by_date)
plt.title('Missing Values by Date')
plt.show()

# Analyze target distribution
train['market_forward_excess_returns'].hist(bins=100)
plt.title('Excess Returns Distribution')
plt.show()
```

#### B. Feature Analysis by Category
```python
# Group by category
D_features = [col for col in train.columns if col.startswith('D')]
E_features = [col for col in train.columns if col.startswith('E')]
# ... etc

# Analyze correlations
correlation_with_target = train[E_features + ['market_forward_excess_returns']].corr()['market_forward_excess_returns']
print(correlation_with_target.sort_values(ascending=False))
```

---

### 2. Feature Engineering

#### A. Handling Missing Values
```python
# Possible strategies:
# 1. Limit analysis to recent years (fewer missing values)
# 2. Forward fill for certain features (prices, rates)
# 3. Models robust to missing values (LightGBM)
```

#### B. Derived Features
```python
# Lag features
for lag in [1, 5, 10, 20]:
    train[f'forward_returns_lag_{lag}'] = train['forward_returns'].shift(lag)

# Rolling statistics
for window in [5, 10, 20, 60]:
    train[f'volatility_{window}d'] = train['forward_returns'].rolling(window).std()
    train[f'mean_return_{window}d'] = train['forward_returns'].rolling(window).mean()

# Momentum indicators
train['momentum_5_20'] = (
    train['forward_returns'].rolling(5).mean() - 
    train['forward_returns'].rolling(20).mean()
)
```

---

### 3. Modeling

#### A. Simple Baseline
```python
# Strategy 1: Constant allocation
def baseline_constant(test_batch):
    return 1.0  # Always 100% invested

# Strategy 2: Based on recent volatility
def baseline_volatility(test_batch):
    recent_vol = test_batch['V1'].iloc[0]  # Example
    if recent_vol > threshold_high:
        return 0.5  # Reduce exposure
    else:
        return 1.5  # Increase exposure
```

#### B. ML Models

**Option 1: Direct Regression**
```python
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Directly predict market_forward_excess_returns
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5
)

# Selected features
features = D_features + E_features + I_features + ['lagged_forward_returns']

# Train
model.fit(train[features], train['market_forward_excess_returns'])

# Convert prediction to allocation
def predict(test_batch):
    pred_return = model.predict(test_batch[features])
    
    
    return pred_return
```

**Option 2: Classification (Bear/Bull/Neutral)**
```python
from sklearn.ensemble import GradientBoostingClassifier

# Create classes
train['signal'] = pd.cut(
    train['market_forward_excess_returns'],
    bins=[-np.inf, -0.003, 0.003, np.inf],
    labels=['bear', 'neutral', 'bull']
)

# Classification model
model = GradientBoostingClassifier()
model.fit(train[features], train['signal'])

# Allocation based on predicted class
def predict(test_batch):
    signal = model.predict(test_batch[features])[0]
    
    
    return signal
```

---

### 4. Risk Management (Volatility Constraint)

```python
def predict_with_risk_management(test_batch, model, max_vol_ratio=1.2):
    # Raw prediction
    raw_allocation = model.predict(test_batch)
    
    # Estimate anticipated volatility
    estimated_vol = estimate_volatility(test_batch)
    market_vol = test_batch['V1'].iloc[0]  # Example
    
    # Adjust if necessary
    if estimated_vol > max_vol_ratio * market_vol:
        # Reduce allocation to respect the constraint
        scaling_factor = (max_vol_ratio * market_vol) / estimated_vol
        adjusted_allocation = raw_allocation * scaling_factor
    else:
        adjusted_allocation = raw_allocation
    
    # Ensure allocation stays within [0, 2]
    return np.clip(adjusted_allocation, 0.0, 2.0)
```

---

### 5. Validation

#### A. Walk-Forward Validation
```python
# Never train on future data
# Simulate the prediction process day by day

results = []
for i in range(train_size, len(train)):
    # Train on past data only
    train_window = train.iloc[max(0, i-lookback):i]
    test_day = train.iloc[i:i+1]
    
    # Train the model
    model.fit(train_window[features], train_window['target'])
    
    # Predict
    prediction = model.predict(test_day[features])
    results.append(prediction)
```

#### B. Sharpe Ratio Calculation
```python
def calculate_sharpe(allocations, returns, risk_free_rates):
    # Portfolio returns
    portfolio_returns = allocations * returns
    
    # Excess returns
    excess_returns = portfolio_returns - risk_free_rates
    
    # Sharpe ratio
    sharpe = excess_returns.mean() / excess_returns.std()
    
    return sharpe
```

---

### 6. Local Testing with the API

```python
from kaggle_evaluation import default_inference_server

# Your prediction function
def predict(test_batch):
    # Your code here
    return allocation

# Create the server
inference_server = default_inference_server.DefaultInferenceServer(predict)

# Test locally on the mock test set
inference_server.run_local_gateway()

# Check submission.parquet
import pandas as pd
submission = pd.read_parquet('submission.parquet')
print(submission.head())
```

---

## 🎲 ADVANCED APPROACHES

### 1. Model Ensemble
```python
# Combine multiple models
predictions = []
predictions.append(model_lgb.predict(test_batch) * 0.4)
predictions.append(model_xgb.predict(test_batch) * 0.3)
predictions.append(model_rf.predict(test_batch) * 0.3)

final_prediction = sum(predictions)
```

### 2. Time Series Models
```python
from statsmodels.tsa.arima.model import ARIMA
# ARIMA, GARCH for volatility
```

### 3. Deep Learning
```python
import torch
import torch.nn as nn

class MarketPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

---

## 📝 PRE-SUBMISSION CHECKLIST

- [ ] The model runs without errors on the local test set
- [ ] The `predict()` function returns values between 0.0 and 2.0
- [ ] The notebook starts the server with `inference_server.serve()`
- [ ] Startup time is < 5 minutes (Kaggle limit)
- [ ] Prediction per batch takes < 5 minutes (timeout)
- [ ] The model has been validated with walk-forward validation
- [ ] The volatility constraint is respected
- [ ] Dependencies are installed correctly
- [ ] The code contains no look-ahead bias

---

## 🚀 PROJECT DEVELOPMENT STEPS

### Step 1: In-Depth EDA
1. Load and explore `train.csv`
2. Analyze missing values by time period
3. Visualize the distribution of `market_forward_excess_returns`
4. Study correlations between features and target

### Step 2: Baseline
1. Create a simple baseline strategy
2. Test with the local API
3. Calculate the Sharpe ratio on the validation set

### Step 3: Feature Engineering
1. Create lag features
2. Calculate rolling statistics
3. Add momentum indicators

### Step 4: Modeling
1. Train multiple models (LightGBM, XGBoost, RF)
2. Walk-forward validation
3. Optimize hyperparameters

### Step 5: Risk Management
1. Implement the volatility constraint
2. Test different allocation strategies
3. Validate the Sharpe ratio

### Step 6: Submission
1. Create the submission notebook
2. Test locally with the API
3. Submit on Kaggle
4. Monitor real-time performance

---

## 📚 RESOURCES

### Kaggle Documentation
- Competition page: https://www.kaggle.com/competitions/hull-tactical-market-prediction
- Example notebook: Available in the "Code" section
- Discussion forum: For asking questions

### Key Concepts
- Efficient Market Hypothesis (EMH)
- Sharpe Ratio
- Volatility constraint
- Walk-forward validation
- Time series forecasting

### Useful Libraries
- `pandas`, `polars`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Classical ML
- `lightgbm`, `xgboost`, `catboost`: Boosting
- `statsmodels`: Time series
- `pytorch`, `tensorflow`: Deep learning

---

## 🏆 OBJECTIVES

### Short term (1-2 weeks)
- [ ] Fully understand the data
- [ ] Create a working baseline
- [ ] Submit a first version

### Medium term (1 month)
- [ ] Develop advanced features
- [ ] Optimize the model
- [ ] Achieve a Sharpe ratio > 1.0 in validation

### Long term (until Dec 15)
- [ ] Model ensemble
- [ ] Robust risk management strategy
- [ ] Aim for the top 10% of the leaderboard

---

**This competition is exciting — it could challenge one of the fundamental theories of modern finance! 🚀📈**  
**🚀 This is a great asset for our future careers in Finance, Data Science, and Machine Learning!**
