# Hull Tactical — Market Prediction

> **Kaggle Competition | S&P 500 Excess Return Forecasting**
> Real-time market prediction system built for the [Hull Tactical Index Tracking](https://www.kaggle.com/competitions/hull-tactical-market-prediction) Kaggle competition.

**Analysis Date:** November 7, 2025

---

## Team

| Participant | Rôle | Responsabilités |
|-------------|------|-----------------|
| **Pierre Chrislin DORIVAL** | Chef de projet | Coordination générale du projet, architecture technique, MLOps, DevOps, LLMOps |
| **Émile STEEVENSON** | DevOps | Infrastructure, automatisation et gestion des environnements de déploiement |
| **Jobed FELIMA** | Data Science — Finance | Analyse financière, calcul de features, exploration des données (EDA) |
| **Sébastien Witchmen ESTANIS** | Data Science — Finance | Analyse financière, calcul de features, exploration des données (EDA) |

### Descriptions détaillées

#### Pierre Chrislin DORIVAL — Chef de projet / MLOps / DevOps
Pierre Chrislin assure la **direction technique et opérationnelle** du projet. Il est responsable de la conception de l'architecture MLOps de bout en bout : depuis l'ingestion des données jusqu'au déploiement des modèles en production. Ses domaines d'expertise couvrent :
- **MLOps** : suivi des expériences avec MLflow (hébergé sur DagsHub), versioning des modèles, enregistrement automatique des artefacts et gestion du cycle de vie des modèles.
- **DevOps** : mise en place des pipelines CI/CD avec **GitHub Actions**, containerisation avec Docker, gestion des environnements reproductibles et orchestration via Makefile.
- **LLMOps** : application des meilleures pratiques de gestion des modèles de langage dans un contexte de production.
- **Développement** : développement du pipeline d'entraînement (`MLflowTrainer`), du tableau de bord Streamlit, des modules de prétraitement et d'ingénierie de features.
- **Déploiement** : intégration avec l'API Kaggle (gRPC), soumission des notebooks de prédiction et gestion de l'inférence en temps réel.

#### Émile STEEVENSON — DevOps
Émile prend en charge l'ensemble de l'**infrastructure technique** du projet. Il intervient sur :
- La configuration et la maintenance des environnements de développement et de production.
- L'automatisation des tâches récurrentes (tests, linting, packaging) via les pipelines CI/CD.
- La gestion des conteneurs Docker pour garantir la reproductibilité des expériences.
- La supervision de la fiabilité et de la disponibilité des services déployés.

#### Jobed FELIMA — Monnaie, Banque & Finance / Data Science
Jobed apporte une expertise en **finance de marché et en analyse quantitative**. Il contribue notamment à :
- La compréhension et l'interprétation des variables financières (taux d'intérêt, spreads de crédit, indicateurs macroéconomiques).
- Le **calcul de features** pertinentes à partir des données brutes : indicateurs techniques (RSI, MACD, moyennes mobiles), métriques de momentum et de volatilité.
- L'**analyse exploratoire des données (EDA)** : détection des valeurs manquantes, analyse des distributions, corrélations entre variables et identification des patterns saisonniers.
- La validation économique des signaux produits par les modèles au regard des dynamiques de marché.

#### Sébastien Witchmen ESTANIS — Monnaie, Banque & Finance / Data Science
Sébastien contribue lui aussi sur le volet **finance quantitative et data science**. Ses missions incluent :
- L'analyse des données financières sous l'angle de la **théorie monétaire et bancaire** : interprétation des indicateurs de politique monétaire (Fed Funds Rate, courbe des taux), du sentiment de marché et des ratios de valorisation.
- La construction et la validation des **features financières** : agrégats par catégorie (V, M, S, P, I, E), indicateurs de sentiment, spreads et variables de régime.
- La conduite de l'**EDA** : visualisation des séries temporelles, analyse de stationnarité, étude des distributions de rendements et identification des biais potentiels.
- La garantie de la cohérence économique des variables utilisées en entrée des modèles de machine learning.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Data Description](#3-data-description)
4. [ML Pipeline](#4-ml-pipeline)
5. [Models](#5-models)
6. [Inference API Architecture](#6-inference-api-architecture)
7. [Experiment Tracking with MLflow & DagsHub](#7-experiment-tracking-with-mlflow--dagshub)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [Validation Strategy](#9-validation-strategy)
10. [Setup & Usage](#10-setup--usage)
11. [Pre-Submission Checklist](#11-pre-submission-checklist)
12. [Competition Phases](#12-competition-phases)
13. [Key Concepts](#13-key-concepts)

---

## 1. Project Overview

### Objective

This project aims to predict the **daily S&P 500 excess returns** (`market_forward_excess_returns`), a normalized measure of how much the market outperforms its historical expectation, while respecting a **120% volatility constraint**.

The predicted value is not the raw return but a risk-adjusted deviation from a 5-year rolling average, winsorized to remove extreme outliers. The model output is an **allocation signal** between `0.0` (fully out of market) and `2.0` (200% leveraged), used directly in a real trading strategy.

### Intellectual Challenge

This competition directly challenges the **Efficient Market Hypothesis (EMH)**, which asserts that prices already incorporate all available information, making systematic outperformance impossible. Our goal is to demonstrate that structured, data-driven signals can capture predictable patterns in market behavior.

### What Makes This Competition Unique

Unlike standard Kaggle challenges where the test set is static:
- Model notebooks are **executed automatically every trading day** on Kaggle's infrastructure.
- Predictions are made on **live market data** over a 6-month forecasting window (December 2025 – June 2026).
- The private leaderboard is built on real, unseen market outcomes — there is no way to overfit to it.

---

## 2. Project Structure

```
hull-tactical-market-prediction/
│
├── src/                         # Core Python package
│   ├── data/
│   │   ├── kaggle_loader.py     # Loads train/test data from Kaggle
│   │   └── preprocessor.py      # DataPipeline: NaN handling, imputation, feature selection
│   │
│   ├── features/
│   │   └── feature_engineer.py  # Advanced feature construction (rolling, RSI, MACD, etc.)
│   │
│   ├── models/
│   │   ├── base.py              # Abstract BaseModel + ModelFactory registry
│   │   ├── gradient_boosting.py # LightGBM, XGBoost, CatBoost, RandomForest implementations
│   │   ├── ensemble.py          # EnsembleModel: weighted/simple averaging of base models
│   │   └── registry.py          # Model registration and lookup
│   │
│   ├── training/
│   │   ├── trainer.py           # MLflowTrainer: end-to-end training orchestrator
│   │   ├── validator.py         # WalkForwardValidator: time-series-safe cross-validation
│   │   └── hyperopt.py          # Hyperparameter optimization utilities
│   │
│   └── utils/
│       ├── config.py            # Config loader (Hydra/YAML-based)
│       ├── logger.py            # Structured logging
│       └── metrics.py           # ModelMetrics: RMSE, MAE, R², Directional Accuracy, Sharpe
│
├── app/                         # Streamlit web dashboard
│   ├── streamlit_app.py         # Main dashboard entry point
│   ├── components/              # Reusable UI components
│   └── pages/                   # Multi-page app (Overview, Predictions, Monitoring)
│
├── configs/
│   ├── config.yaml              # Main configuration (data, MLflow, training, preprocessing)
│   ├── features.yaml            # Feature engineering configuration
│   └── model_params.yaml        # Hyperparameters for all model types
│
├── notebooks/                   # Jupyter notebooks for EDA and experimentation
├── scripts/                     # Standalone training and evaluation scripts
├── tests/                       # Unit and integration tests
├── artifacts/                   # Saved models, feature lists, logs (generated)
├── visualizations/              # Saved plots and charts
│
├── docker/                      # Docker configuration for reproducible environments
├── Makefile                     # Common commands (train, test, lint, deploy)
├── pyproject.toml               # Package metadata and tool configuration
├── requirements.txt             # Production dependencies
└── requirements-dev.txt         # Development dependencies
```

---

## 3. Data Description

### Training Set — `train.csv` (8,991 rows × 98 columns)

Each row represents one **trading day**, identified by a sequential `date_id` (0 to 8990, spanning roughly 35 years of market data).

#### Feature Categories (95 predictive columns)

| Prefix | Category | Count | Description |
|--------|----------|-------|-------------|
| `D*` | Dummy / Binary | 9 | Binary and categorical regime indicators (D1–D9) |
| `E*` | Macroeconomic | 20 | Economic indicators: GDP, inflation, employment (E1–E20) |
| `I*` | Interest Rates | 9 | Yield curve, Fed Funds Rate, credit spreads (I1–I9) |
| `M*` | Market Dynamics | 18 | Breadth, sector rotation, market structure (M1–M18) |
| `P*` | Price / Valuation | 13 | P/E ratios, earnings yield, valuation multiples (P1–P13) |
| `S*` | Sentiment | 12 | Survey data, put/call ratios, investor positioning (S1–S12) |
| `V*` | Volatility | 13 | Realized and implied volatility measures (V1–V13) |
| `MOM*` | Momentum | 1 | Aggregate momentum indicator |

> **Note:** The earliest years contain substantial missing values due to incomplete historical data coverage. Columns with more than 30% missing values are dropped during preprocessing.

#### Target Columns (train only)

| Column | Description |
|--------|-------------|
| `forward_returns` | Next-day raw return: `(Price_t+1 - Price_t) / Price_t` |
| `risk_free_rate` | Daily Federal Funds Rate |
| `market_forward_excess_returns` | **Main prediction target** (see formula below) |

**Target formula:**
```
excess_returns     = forward_returns - risk_free_rate
mean_5y            = 5-year rolling mean of excess_returns
deviation          = excess_returns - mean_5y
MAD                = Median Absolute Deviation of deviation
market_forward_excess_returns = winsorize(deviation, threshold = MAD × 4)
```

This construction removes the long-term trend and caps extreme values, producing a stationary, bounded signal.

---

### Test Set — `test.csv` (progressive, ~180 rows)

During training (Phase 1), a mock test file is provided containing the **last 180 date_ids** from the training set (8811–8990). During forecasting (Phase 2), new rows are appended daily with real market data.

#### Additional columns (test only)

| Column | Description |
|--------|-------------|
| `is_scored` | Whether this row counts in the leaderboard |
| `lagged_forward_returns` | Previous day's raw return (1-day lag) |
| `lagged_risk_free_rate` | Previous day's risk-free rate |
| `lagged_market_forward_excess_returns` | Previous day's target value |

> The lag design prevents **look-ahead bias**: in production, we only know what happened after market close, so yesterday's returns are the most recent observable signal.

---

## 4. ML Pipeline

The training pipeline is orchestrated by `MLflowTrainer` (`src/training/trainer.py`) and follows these steps:

```
Raw Data (train.csv)
      │
      ▼
[1] KaggleDataLoader          — Load raw CSV, optional date cutoff
      │
      ▼
[2] FeatureEngineer           — Build derived features (see below)
      │
      ▼
[3] Warm-up period skip       — Drop first N rows (rolling feature instability)
      │
      ▼
[4] DataPipeline
    ├── Drop columns > 30% NaN
    ├── CatBoost imputation (remaining NaN)
    └── Feature selection (top 150 features)
          ├── LGB importance  (weight: 50%)
          ├── Pearson correlation (weight: 25%)
          └── Mutual information (weight: 25%)
      │
      ▼
[5] Train/Val split (80% / 20%, time-ordered)
      │
      ▼
[6] Model training (with early stopping on validation set)
      │
      ▼
[7] MLflow logging (params, metrics, artifacts, model)
      │
      ▼
[8] Save to artifacts/
```

### Feature Engineering (`src/features/feature_engineer.py`)

The `FeatureEngineer` class constructs the following feature groups:

| Group | Description |
|-------|-------------|
| **Lagged targets** | 1-day lag of excess returns, returns, risk-free rate |
| **Rolling statistics** | Mean, std, min, max, skew, kurtosis over 5/10/20/60-day windows |
| **Target-derived** | Z-score, mean reversion, autocorrelation, momentum over lagged target |
| **Category aggregates** | Per-category (V, M, S, P, I, E) mean, std, range, percentile rank |
| **Category changes** | 5-day / 20-day delta per category |
| **Technical indicators** | RSI (7/14/21 periods), MACD (12/26/9) on returns and excess returns |
| **Interaction features** | Volatility × Momentum, Sentiment × Volatility, Price/Volatility ratio |
| **Time features** | Day/week/month of year (cyclical sin/cos encoding, based on 252 trading days/year) |

---

## 5. Models

### Implemented Models

All models share a common `BaseModel` interface and are registered in `ModelFactory`:

| Model | Class | Library | Notes |
|-------|-------|---------|-------|
| LightGBM | `LightGBMModel` | `lightgbm` | Early stopping on validation set |
| XGBoost | `XGBoostModel` | `xgboost` | Early stopping on validation set |
| CatBoost | `CatBoostModel` | `catboost` | Best model selection enabled |
| Random Forest | `RandomForestModel` | `scikit-learn` | No early stopping (trains on full pass) |
| Ensemble | `EnsembleModel` | custom | Weighted or simple average of above |

### Ensemble Strategy

The `EnsembleModel` combines predictions from multiple trained models:

```python
# Default composition
models = [LightGBM (40%), XGBoost (30%), CatBoost (20%), RandomForest (10%)]

# Prediction: weighted average of individual model outputs
final_prediction = Σ (weight_i × model_i.predict(X))
```

Weights are configurable in `configs/model_params.yaml`. The best single model (by R²) can optionally be registered to the MLflow Model Registry.

### Metrics Tracked

- **RMSE** — Root Mean Squared Error (primary regression metric)
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of determination
- **Directional Accuracy** — % of days where sign of prediction matches actual sign
- **Sharpe Ratio** — Risk-adjusted return of the simulated strategy

---

## 6. Inference API Architecture

Kaggle's evaluation framework requires a **client-server architecture** using gRPC:

```
┌─────────────────────────────────┐      gRPC      ┌───────────────────────────────┐
│         DefaultGateway          │ ◄────────────► │     DefaultInferenceServer    │
│  (Kaggle evaluation framework)  │                │     (our prediction code)     │
│                                 │                │                               │
│  - Loads test.csv               │                │  - Receives one batch/day     │
│  - Sends batches by date_id     │                │  - Applies feature pipeline   │
│  - Validates predictions        │                │  - Returns allocation [0, 2]  │
│  - Writes submission.parquet    │                │                               │
└─────────────────────────────────┘                └───────────────────────────────┘
```

### Allocation Mapping

The model's raw regression output (predicted excess return) is mapped to an allocation:

```
allocation ∈ [0.0, 2.0]

  0.0 → 100% cash (no market exposure)
  1.0 → 100% in S&P 500 (neutral/baseline)
  2.0 → 200% leveraged long position
```

### Submission Notebook Template

```python
from kaggle_evaluation import default_inference_server
import pickle, json

# Load trained artifacts
with open("artifacts/ensemble_final.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/selected_features.json") as f:
    feature_cols = json.load(f)

def predict(test_batch):
    """
    Args:
        test_batch: DataFrame — features for one trading day

    Returns:
        float: Market allocation between 0.0 and 2.0
    """
    X = test_batch[feature_cols].fillna(0)
    raw_pred = model.predict(X)[0]

    # Map predicted excess return to allocation
    allocation = 1.0 + raw_pred * scaling_factor
    return float(np.clip(allocation, 0.0, 2.0))

# Start server
inference_server = default_inference_server.DefaultInferenceServer(predict)
inference_server.serve()  # For Kaggle submission
# inference_server.run_local_gateway()  # For local testing
```

---

## 7. Experiment Tracking with MLflow & DagsHub

All training runs are tracked via **MLflow** hosted on **DagsHub**:

- **Tracking URI:** `https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow`
- **Experiment name:** `hull-tactical-experiments`

### Logged Artifacts per Run

| Type | Content |
|------|---------|
| Parameters | NaN threshold, imputation method, feature selection method, n_features, model hyperparameters |
| Metrics | RMSE, MAE, R², Directional Accuracy, Sharpe Ratio |
| Artifacts | `feature_importance.csv`, `selected_features.json`, serialized model |
| Model | Logged via `mlflow.sklearn.log_model`, optionally registered |

### Configuration (`configs/config.yaml`)

```yaml
mlflow:
  tracking_uri: https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow
  experiment_name: hull-tactical-experiments

preprocessing:
  nan_threshold: 0.30          # Drop columns with >30% missing values
  use_catboost_imputation: true
  feature_selection:
    method: combined            # lgb_importance + correlation + mutual_info
    n_features: 150

training:
  test_size: 0.2
  warmup_period: 100           # Skip first 100 rows (rolling features warm-up)
  early_stopping_rounds: 50
```

---

## 8. Streamlit Dashboard

A monitoring and visualization dashboard is available at `app/streamlit_app.py`.

**Launch:**
```bash
streamlit run app/streamlit_app.py
```

**Features:**
- Model selector (LightGBM, XGBoost, CatBoost, Ensemble)
- Key metrics display: Latest prediction, R², RMSE, Directional Accuracy
- Market Overview tab
- Prediction Analysis tab
- Performance Monitoring tab
- Auto-refresh mode (configurable interval)

---

## 9. Validation Strategy

### Walk-Forward Validation

Standard k-fold cross-validation is **invalid for time series** because it allows training on future data. We use `WalkForwardValidator` (`src/training/validator.py`), which respects temporal order:

```
Split 1:  [Train: 0 → 7200] → [Test: 7200 → 7600]
Split 2:  [Train: 0 → 7600] → [Test: 7600 → 8000]
Split 3:  [Train: 0 → 8000] → [Test: 8000 → 8400]
Split 4:  [Train: 0 → 8400] → [Test: 8400 → 8700]
Split 5:  [Train: 0 → 8700] → [Test: 8700 → 8990]
```

Configuration: 5 splits, each test window covering ~16.7% of data.

### Sharpe Ratio Calculation

Used to measure the risk-adjusted quality of the allocation strategy:

```python
portfolio_returns = allocation * forward_returns
excess_returns    = portfolio_returns - risk_free_rate
sharpe_ratio      = mean(excess_returns) / std(excess_returns)
```

A Sharpe ratio above **1.0** on the validation set is the medium-term target.

---

## 10. Setup & Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hull-tactical-market-prediction.git
cd hull-tactical-market-prediction

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev tools (linting, testing)
```

### Environment Variables

```bash
# Set MLflow / DagsHub credentials
export MLFLOW_TRACKING_URI=https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_token
```

### Training a Model

```python
from src.training.trainer import MLflowTrainer

trainer = MLflowTrainer(
    n_features=150,
    nan_threshold=0.30,
    use_catboost_imputation=True,
    feature_selection_method="combined",
)

# Load and preprocess data
trainer.load_and_prepare_data(use_feature_engineering=True)

# Train a single model
model, metrics = trainer.train_model("lightgbm")

# Train all models + select best
results = trainer.train_all_models(register_best=True)

# Train ensemble
ensemble, metrics = trainer.train_ensemble()
```

### Walk-Forward Validation

```python
results = trainer.run_walk_forward_validation(
    model_type="lightgbm",
    n_splits=5,
)
```

### Local API Testing

```bash
# Run the inference server locally against the mock test set
python -c "
from kaggle_evaluation import default_inference_server
inference_server = default_inference_server.DefaultInferenceServer(predict)
inference_server.run_local_gateway()
"
```

### Run Tests

```bash
pytest tests/ -v
```

### Makefile Shortcuts

```bash
make train    # Run full training pipeline
make test     # Run test suite
make lint     # Run linting (ruff, black)
make dashboard  # Launch Streamlit app
```

---

## 11. Pre-Submission Checklist

- [ ] Model runs without errors on the local mock test set
- [ ] `predict()` returns a value strictly in `[0.0, 2.0]`
- [ ] Notebook starts the server with `inference_server.serve()`
- [ ] Cold start time < 5 minutes (Kaggle execution limit)
- [ ] Per-batch prediction time < 5 minutes (timeout limit)
- [ ] Model validated with walk-forward validation (no look-ahead)
- [ ] Volatility constraint (120%) is respected
- [ ] All dependencies installable in the Kaggle environment
- [ ] No look-ahead bias anywhere in the pipeline

---

## 12. Competition Phases

### Phase 1 — Model Development (Sep 16 – Dec 15, 2025)

- Data: 8,991 training days + mock test (last 180 days)
- Public leaderboard: **not meaningful** (test data already seen in training)
- Goal: build, validate, and finalize the submission notebook

### Phase 2 — Live Forecasting (Dec 15, 2025 – Jun 16, 2026)

- New market data added daily (~180 new trading days)
- Notebooks re-executed automatically each day
- Private leaderboard: scored on real, unseen market outcomes
- Final ranking: based on **Sharpe ratio** of the live allocation strategy

---

## 13. Key Concepts

| Concept | Definition |
|---------|------------|
| **Efficient Market Hypothesis (EMH)** | Theory that all market information is instantly reflected in prices, making systematic outperformance impossible |
| **Excess Returns** | Returns above the risk-free rate (Federal Funds Rate) |
| **Winsorization** | Capping extreme values at a threshold (here: MAD × 4) to reduce outlier influence |
| **Sharpe Ratio** | Mean excess return divided by its standard deviation — measures risk-adjusted performance |
| **Walk-Forward Validation** | Time-series cross-validation that always trains on past data and tests on future data |
| **Look-Ahead Bias** | Using future information during training, leading to unrealistically optimistic results |
| **Allocation Signal** | The model output [0, 2]: fraction of capital deployed in the market |
| **Volatility Constraint** | Portfolio volatility must not exceed 120% of market volatility |
| **CatBoost Imputation** | Using a gradient boosting model to predict and fill missing feature values |

---

## Resources

- Competition page: [Hull Tactical Index Tracking on Kaggle](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- MLflow tracking: [DagsHub — hull-tactical-market-prediction](https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow)
- Libraries: `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `mlflow`, `streamlit`, `pandas`, `numpy`, `statsmodels`

---

*This project combines quantitative finance and applied machine learning to tackle one of the most challenging prediction problems in financial markets.*
