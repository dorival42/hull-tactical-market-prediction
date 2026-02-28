# Hull Tactical — Airflow MLOps Stack

Full orchestration layer for the Hull Tactical Market Prediction project.
Adds Apache Airflow 2.8, ChromaDB, and a local MLflow server on top of the existing Streamlit dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      hull-tactical-net                           │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────────────────────────┐    │
│  │  PostgreSQL  │◄───│  Airflow webserver  :8080            │    │
│  │  :5432       │    │  Airflow scheduler  (internal)       │    │
│  │  (metadata + │    └──────────────────────────────────────┘    │
│  │   mlflow DB) │                   │ triggers                   │
│  └─────────────┘                   ▼                             │
│                          ┌─────────────────┐                     │
│                          │   3 DAGs        │                     │
│                          │                 │                     │
│                          │  dag_data_      │──► artifacts/data/  │
│                          │  pipeline       │    market/*.parquet  │
│                          │  (hourly)       │                     │
│                          │                 │                     │
│                          │  dag_rag_       │──► ChromaDB :8000   │
│                          │  update         │    financial_news   │
│                          │  (hourly)       │                     │
│                          │                 │                     │
│                          │  dag_model_     │──► MLflow :5000     │
│                          │  retraining     │    artifacts/*.pkl  │
│                          │  (weekly)       │                     │
│                          └─────────────────┘                     │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │  MLflow      │    │  ChromaDB    │    │  Streamlit :8501 │    │
│  │  :5000       │    │  :8000       │    │  (unchanged)     │    │
│  └─────────────┘    └──────────────┘    └──────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Services summary

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | postgres:15-alpine | 5432 | Airflow metadata DB + MLflow backend |
| `airflow-init` | hull-tactical-airflow:2.8 | — | One-shot DB migration + admin user |
| `airflow-webserver` | hull-tactical-airflow:2.8 | **8080** | Airflow UI |
| `airflow-scheduler` | hull-tactical-airflow:2.8 | — | DAG trigger loop |
| `mlflow` | ghcr.io/mlflow/mlflow:v2.9.0 | **5000** | Experiment tracking UI |
| `chromadb` | chromadb/chroma:0.4.22 | **8000** | Vector store for RAG |
| `streamlit` | hull-tactical-app (local build) | **8501** | Monitoring dashboard |

---

## Quick start

### 1. Prerequisites

- Docker ≥ 24 + **Docker Compose v2 standalone** (`docker-compose` binary, not the `docker compose` plugin)
  - Check: `docker-compose --version` → should print `Docker Compose version v2.x`
- At least **8 GB RAM** allocated to Docker (Airflow + ML libs)
- A [NewsAPI](https://newsapi.org) free API key

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```dotenv
# Generate a Fernet key:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW_FERNET_KEY=<generated_key>
AIRFLOW_SECRET_KEY=<any_random_string>

POSTGRES_PASSWORD=<strong_password>
AIRFLOW_ADMIN_PASSWORD=<strong_password>

NEWS_API_KEY=<your_newsapi_key>

# Kaggle credentials (for the retraining DAG to load train.csv)
KAGGLE_USERNAME=<your_kaggle_username>
KAGGLE_KEY=<your_kaggle_api_key>
```

### 3. Build and start the stack

```bash
# Build the custom Airflow image (first run only — ~5 min)
docker-compose -f docker-compose.airflow.yml build airflow-webserver

# Start all services
docker-compose -f docker-compose.airflow.yml up -d

# Watch logs until airflow-init completes
docker-compose -f docker-compose.airflow.yml logs -f airflow-init
```

Wait for the message: `✅ Airflow init complete.`

### 4. Open the UIs

| UI | URL | Default credentials |
|----|-----|---------------------|
| Airflow | http://localhost:8080 | admin / (your AIRFLOW_ADMIN_PASSWORD) |
| MLflow | http://localhost:5000 | — (no auth) |
| ChromaDB | http://localhost:8000/docs | — (Swagger UI) |
| Streamlit | http://localhost:8501 | — |

### 5. Enable a DAG

DAGs are paused by default (`AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true`).

In the Airflow UI:
1. Go to **DAGs** tab.
2. Toggle the switch next to the DAG you want to enable.
3. Trigger a manual run with the **▶** button to verify it works.

Or via CLI:
```bash
docker exec hull-airflow-scheduler airflow dags unpause dag_data_pipeline
docker exec hull-airflow-scheduler airflow dags trigger dag_data_pipeline
```

---

## DAG reference

### `dag_data_pipeline` — Hourly market data

| Task | What it does |
|------|-------------|
| `fetch_sp500_data` | Downloads 1 year of OHLCV for `^GSPC` via yfinance |
| `run_feature_engineering` | Computes RSI-14, MACD, Bollinger Bands, rolling vols, volume z-score |
| `save_to_parquet` | Writes `artifacts/data/market/sp500_features_latest.parquet` |

**Schedule:** `0 14-22 * * 1-5` (Mon–Fri, 14:00–22:00 UTC ≈ NYSE hours)

---

### `dag_rag_update` — Hourly news + ChromaDB

| Task | What it does |
|------|-------------|
| `fetch_financial_news` | Fetches articles from NewsAPI for 4 S&P 500 queries |
| `chunk_and_embed` | Splits articles into title/description/content chunks, embeds with `all-MiniLM-L6-v2` |
| `update_chromadb` | Upserts embeddings into ChromaDB collection `financial_news` |

**Schedule:** `30 14-22 * * 1-5` (offset by 30 min vs data pipeline)

**Airflow Variable required:** `NEWS_API_KEY` (set automatically by `airflow-init`)

---

### `dag_model_retraining` — Weekly retraining

| Task | What it does |
|------|-------------|
| `load_training_data` | Loads latest parquet + Kaggle train.csv (uses whichever is larger) |
| `run_preprocessing` | Drops high-NaN cols, median imputation, top-100 variance feature selection, 80/20 split |
| `train_models` | Trains LightGBM, XGBoost, CatBoost (with early stopping) + Ensemble average |
| `evaluate_models` | Computes RMSE/MAE/R²/Dir.Acc per model, writes `artifacts/metrics.json` |
| `promote_if_better` | Logs run to MLflow, registers model, promotes to Production if RMSE improved |

**Schedule:** `0 6 * * 1` (every Monday at 06:00 UTC)

After this DAG runs, the Streamlit dashboard auto-refreshes its metrics within 60 seconds.

---

## Airflow Variables

All sensitive keys are stored as **Airflow Variables** (not in DAG code).

| Variable | Set by | Used by |
|----------|--------|---------|
| `NEWS_API_KEY` | `airflow-init` (reads from `.env`) | `dag_rag_update` |
| `KAGGLE_USERNAME` | `airflow-init` | `dag_model_retraining` |
| `KAGGLE_KEY` | `airflow-init` | `dag_model_retraining` |
| `MLFLOW_TRACKING_URI` | `airflow-init` | `dag_model_retraining` |
| `CHROMADB_HOST` | `airflow-init` | `dag_rag_update` |
| `CHROMADB_PORT` | `airflow-init` | `dag_rag_update` |

To update a variable after init:
```bash
docker exec hull-airflow-webserver airflow variables set NEWS_API_KEY "new_key"
```

---

## Folder structure added

```
hull-tactical-market-prediction/
├── airflow/
│   ├── Dockerfile.airflow       # Extends apache/airflow:2.8.1 with ML deps
│   ├── init-mlflow-db.sql       # Creates 'mlflow' database in PostgreSQL
│   └── plugins/                 # Empty — add custom Airflow plugins here
├── dags/
│   ├── dag_data_pipeline.py     # DAG 1: hourly S&P 500 data + features
│   ├── dag_rag_update.py        # DAG 2: hourly news → ChromaDB
│   ├── dag_model_retraining.py  # DAG 3: weekly retrain → MLflow
│   └── utils/
│       ├── data_utils.py        # yfinance helpers, feature engineering, parquet I/O
│       └── mlflow_utils.py      # MLflow run logging, model registry helpers
├── logs/airflow/                # Airflow task logs (git-ignored)
├── docker-compose.airflow.yml   # Full stack definition
├── requirements-airflow.txt     # Extra pip packages for Airflow image
└── .env.example                 # Updated with Airflow/ChromaDB/NewsAPI vars
```

---

## Stop the stack

```bash
docker-compose -f docker-compose.airflow.yml down

# Remove all volumes (wipes DB, MLflow data, ChromaDB index):
docker-compose -f docker-compose.airflow.yml down -v
```

---

## Troubleshooting

### `docker compose` not found / unknown flag `-f`
Your Docker installation uses the **standalone binary** (not the CLI plugin). Always use the hyphenated form:
```bash
# Wrong (plugin syntax — not installed):
docker compose -f docker-compose.airflow.yml up -d

# Correct (standalone binary):
docker-compose -f docker-compose.airflow.yml up -d
```
Verify: `which docker-compose` → `/usr/local/bin/docker-compose`

### `airflow-init` exits with code 1
Check logs: `docker-compose -f docker-compose.airflow.yml logs airflow-init`
Most common cause: PostgreSQL not ready yet. The init container retries automatically.

### DAG shows "Import Error" in the UI
```bash
docker exec hull-airflow-scheduler airflow dags list-import-errors
```
Ensure the `PYTHONPATH` volume mount is correct in `docker-compose.airflow.yml`.

### ChromaDB connection refused
Verify the ChromaDB container is healthy:
```bash
docker-compose -f docker-compose.airflow.yml ps chromadb
curl http://localhost:8000/api/v1/heartbeat
```

### MLflow runs not appearing
The local MLflow uses PostgreSQL as backend. Check:
```bash
docker-compose -f docker-compose.airflow.yml logs mlflow
```
The `mlflow` database is created by `airflow/init-mlflow-db.sql` on first PostgreSQL start.
