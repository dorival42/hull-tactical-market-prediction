# Hull Tactical - Makefile
# Commands for development, testing, and deployment

.PHONY: help install install-dev lint format test train evaluate docker-build docker-run clean

# Default target
help:
	@echo "Hull Tactical Market Prediction - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  lint           Run linters (ruff, black check)"
	@echo "  format         Format code with black and isort"
	@echo "  typecheck      Run mypy type checking"
	@echo "  test           Run tests with pytest"
	@echo "  test-cov       Run tests with coverage report"
	@echo ""
	@echo "Training:"
	@echo "  train          Train all models"
	@echo "  train-lgb      Train LightGBM model"
	@echo "  train-xgb      Train XGBoost model"
	@echo "  train-cat      Train CatBoost model"
	@echo "  train-ensemble Train ensemble model"
	@echo "  evaluate       Evaluate trained models"
	@echo ""
	@echo "Application:"
	@echo "  streamlit      Run Streamlit app locally"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo "  docker-train   Run training in Docker"
	@echo ""
	@echo "Utility:"
	@echo "  clean          Clean up artifacts and cache"
	@echo "  download-data  Download data from Kaggle"

# ============================================================================
# Setup
# ============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

# ============================================================================
# Development
# ============================================================================

lint:
	ruff check src/ tests/ scripts/ app/
	black --check src/ tests/ scripts/ app/

format:
	black src/ tests/ scripts/ app/
	isort src/ tests/ scripts/ app/
	ruff check --fix src/ tests/ scripts/ app/

typecheck:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ============================================================================
# Training
# ============================================================================

train:
	python scripts/train.py --model all --n-features 150

train-lgb:
	python scripts/train.py --model lightgbm --n-features 150

train-xgb:
	python scripts/train.py --model xgboost --n-features 150

train-cat:
	python scripts/train.py --model catboost --n-features 150

train-ensemble:
	python scripts/train.py --model ensemble --n-features 150

train-validate:
	python scripts/train.py --model all --validate --n-splits 5

evaluate:
	python scripts/evaluate.py --artifacts-dir artifacts/models

# ============================================================================
# Application
# ============================================================================

streamlit:
	streamlit run app/streamlit_app.py --server.port 8501

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker build -t hull-tactical:latest -f docker/Dockerfile .

docker-build-train:
	docker build -t hull-tactical-train:latest -f docker/Dockerfile.train .

docker-run:
	docker run -p 8501:8501 \
		-e KAGGLE_USERNAME=${KAGGLE_USERNAME} \
		-e KAGGLE_KEY=${KAGGLE_KEY} \
		-e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
		-v $(PWD)/artifacts:/app/artifacts \
		hull-tactical:latest

docker-train:
	docker run \
		-e KAGGLE_USERNAME=${KAGGLE_USERNAME} \
		-e KAGGLE_KEY=${KAGGLE_KEY} \
		-e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
		-v $(PWD)/artifacts:/app/artifacts \
		hull-tactical-train:latest

docker-compose-up:
	docker-compose -f docker/docker-compose.yml up -d app

docker-compose-down:
	docker-compose -f docker/docker-compose.yml down

# ============================================================================
# Utility
# ============================================================================

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf *.egg-info build dist
	rm -rf artifacts/logs/*.log
	rm -rf catboost_info
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf artifacts/models/*.pkl
	rm -rf artifacts/data/*

download-data:
	python -c "from src.data.kaggle_loader import KaggleDataLoader; KaggleDataLoader().download_data()"

# ============================================================================
# MLflow
# ============================================================================

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# ============================================================================
# Pre-commit
# ============================================================================

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files
