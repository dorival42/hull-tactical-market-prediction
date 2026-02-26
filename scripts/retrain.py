#!/usr/bin/env python3
"""
Retrain all Hull Tactical models using parameters from
artifacts/preprocessing_config.json (saved from the Streamlit UI).

Usage:
    python scripts/retrain.py [--models lgbm xgb cat rf] [--no-ensemble]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Load .env (credentials for Kaggle / MLflow / DagsHub)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── Config path ───────────────────────────────────────────────────────────────
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_PATH   = ARTIFACTS_DIR / "preprocessing_config.json"

_DEFAULTS: dict = {
    "lower_pct":         0.0,
    "upper_pct":         100.0,
    "nan_threshold_pct": 30,
    "imputation_method": "Median",
    "n_features":        100,
}

# Imputation methods supported natively by DataPreprocessor
# Others fall back to Médiane (median) in the trainer
_CATBOOST_METHODS = {"CatBoost"}


def load_config() -> dict:
    cfg = _DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"⚠️  Could not read preprocessing config: {e}. Using defaults.")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain Hull Tactical models")
    p.add_argument(
        "--models", nargs="+",
        default=["lightgbm", "xgboost", "catboost", "random_forest"],
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
        help="Models to train (default: all four)",
    )
    p.add_argument("--no-ensemble", action="store_true",
                   help="Skip ensemble training after individual models")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg  = load_config()

    # ── Extract preprocessing parameters ─────────────────────────────────────
    nan_threshold        = cfg["nan_threshold_pct"] / 100.0
    n_features           = int(cfg["n_features"])
    lower_pct            = float(cfg["lower_pct"])
    upper_pct            = float(cfg["upper_pct"])
    imputation_method    = cfg["imputation_method"]
    use_catboost_imp     = imputation_method in _CATBOOST_METHODS

    print("=" * 65)
    print("  HULL TACTICAL — RETRAIN WITH PREPROCESSING CONFIG")
    print("=" * 65)
    print(f"  NaN threshold      : {nan_threshold*100:.0f}%")
    print(f"  N features target  : {n_features}")
    print(f"  Row range          : {lower_pct:.1f}% → {upper_pct:.1f}%")
    print(f"  Imputation         : {imputation_method}"
          + ("" if use_catboost_imp else " → mapped to Médiane (trainer)"))
    print(f"  Models             : {args.models}")
    print(f"  Ensemble           : {'no' if args.no_ensemble else 'yes'}")
    if cfg.get("saved_at"):
        print(f"  Config saved at    : {cfg['saved_at']}")
    print("=" * 65 + "\n")

    # ── Import trainer (after sys.path is set) ────────────────────────────────
    from src.training.trainer import MLflowTrainer  # noqa: PLC0415

    trainer = MLflowTrainer(
        nan_threshold=nan_threshold,
        use_catboost_imputation=use_catboost_imp,
        n_features=n_features,
    )

    # ── Load & prepare data with row-range cutoff ─────────────────────────────
    trainer.load_and_prepare_data(
        start_row_pct=lower_pct if lower_pct > 0.0   else None,
        end_row_pct=upper_pct   if upper_pct < 100.0  else None,
    )

    # ── Train individual models ───────────────────────────────────────────────
    results = trainer.train_all_models(
        model_types=args.models,
        output_dir=str(ARTIFACTS_DIR),
    )

    # ── Train ensemble ────────────────────────────────────────────────────────
    if not args.no_ensemble:
        print("\n" + "=" * 65)
        print("  TRAINING ENSEMBLE")
        print("=" * 65)
        try:
            ensemble, ens_metrics = trainer.train_ensemble()
            trainer.save_artifacts(ensemble, str(ARTIFACTS_DIR))
            ens_rmse = ens_metrics.get("rmse")
            print(f"  Ensemble RMSE : {ens_rmse:.6f}" if ens_rmse else "  Ensemble RMSE : N/A")
        except Exception as exc:
            print(f"  ⚠️  Ensemble training failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RETRAIN COMPLETE — results written to artifacts/")
    print("=" * 65)

    trained = {k: v for k, v in results.items() if v}
    for model_name, (_, metrics) in trained.items():
        rmse = metrics.get("rmse")
        r2   = metrics.get("r2")
        dacc = metrics.get("directional_accuracy")
        print(
            f"  {model_name:<14} RMSE={rmse:.6f}  R²={r2:+.4f}  DirAcc={dacc:.2f}%"
            if rmse else f"  {model_name:<14} — metrics unavailable"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
