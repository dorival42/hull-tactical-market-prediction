"""
===============================================================================
HULL TACTICAL - BASELINE MODELS
===============================================================================

Ce script implemente plusieurs strategies baseline :
1. Prediction constante (moyenne)
2. Momentum Simple
3. LightGBM Baseline
4. XGBoost Baseline

Objectif : Etablir une performance de reference (RMSE, MAE, R2)

Auteur: Baseline Hull Tactical
Date: 7 Novembre 2025
===============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HULL TACTICAL - BASELINE MODELS")
print("="*80)

# ===============================================================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ===============================================================================

print("\n" + "="*80)
print("1. CHARGEMENT ET PREPARATION DES DONNEES")
print("="*80)

train = pd.read_csv('train.csv')
print(f"\n✓ Donnees chargees: {train.shape[0]:,} lignes x {train.shape[1]} colonnes")

# Utiliser seulement les donnees recentes (moins de valeurs manquantes)
CUTOFF_DATE = 7000
train_clean = train[train['date_id'] >= CUTOFF_DATE].copy()
print(f"✓ Donnees apres cutoff (date_id >= {CUTOFF_DATE}): {train_clean.shape[0]:,} lignes")

# Verifier les valeurs manquantes
missing_pct = (train_clean.isnull().sum().sum() / (train_clean.shape[0] * train_clean.shape[1]) * 100)
print(f"✓ Valeurs manquantes: {missing_pct:.2f}%")

# Definir les features et la target
target_col = 'market_forward_excess_returns'
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', target_col]
feature_cols = [col for col in train_clean.columns if col not in exclude_cols]

print(f"\n CONFIGURATION:")
print(f"   Target: {target_col}")
print(f"   Features: {len(feature_cols)}")
print(f"   Periode: date_id {train_clean['date_id'].min()} a {train_clean['date_id'].max()}")

# ===============================================================================
# 2. SPLIT TEMPOREL (WALK-FORWARD)
# ===============================================================================

print("\n" + "="*80)
print("2. SPLIT TEMPOREL (WALK-FORWARD VALIDATION)")
print("="*80)

# Split: 80% train, 20% validation
split_idx = int(len(train_clean) * 0.8)
train_df = train_clean.iloc[:split_idx].copy()
val_df = train_clean.iloc[split_idx:].copy()

print(f"\n SPLIT DES DONNEES:")
print(f"   Train set: {len(train_df):,} lignes (date_id {train_df['date_id'].min()}-{train_df['date_id'].max()})")
print(f"   Val set:   {len(val_df):,} lignes (date_id {val_df['date_id'].min()}-{val_df['date_id'].max()})")

# Preparer X et y
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

print(f"\n✓ Features preparees: {X_train.shape[1]} colonnes")

# ===============================================================================
# 3. BASELINE 1: PREDICTION CONSTANTE (MOYENNE)
# ===============================================================================

print("\n" + "="*80)
print("3. BASELINE 1: PREDICTION CONSTANTE (MOYENNE)")
print("="*80)

# Predire la moyenne du train set
mean_pred = y_train.mean()
y_pred_mean = np.full(len(y_val), mean_pred)

rmse_mean = np.sqrt(mean_squared_error(y_val, y_pred_mean))
mae_mean = mean_absolute_error(y_val, y_pred_mean)
r2_mean = r2_score(y_val, y_pred_mean)

print(f"\n   Prediction constante = {mean_pred:.6f}")
print(f"      RMSE: {rmse_mean:.6f}")
print(f"      MAE:  {mae_mean:.6f}")
print(f"      R2:   {r2_mean:.4f}")

# ===============================================================================
# 4. BASELINE 2: MOMENTUM SIMPLE
# ===============================================================================

print("\n" + "="*80)
print("4. BASELINE 2: MOMENTUM SIMPLE")
print("="*80)

# Calculer le momentum sur differentes fenetres
windows = [5, 10, 20]
momentum_results = []

for window in windows:
    # Utiliser la moyenne glissante comme prediction
    val_df[f'momentum_{window}'] = val_df[target_col].rolling(window).mean()

    valid_idx = ~val_df[f'momentum_{window}'].isna()

    if valid_idx.sum() > 0:
        y_true_m = y_val[valid_idx]
        y_pred_m = val_df.loc[valid_idx, f'momentum_{window}']

        rmse_m = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
        mae_m = mean_absolute_error(y_true_m, y_pred_m)
        r2_m = r2_score(y_true_m, y_pred_m)

        momentum_results.append({
            'window': window,
            'rmse': rmse_m,
            'mae': mae_m,
            'r2': r2_m,
        })

        print(f"\n   Momentum Window = {window} jours:")
        print(f"      RMSE: {rmse_m:.6f}")
        print(f"      MAE:  {mae_m:.6f}")
        print(f"      R2:   {r2_m:.4f}")

best_momentum = min(momentum_results, key=lambda x: x['rmse'])
print(f"\n MEILLEUR MOMENTUM: Window = {best_momentum['window']} jours")
print(f"   RMSE: {best_momentum['rmse']:.6f}")

# ===============================================================================
# 5. BASELINE 3: LIGHTGBM
# ===============================================================================

print("\n" + "="*80)
print("5. BASELINE 3: LIGHTGBM")
print("="*80)

print("\n   Entrainement du modele LightGBM...")

# Parametres LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 20,
    'random_state': 42
}

# Entrainer le modele
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# Predictions
y_pred_lgb = lgb_model.predict(X_val)

# Metriques de prediction
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
mae_lgb = mean_absolute_error(y_val, y_pred_lgb)
r2_lgb = r2_score(y_val, y_pred_lgb)

print(f"\n   Metriques de prediction:")
print(f"      RMSE: {rmse_lgb:.6f}")
print(f"      MAE:  {mae_lgb:.6f}")
print(f"      R2:   {r2_lgb:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 10 features importantes:")
for i, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']:15s}: {row['importance']:8.2f}")

# ===============================================================================
# 6. BASELINE 4: XGBOOST
# ===============================================================================

print("\n" + "="*80)
print("6. BASELINE 4: XGBOOST")
print("="*80)

print("\n   Entrainement du modele XGBoost...")

# Parametres XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0
}

# Entrainer le modele
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predictions
y_pred_xgb = xgb_model.predict(X_val)

# Metriques de prediction
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"\n   Metriques de prediction:")
print(f"      RMSE: {rmse_xgb:.6f}")
print(f"      MAE:  {mae_xgb:.6f}")
print(f"      R2:   {r2_xgb:.4f}")

# ===============================================================================
# 7. COMPARAISON FINALE
# ===============================================================================

print("\n" + "="*80)
print("7. COMPARAISON FINALE DES BASELINES")
print("="*80)

# Compiler tous les resultats
all_results = [
    {'strategy': 'Prediction Moyenne', 'rmse': rmse_mean, 'mae': mae_mean, 'r2': r2_mean},
    {'strategy': f'Momentum ({best_momentum["window"]}d)', 'rmse': best_momentum['rmse'], 'mae': best_momentum['mae'], 'r2': best_momentum['r2']},
    {'strategy': 'LightGBM', 'rmse': rmse_lgb, 'mae': mae_lgb, 'r2': r2_lgb},
    {'strategy': 'XGBoost', 'rmse': rmse_xgb, 'mae': mae_xgb, 'r2': r2_xgb},
]

# Trier par RMSE
all_results_sorted = sorted(all_results, key=lambda x: x['rmse'])

print("\n CLASSEMENT PAR RMSE:\n")
print(f"{'Rang':<5} {'Strategie':<25} {'RMSE':<12} {'MAE':<12} {'R2':<10}")
print("-" * 70)

for i, result in enumerate(all_results_sorted, 1):
    print(f"{i:<5} {result['strategy']:<25} {result['rmse']:<12.6f} {result['mae']:<12.6f} {result['r2']:<10.4f}")

# Meilleure strategie globale
best_overall = all_results_sorted[0]
print(f"\n{'='*80}")
print(f"MEILLEURE STRATEGIE BASELINE: {best_overall['strategy']}")
print(f"{'='*80}")
print(f"   RMSE: {best_overall['rmse']:.6f}")
print(f"   MAE:  {best_overall['mae']:.6f}")
print(f"   R2:   {best_overall['r2']:.4f}")

# Sauvegarder les modeles
import pickle

print("\n Sauvegarde des modeles...")
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
print("   ✓ LightGBM sauvegarde: lgb_model.pkl")

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("   ✓ XGBoost sauvegarde: xgb_model.pkl")

# Sauvegarder les resultats
results_df = pd.DataFrame(all_results_sorted)
results_df.to_csv('baseline_results.csv', index=False)
print("   ✓ Resultats sauvegardes: baseline_results.csv")

print("\n✓ Tous les fichiers sauvegardes!")
print("\n PROCHAINE ETAPE: Tester avec l'API Kaggle locale")
