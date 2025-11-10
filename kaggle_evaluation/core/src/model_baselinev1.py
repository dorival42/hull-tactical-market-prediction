"""
═══════════════════════════════════════════════════════════════════════════════
HULL TACTICAL - BASELINE MODELS
═══════════════════════════════════════════════════════════════════════════════

Ce script implémente plusieurs stratégies baseline :
1. Buy & Hold (allocation constante)
2. Momentum Simple
3. Volatility-based
4. LightGBM Baseline
5. XGBoost Baseline

Objectif : Établir une performance de référence et calculer le Sharpe ratio

Auteur: Baseline Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HULL TACTICAL - BASELINE MODELS")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("1. CHARGEMENT ET PRÉPARATION DES DONNÉES")
print("="*80)

train = pd.read_csv('train.csv')
print(f"\n✓ Données chargées: {train.shape[0]:,} lignes × {train.shape[1]} colonnes")

# Utiliser seulement les données récentes (moins de valeurs manquantes)
# Basé sur l'EDA : date_id > 7000 a 0% de valeurs manquantes
CUTOFF_DATE = 7000
train_clean = train[train['date_id'] >= CUTOFF_DATE].copy()
print(f"✓ Données après cutoff (date_id >= {CUTOFF_DATE}): {train_clean.shape[0]:,} lignes")

# Vérifier les valeurs manquantes
missing_pct = (train_clean.isnull().sum().sum() / (train_clean.shape[0] * train_clean.shape[1]) * 100)
print(f"✓ Valeurs manquantes: {missing_pct:.2f}%")

# Définir les features et la target
target_col = 'market_forward_excess_returns'
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', target_col]
feature_cols = [col for col in train_clean.columns if col not in exclude_cols]

print(f"\n📊 CONFIGURATION:")
print(f"   Target: {target_col}")
print(f"   Features: {len(feature_cols)}")
print(f"   Période: date_id {train_clean['date_id'].min()} à {train_clean['date_id'].max()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL (WALK-FORWARD)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("2. SPLIT TEMPOREL (WALK-FORWARD VALIDATION)")
print("="*80)

# Split: 80% train, 20% validation
split_idx = int(len(train_clean) * 0.8)
train_df = train_clean.iloc[:split_idx].copy()
val_df = train_clean.iloc[split_idx:].copy()

print(f"\n📊 SPLIT DES DONNÉES:")
print(f"   Train set: {len(train_df):,} lignes (date_id {train_df['date_id'].min()}-{train_df['date_id'].max()})")
print(f"   Val set:   {len(val_df):,} lignes (date_id {val_df['date_id'].min()}-{val_df['date_id'].max()})")

# Préparer X et y
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

print(f"\n✓ Features préparées: {X_train.shape[1]} colonnes")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FONCTION DE CALCUL DU SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sharpe_ratio(allocations, returns, risk_free_rates, max_volatility_ratio=1.2):
    """
    Calcule le Sharpe ratio avec contrainte de volatilité.
    
    Args:
        allocations: Array d'allocations (0-2)
        returns: Array de forward_returns
        risk_free_rates: Array de risk_free_rate
        max_volatility_ratio: Ratio maximal de volatilité autorisé (défaut 1.2)
    
    Returns:
        dict avec sharpe_ratio, annualized_return, volatility, etc.
    """
    # Rendements du portefeuille
    portfolio_returns = allocations * returns
    
    # Rendements excédentaires
    excess_returns = portfolio_returns - risk_free_rates
    
    # Calculs
    mean_excess_return = excess_returns.mean()
    volatility = excess_returns.std()
    
    # Sharpe ratio (annualisé, 252 jours de trading par an)
    sharpe_ratio = (mean_excess_return / volatility) * np.sqrt(252) if volatility > 0 else 0
    
    # Rendement annualisé
    annualized_return = mean_excess_return * 252
    annualized_volatility = volatility * np.sqrt(252)
    
    # Volatilité du marché (pour la contrainte)
    market_volatility = returns.std()
    volatility_ratio = volatility / market_volatility if market_volatility > 0 else 0
    
    # Pénalité si on dépasse la contrainte de volatilité
    exceeds_volatility_constraint = volatility_ratio > max_volatility_ratio
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'volatility_ratio': volatility_ratio,
        'exceeds_constraint': exceeds_volatility_constraint,
        'mean_allocation': allocations.mean(),
        'std_allocation': allocations.std()
    }

def convert_prediction_to_allocation(predictions, method='simple', threshold=0.003):
    """
    Convertit les prédictions en allocations (0-2).
    
    Args:
        predictions: Array de prédictions de rendements
        method: 'simple', 'proportional', 'threshold'
        threshold: Seuil pour la méthode 'threshold'
    
    Returns:
        Array d'allocations
    """
    if method == 'simple':
        # Simple: si positif -> 1.5, si négatif -> 0.5
        allocations = np.where(predictions > 0, 1.5, 0.5)
    
    elif method == 'proportional':
        # Proportionnel: allocation basée sur la magnitude de la prédiction
        # Normaliser entre 0 et 2
        min_pred = predictions.min()
        max_pred = predictions.max()
        if max_pred > min_pred:
            allocations = 0.2 + 1.6 * (predictions - min_pred) / (max_pred - min_pred)
        else:
            allocations = np.full_like(predictions, 1.0)
    
    elif method == 'threshold':
        # Basé sur des seuils
        allocations = np.ones_like(predictions)  # Défaut: 1.0
        allocations[predictions > threshold] = 1.8
        allocations[(predictions > 0) & (predictions <= threshold)] = 1.3
        allocations[(predictions < 0) & (predictions >= -threshold)] = 0.7
        allocations[predictions < -threshold] = 0.2
    
    # S'assurer que les allocations sont dans [0, 2]
    allocations = np.clip(allocations, 0.0, 2.0)
    
    return allocations

# ═══════════════════════════════════════════════════════════════════════════════
# 4. BASELINE 1: BUY & HOLD (ALLOCATION CONSTANTE)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("4. BASELINE 1: BUY & HOLD (ALLOCATION CONSTANTE)")
print("="*80)

# Tester différentes allocations constantes
constant_allocations = [0.5, 1.0, 1.5, 2.0]
buy_hold_results = []

for allocation in constant_allocations:
    allocations = np.full(len(val_df), allocation)
    metrics = calculate_sharpe_ratio(
        allocations,
        val_df['forward_returns'].values,
        val_df['risk_free_rate'].values
    )
    buy_hold_results.append({
        'allocation': allocation,
        **metrics
    })
    
    print(f"\n   Allocation constante = {allocation:.1f}:")
    print(f"      Sharpe Ratio:      {metrics['sharpe_ratio']:+.4f}")
    print(f"      Annual Return:     {metrics['annualized_return']*100:+.2f}%")
    print(f"      Annual Volatility: {metrics['annualized_volatility']*100:.2f}%")
    print(f"      Volatility Ratio:  {metrics['volatility_ratio']:.2f}x")
    print(f"      Constraint OK:     {'NON ❌' if metrics['exceeds_constraint'] else 'OUI ✓'}")

# Meilleure allocation constante
best_buy_hold = max(buy_hold_results, key=lambda x: x['sharpe_ratio'])
print(f"\n🏆 MEILLEURE ALLOCATION CONSTANTE: {best_buy_hold['allocation']:.1f}")
print(f"   Sharpe Ratio: {best_buy_hold['sharpe_ratio']:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BASELINE 2: MOMENTUM SIMPLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("5. BASELINE 2: MOMENTUM SIMPLE")
print("="*80)

# Calculer le momentum sur différentes fenêtres
windows = [5, 10, 20]
momentum_results = []

for window in windows:
    # Calculer le momentum
    val_df[f'momentum_{window}'] = val_df[target_col].rolling(window).mean()
    
    # Stratégie: si momentum > 0 -> 1.5, sinon -> 0.5
    allocations = np.where(val_df[f'momentum_{window}'].values > 0, 1.5, 0.5)
    
    # Gérer les NaN au début
    valid_idx = ~np.isnan(allocations)
    
    if valid_idx.sum() > 0:
        metrics = calculate_sharpe_ratio(
            allocations[valid_idx],
            val_df['forward_returns'].values[valid_idx],
            val_df['risk_free_rate'].values[valid_idx]
        )
        momentum_results.append({
            'window': window,
            **metrics
        })
        
        print(f"\n   Momentum Window = {window} jours:")
        print(f"      Sharpe Ratio:      {metrics['sharpe_ratio']:+.4f}")
        print(f"      Annual Return:     {metrics['annualized_return']*100:+.2f}%")
        print(f"      Annual Volatility: {metrics['annualized_volatility']*100:.2f}%")
        print(f"      Volatility Ratio:  {metrics['volatility_ratio']:.2f}x")
        print(f"      Constraint OK:     {'NON ❌' if metrics['exceeds_constraint'] else 'OUI ✓'}")

# Meilleur momentum
best_momentum = max(momentum_results, key=lambda x: x['sharpe_ratio'])
print(f"\n🏆 MEILLEUR MOMENTUM: Window = {best_momentum['window']} jours")
print(f"   Sharpe Ratio: {best_momentum['sharpe_ratio']:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. BASELINE 3: VOLATILITY-BASED
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("6. BASELINE 3: VOLATILITY-BASED")
print("="*80)

# Stratégie: réduire l'allocation en période de haute volatilité
window = 20
val_df['rolling_vol'] = val_df['forward_returns'].rolling(window).std()

# Normaliser la volatilité
vol_mean = val_df['rolling_vol'].mean()
vol_std = val_df['rolling_vol'].std()

# Allocation inversement proportionnelle à la volatilité
# Haute vol -> faible allocation, faible vol -> haute allocation
normalized_vol = (val_df['rolling_vol'] - vol_mean) / vol_std
allocations = 1.5 - 0.5 * normalized_vol  # Range: environ 0.5 à 2.0
allocations = np.clip(allocations, 0.2, 2.0)

# Gérer les NaN
valid_idx = ~np.isnan(allocations)
metrics = calculate_sharpe_ratio(
    allocations[valid_idx],
    val_df['forward_returns'].values[valid_idx],
    val_df['risk_free_rate'].values[valid_idx]
)

print(f"\n   Volatility-Based (Window = {window} jours):")
print(f"      Sharpe Ratio:      {metrics['sharpe_ratio']:+.4f}")
print(f"      Annual Return:     {metrics['annualized_return']*100:+.2f}%")
print(f"      Annual Volatility: {metrics['annualized_volatility']*100:.2f}%")
print(f"      Volatility Ratio:  {metrics['volatility_ratio']:.2f}x")
print(f"      Mean Allocation:   {metrics['mean_allocation']:.2f}")
print(f"      Constraint OK:     {'NON ❌' if metrics['exceeds_constraint'] else 'OUI ✓'}")

volatility_sharpe = metrics['sharpe_ratio']

# ═══════════════════════════════════════════════════════════════════════════════
# 7. BASELINE 4: LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("7. BASELINE 4: LIGHTGBM")
print("="*80)

print("\n   Entraînement du modèle LightGBM...")

# Paramètres LightGBM
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

# Entraîner le modèle
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# Prédictions
y_pred_lgb = lgb_model.predict(X_val)

# Métriques de prédiction
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
mae_lgb = mean_absolute_error(y_val, y_pred_lgb)
r2_lgb = r2_score(y_val, y_pred_lgb)

print(f"\n   Métriques de prédiction:")
print(f"      RMSE: {rmse_lgb:.6f}")
print(f"      MAE:  {mae_lgb:.6f}")
print(f"      R²:   {r2_lgb:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 10 features importantes:")
for i, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']:15s}: {row['importance']:8.2f}")

# Convertir en allocations (tester différentes méthodes)
allocation_methods = ['simple', 'proportional', 'threshold']
lgb_results = []

for method in allocation_methods:
    allocations = convert_prediction_to_allocation(y_pred_lgb, method=method)
    metrics = calculate_sharpe_ratio(
        allocations,
        val_df['forward_returns'].values,
        val_df['risk_free_rate'].values
    )
    lgb_results.append({
        'method': method,
        **metrics
    })
    
    print(f"\n   Méthode d'allocation: '{method}'")
    print(f"      Sharpe Ratio:      {metrics['sharpe_ratio']:+.4f}")
    print(f"      Annual Return:     {metrics['annualized_return']*100:+.2f}%")
    print(f"      Annual Volatility: {metrics['annualized_volatility']*100:.2f}%")
    print(f"      Mean Allocation:   {metrics['mean_allocation']:.2f}")
    print(f"      Std Allocation:    {metrics['std_allocation']:.2f}")
    print(f"      Constraint OK:     {'NON ❌' if metrics['exceeds_constraint'] else 'OUI ✓'}")

# Meilleure méthode LightGBM
best_lgb = max(lgb_results, key=lambda x: x['sharpe_ratio'])
print(f"\n🏆 MEILLEURE MÉTHODE LIGHTGBM: '{best_lgb['method']}'")
print(f"   Sharpe Ratio: {best_lgb['sharpe_ratio']:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. BASELINE 5: XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("8. BASELINE 5: XGBOOST")
print("="*80)

print("\n   Entraînement du modèle XGBoost...")

# Paramètres XGBoost
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

# Entraîner le modèle
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Prédictions
y_pred_xgb = xgb_model.predict(X_val)

# Métriques de prédiction
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"\n   Métriques de prédiction:")
print(f"      RMSE: {rmse_xgb:.6f}")
print(f"      MAE:  {mae_xgb:.6f}")
print(f"      R²:   {r2_xgb:.4f}")

# Convertir en allocations
xgb_results = []

for method in allocation_methods:
    allocations = convert_prediction_to_allocation(y_pred_xgb, method=method)
    metrics = calculate_sharpe_ratio(
        allocations,
        val_df['forward_returns'].values,
        val_df['risk_free_rate'].values
    )
    xgb_results.append({
        'method': method,
        **metrics
    })
    
    print(f"\n   Méthode d'allocation: '{method}'")
    print(f"      Sharpe Ratio:      {metrics['sharpe_ratio']:+.4f}")
    print(f"      Annual Return:     {metrics['annualized_return']*100:+.2f}%")
    print(f"      Annual Volatility: {metrics['annualized_volatility']*100:.2f}%")
    print(f"      Mean Allocation:   {metrics['mean_allocation']:.2f}")
    print(f"      Constraint OK:     {'NON ❌' if metrics['exceeds_constraint'] else 'OUI ✓'}")

# Meilleure méthode XGBoost
best_xgb = max(xgb_results, key=lambda x: x['sharpe_ratio'])
print(f"\n🏆 MEILLEURE MÉTHODE XGBOOST: '{best_xgb['method']}'")
print(f"   Sharpe Ratio: {best_xgb['sharpe_ratio']:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMPARAISON FINALE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("9. COMPARAISON FINALE DES BASELINES")
print("="*80)

# Compiler tous les résultats
all_results = [
    {'strategy': f'Buy & Hold ({best_buy_hold["allocation"]:.1f})', **best_buy_hold},
    {'strategy': f'Momentum ({best_momentum["window"]}d)', **best_momentum},
    {'strategy': 'Volatility-Based', 'sharpe_ratio': volatility_sharpe, **metrics},
    {'strategy': f'LightGBM ({best_lgb["method"]})', **best_lgb},
    {'strategy': f'XGBoost ({best_xgb["method"]})', **best_xgb}
]

# Trier par Sharpe Ratio
all_results_sorted = sorted(all_results, key=lambda x: x['sharpe_ratio'], reverse=True)

print("\n📊 CLASSEMENT PAR SHARPE RATIO:\n")
print(f"{'Rang':<5} {'Stratégie':<30} {'Sharpe':<10} {'Return':<10} {'Volatility':<12} {'Constraint':<10}")
print("-" * 80)

for i, result in enumerate(all_results_sorted, 1):
    strategy = result['strategy']
    sharpe = result['sharpe_ratio']
    ret = result['annualized_return'] * 100
    vol = result['annualized_volatility'] * 100
    constraint = '✓ OK' if not result['exceeds_constraint'] else '❌ NON'
    
    print(f"{i:<5} {strategy:<30} {sharpe:+.4f}    {ret:+6.2f}%    {vol:6.2f}%      {constraint:<10}")

# Meilleure stratégie globale
best_overall = all_results_sorted[0]
print(f"\n{'='*80}")
print(f"🏆 MEILLEURE STRATÉGIE BASELINE: {best_overall['strategy']}")
print(f"{'='*80}")
print(f"   Sharpe Ratio:        {best_overall['sharpe_ratio']:+.4f}")
print(f"   Rendement annualisé: {best_overall['annualized_return']*100:+.2f}%")
print(f"   Volatilité annualisé: {best_overall['annualized_volatility']*100:.2f}%")
print(f"   Ratio de volatilité:  {best_overall['volatility_ratio']:.2f}x")
print(f"   Respecte contrainte:  {'OUI ✓' if not best_overall['exceeds_constraint'] else 'NON ❌'}")

# Benchmark du marché (allocation = 1.0)
market_benchmark = [r for r in all_results if 'Buy & Hold (1.0)' in r['strategy']]
if market_benchmark:
    market_sharpe = market_benchmark[0]['sharpe_ratio']
    improvement = ((best_overall['sharpe_ratio'] - market_sharpe) / abs(market_sharpe) * 100) if market_sharpe != 0 else 0
    print(f"\n📈 AMÉLIORATION VS MARCHÉ (Buy & Hold 1.0):")
    print(f"   Market Sharpe:  {market_sharpe:+.4f}")
    print(f"   Best Sharpe:    {best_overall['sharpe_ratio']:+.4f}")
    print(f"   Amélioration:   {improvement:+.2f}%")

print("\n" + "="*80)
print("ANALYSE BASELINE TERMINÉE")
print("="*80)

# Sauvegarder les modèles
import pickle

print("\n💾 Sauvegarde des modèles...")
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
print("   ✓ LightGBM sauvegardé: lgb_model.pkl")

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("   ✓ XGBoost sauvegardé: xgb_model.pkl")

# Sauvegarder les résultats
results_df = pd.DataFrame(all_results_sorted)
results_df.to_csv('baseline_results.csv', index=False)
print("   ✓ Résultats sauvegardés: baseline_results.csv")

print("\n✓ Tous les fichiers sauvegardés!")
print("\n🎯 PROCHAINE ÉTAPE: Tester avec l'API Kaggle locale")