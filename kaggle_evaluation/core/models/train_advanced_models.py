"""
═══════════════════════════════════════════════════════════════════════════════
ADVANCED MODEL TRAINING - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

Script principal pour:
1. Feature Engineering avancé
2. Feature Selection
3. Entraînement de multiples modèles
4. Walk-Forward Validation
5. Hyperparameter Tuning
6. Comparaison des performances

Auteur: Advanced Modeling Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
sys.path.append('/home/claude')

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from models.base_model import ModelMetrics
from models.feature_engineer import FeatureEngineer
from models.gradient_boosting_models import (
    LightGBMModel, XGBoostModel, RandomForestModel
)

print("="*80)
print("ADVANCED MODEL TRAINING - HULL TACTICAL")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("1. CHARGEMENT ET FEATURE ENGINEERING")
print("="*80)

# Charger les données
train = pd.read_csv('/home/claude/train.csv')
print(f"\n✓ Données chargées: {train.shape[0]:,} lignes × {train.shape[1]} colonnes")

# Utiliser données récentes (moins de valeurs manquantes)
CUTOFF_DATE = 7000
train_clean = train[train['date_id'] >= CUTOFF_DATE].copy()
print(f"✓ Après cutoff (date_id >= {CUTOFF_DATE}): {train_clean.shape[0]:,} lignes")

# Feature Engineering
fe = FeatureEngineer(verbose=True)
train_enhanced = fe.create_all_features(train_clean)

# Supprimer les lignes avec trop de NaN (dues aux rolling features)
train_enhanced = train_enhanced.dropna(subset=['market_forward_excess_returns'])
train_enhanced = train_enhanced.iloc[100:].copy()  # Skip first 100 rows (warm-up period)
print(f"\n✓ Après suppression des NaN: {train_enhanced.shape[0]:,} lignes")

# Feature Selection
print("\n" + "-"*80)
print("FEATURE SELECTION")
print("-"*80)

target_col = 'market_forward_excess_returns'
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', target_col]

# Méthode 1: Corrélation
selected_corr = fe.select_features(train_enhanced, target_col, 
                                   method='correlation', n_features=50)

# Méthode 2: Mutual Information
selected_mi = fe.select_features(train_enhanced, target_col,
                                method='mutual_info', n_features=50)

# Combiner les deux méthodes
selected_features = list(set(selected_corr[:30]) | set(selected_mi[:30]))
print(f"\n✓ {len(selected_features)} features sélectionnées au total")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("2. WALK-FORWARD VALIDATION")
print("="*80)

def walk_forward_validation(df: pd.DataFrame, 
                            feature_cols: List[str],
                            target_col: str,
                            n_splits: int = 5) -> Dict:
    """
    Walk-Forward Validation.
    
    Args:
        df: DataFrame
        feature_cols: Colonnes de features
        target_col: Colonne target
        n_splits: Nombre de splits
        
    Returns:
        Dictionnaire de résultats
    """
    print(f"\n   Walk-Forward avec {n_splits} splits")
    print("   " + "-"*60)
    
    # Calculer les tailles des splits
    total_size = len(df)
    test_size = total_size // (n_splits + 1)
    
    results = {
        'lightgbm': {'predictions': [], 'actuals': [], 'metrics': []},
        'xgboost': {'predictions': [], 'actuals': [], 'metrics': []},
        'random_forest': {'predictions': [], 'actuals': [], 'metrics': []}
    }
    
    for fold in range(n_splits):
        print(f"\n   Fold {fold + 1}/{n_splits}")
        
        # Split temporel
        train_end = (fold + 1) * test_size + (total_size - (n_splits + 1) * test_size)
        test_start = train_end
        test_end = train_end + test_size
        
        if test_end > total_size:
            test_end = total_size
        
        train_fold = df.iloc[:train_end]
        test_fold = df.iloc[test_start:test_end]
        
        print(f"      Train: {len(train_fold)} lignes | Test: {len(test_fold)} lignes")
        
        # Préparer les données
        X_train = train_fold[feature_cols].fillna(0)
        y_train = train_fold[target_col]
        X_test = test_fold[feature_cols].fillna(0)
        y_test = test_fold[target_col]
        
        # LightGBM
        lgb_model = LightGBMModel()
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        
        results['lightgbm']['predictions'].extend(y_pred_lgb)
        results['lightgbm']['actuals'].extend(y_test.values)
        
        metrics_lgb = ModelMetrics.calculate_regression_metrics(y_test.values, y_pred_lgb)
        results['lightgbm']['metrics'].append(metrics_lgb)
        print(f"      LightGBM - RMSE: {metrics_lgb['rmse']:.6f}, R²: {metrics_lgb['r2']:.4f}")
        
        # XGBoost
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        
        results['xgboost']['predictions'].extend(y_pred_xgb)
        results['xgboost']['actuals'].extend(y_test.values)
        
        metrics_xgb = ModelMetrics.calculate_regression_metrics(y_test.values, y_pred_xgb)
        results['xgboost']['metrics'].append(metrics_xgb)
        print(f"      XGBoost  - RMSE: {metrics_xgb['rmse']:.6f}, R²: {metrics_xgb['r2']:.4f}")
        
        # Random Forest
        rf_model = RandomForestModel()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        results['random_forest']['predictions'].extend(y_pred_rf)
        results['random_forest']['actuals'].extend(y_test.values)
        
        metrics_rf = ModelMetrics.calculate_regression_metrics(y_test.values, y_pred_rf)
        results['random_forest']['metrics'].append(metrics_rf)
        print(f"      RF       - RMSE: {metrics_rf['rmse']:.6f}, R²: {metrics_rf['r2']:.4f}")
    
    return results

# Exécuter la validation
wf_results = walk_forward_validation(train_enhanced, selected_features, target_col, n_splits=5)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RÉSULTATS GLOBAUX
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("3. RÉSULTATS GLOBAUX DE LA WALK-FORWARD VALIDATION")
print("="*80)

summary = []

for model_name, data in wf_results.items():
    # Calculer les métriques globales
    y_true = np.array(data['actuals'])
    y_pred = np.array(data['predictions'])
    
    global_metrics = ModelMetrics.calculate_regression_metrics(y_true, y_pred)
    
    # Moyenne des métriques par fold
    avg_rmse = np.mean([m['rmse'] for m in data['metrics']])
    avg_r2 = np.mean([m['r2'] for m in data['metrics']])
    std_rmse = np.std([m['rmse'] for m in data['metrics']])
    std_r2 = np.std([m['r2'] for m in data['metrics']])
    
    summary.append({
        'model': model_name,
        'global_rmse': global_metrics['rmse'],
        'global_r2': global_metrics['r2'],
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'avg_r2': avg_r2,
        'std_r2': std_r2
    })

summary_df = pd.DataFrame(summary).sort_values('global_rmse')

print("\n" + "-"*80)
print("CLASSEMENT PAR RMSE GLOBAL")
print("-"*80)
print(f"\n{'Modèle':<20} {'RMSE Global':<15} {'R² Global':<15} {'RMSE Avg±Std':<20}")
print("-"*80)

for _, row in summary_df.iterrows():
    print(f"{row['model']:<20} {row['global_rmse']:<15.6f} {row['global_r2']:<15.4f} {row['avg_rmse']:.6f}±{row['std_rmse']:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT FINAL SUR TOUTES LES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("4. ENTRAÎNEMENT FINAL DES MEILLEURS MODÈLES")
print("="*80)

# Split final (80% train, 20% validation)
split_idx = int(len(train_enhanced) * 0.8)
train_final = train_enhanced.iloc[:split_idx]
val_final = train_enhanced.iloc[split_idx:]

X_train_final = train_final[selected_features].fillna(0)
y_train_final = train_final[target_col]
X_val_final = val_final[selected_features].fillna(0)
y_val_final = val_final[target_col]

print(f"\n   Train final: {len(train_final):,} lignes")
print(f"   Val final:   {len(val_final):,} lignes")

# Entraîner les 3 meilleurs modèles
final_models = {}

print("\n   Entraînement LightGBM...")
lgb_final = LightGBMModel(name='LightGBM_Final')
lgb_final.fit(X_train_final, y_train_final, X_val_final, y_val_final)
final_models['lightgbm'] = lgb_final

print("   Entraînement XGBoost...")
xgb_final = XGBoostModel(name='XGBoost_Final')
xgb_final.fit(X_train_final, y_train_final, X_val_final, y_val_final)
final_models['xgboost'] = xgb_final

print("   Entraînement Random Forest...")
rf_final = RandomForestModel(name='RF_Final')
rf_final.fit(X_train_final, y_train_final, X_val_final, y_val_final)
final_models['random_forest'] = rf_final

# Résultats sur validation finale
print("\n" + "-"*80)
print("PERFORMANCE SUR VALIDATION FINALE")
print("-"*80)

for name, model in final_models.items():
    metrics = model.get_metrics()
    print(f"\n   {name.upper()}:")
    print(f"      RMSE: {metrics['rmse']:.6f}")
    print(f"      MAE:  {metrics['mae']:.6f}")
    print(f"      R²:   {metrics['r2']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CALCUL DU SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("5. CALCUL DU SHARPE RATIO")
print("="*80)

def convert_to_allocation(predictions, method='threshold'):
    """Convertir prédictions en allocations."""
    if method == 'threshold':
        allocations = np.ones_like(predictions)
        allocations[predictions > 0.003] = 1.8
        allocations[(predictions > 0) & (predictions <= 0.003)] = 1.3
        allocations[(predictions < 0) & (predictions >= -0.003)] = 0.7
        allocations[predictions < -0.003] = 0.2
    return np.clip(allocations, 0.0, 2.0)

# Calcul pour chaque modèle
sharpe_results = []

for name, model in final_models.items():
    y_pred = model.predict(X_val_final)
    allocations = convert_to_allocation(y_pred)
    
    sharpe_metrics = ModelMetrics.calculate_sharpe_ratio(
        allocations,
        val_final['forward_returns'].values,
        val_final['risk_free_rate'].values
    )
    
    sharpe_metrics['model'] = name
    sharpe_results.append(sharpe_metrics)
    
    print(f"\n   {name.upper()}:")
    print(f"      Sharpe Ratio:      {sharpe_metrics['sharpe_ratio']:+.4f}")
    print(f"      Annual Return:     {sharpe_metrics['annualized_return']*100:+.2f}%")
    print(f"      Annual Volatility: {sharpe_metrics['annualized_volatility']*100:.2f}%")
    print(f"      Volatility Ratio:  {sharpe_metrics['volatility_ratio']:.2f}x")
    print(f"      Constraint OK:     {'NON ❌' if sharpe_metrics['exceeds_constraint'] else 'OUI ✓'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE DES MODÈLES ET RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("6. SAUVEGARDE")
print("="*80)

# Sauvegarder les modèles
for name, model in final_models.items():
    filepath = f'/home/claude/models/saved/{name}_final.pkl'
    model.save(filepath)

# Sauvegarder les features sélectionnées
import json
with open('/home/claude/models/saved/selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)
print("✓ Features sélectionnées sauvegardées")

# Sauvegarder les résultats
results_df = pd.DataFrame(sharpe_results)
results_df.to_csv('/home/claude/models/saved/sharpe_results.csv', index=False)
print("✓ Résultats Sharpe sauvegardés")

summary_df.to_csv('/home/claude/models/saved/validation_summary.csv', index=False)
print("✓ Résumé de validation sauvegardé")

print("\n" + "="*80)
print("✓ ENTRAÎNEMENT AVANCÉ TERMINÉ")
print("="*80)

# Meilleur modèle
best_model = max(sharpe_results, key=lambda x: x['sharpe_ratio'])
print(f"\n🏆 MEILLEUR MODÈLE: {best_model['model'].upper()}")
print(f"   Sharpe Ratio: {best_model['sharpe_ratio']:+.4f}")
print(f"   Rendement:    {best_model['annualized_return']*100:+.2f}%")
