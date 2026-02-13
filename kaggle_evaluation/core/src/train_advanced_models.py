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

Auteur:Pierre Chrislin DORIVAL
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import sys,  os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from base_model import ModelMetrics
from feature_engineering.feature_engineer import FeatureEngineer
from gradient_boosting_models import (
    LightGBMModel, XGBoostModel, RandomForestModel, CatBoostModel
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
train = pd.read_csv('train.csv')

#print(f"\n Valeurs manquantes dans train:\n{train.isna().sum().sum()}")

print(f"\n Données train chargées: {train.shape[0]:,} lignes × {train.shape[1]} colonnes")


#test = pd.read_csv('test.csv')

#print(f"\n Valeurs manquantes dans test:\n{test.isna().sum().sum}")


#print(f"\n Données test chargées:  {test.shape[0]:,} lignes × {test.shape[1]} colonnes")


# Utiliser données récentes (moins de valeurs manquantes)
#CUTOFF_DATE = 6000
#train = train[train['date_id'] >= CUTOFF_DATE].copy()
#print(f"\n✓ Après cutoff (date_id >= {CUTOFF_DATE}): {train.shape[0]:,} lignes")

# Feature Engineering
fe = FeatureEngineer(verbose=True)
train_enhanced = fe.hull_transform(train)
print(f"\n Après feature engineering: {train_enhanced.shape[0]:,} lignes × {train_enhanced.shape[1]} colonnes")
print(f"\n Valeurs manquantes dans train après transformation:\n{train_enhanced.isna().sum().sum}")


#test_enhanced = fe.hull_transform(test)
#print(f"\n✓ Après feature engineering (test): {test_enhanced.shape[0]:,} lignes × {test_enhanced.shape[1]} colonnes")
#print(f"\n Valeurs manquantes dans test après transformation:\n{test_enhanced.isna().sum().sum}")

#print(test_enhanced.columns.tolist())
# Supprimer les lignes avec trop de NaN (dues aux rolling features)
#train_enhanced = train_enhanced.dropna(subset=['market_forward_excess_returns'])
#train_enhanced = train_enhanced.iloc[100:].copy()  # Skip first 100 rows (warm-up period)
#print(f"\n✓ Après suppression des NaN: {train_enhanced.shape[0]:,} lignes")

# Feature Selection
print("\n" + "-"*80)
print("FEATURE SELECTION")
print("-"*80)

target_col = 'market_forward_excess_returns'
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', target_col]

# Méthode 1: Corrélation
#selected_corr = fe.select_features(train_enhanced, target_col, 
 #                                  method='correlation', n_features=None)

 #Méthode 2: Mutual Information
selected_lgb = fe.select_features(train_enhanced, target_col, 'lgb_importance', 150)

# Combiner les deux méthodes
#selected_features = list(set(selected_corr[:150]) | set(selected_mi[:150]))
#print(f"\n✓ {len(selected_features)} features sélectionnées au total")

selected_features = selected_lgb

#selected_features = selected_corr
print(f"\n✓ {len(selected_features)} features sélectionnées au total")


#others_target_cols = ['forward_returns', 'risk_free_rate']
important_cols = [col for col in train_enhanced.columns 
                  if not col.startswith('feat_') and col not in exclude_cols]


train_enhanced = train_enhanced[['date_id'] + selected_features + [target_col]].copy()


# les colonnes train doit etre les memes que test autre que target


# Colonnes manquantes dans le test (hors target)
#missing_in_test = set(train_enhanced.columns) - set(test_enhanced.columns)
#missing_in_test.discard(target_col)  # supprime la colonne target si elle est présente

# Colonnes manquantes dans le train
#missing_in_train = set(test_enhanced.columns) - set(train_enhanced.columns)

# Supprimer les colonnes manquantes dans le train
#for col in missing_in_test:
 #   train_enhanced.drop(columns=col, inplace=True)

# Supprimer les colonnes manquantes dans le test
#for col in missing_in_train:
 #   test_enhanced.drop(columns=col, inplace=True)



#print(f"\nLes colonnes utilisées pour l'entraînement sont: {train_enhanced.columns.tolist()}")

#print(f"\nLes colonnes utilisées pour le test sont: {test_enhanced.columns.tolist()}")

print(f"\n Dataset final prêt pour l'entraînement: {train_enhanced.shape[0]:,} lignes : {train_enhanced.shape[1]} colonnes")
#print(f"\n Dataset final prêt pour le test: {test_enhanced.shape[0]:,} lignes : {test_enhanced.shape[1]} colonnes")
# ═══════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("2. WALK-FORWARD VALIDATION")
print("="*80)

def walk_forward_validation(df: pd.DataFrame, 
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
        
        train_fold.set_index('date_id', inplace=True)
        test_fold.set_index('date_id', inplace=True)

        # Préparer les données
        X_train = train_fold.drop(columns=target_col).fillna(0)
        y_train = train_fold[target_col]
        X_test = test_fold.drop(columns=target_col).fillna(0)
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
#wf_results = walk_forward_validation(train_enhanced, target_col, n_splits=5)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RÉSULTATS GLOBAUX
# ═══════════════════════════════════════════════════════════════════════════════
"""
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
"""
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


train_final.set_index('date_id', inplace=True)
val_final.set_index('date_id', inplace=True)
#train_final.pop('date_id', None)
#val_final.pop('date_id', None)

# Préparer les données
X_train_final = train_final.drop(columns=target_col).fillna(0)
y_train_final = train_final[target_col]
X_val_final = val_final.drop(columns=target_col).fillna(0)
y_val_final = val_final[target_col]

print(f"\n   Train final: {len(train_final):,} lignes")
print(f"   Val final:   {len(val_final):,} lignes")

# Entraîner les 4 meilleurs modèles
final_models = {}

print("\n   Entraînement LightGBM...")
lgb_final = LightGBMModel(name='LightGBM_Final')
lgb_final.fit(X_train_final, y_train_final, X_val_final, y_val_final)
final_models['lightgbm'] = lgb_final

print("\n   Entraînement CatGBM...")
cat_final = CatBoostModel(name='Catboost_Final')
cat_final.fit(X_train_final, y_train_final, X_val_final, y_val_final)
final_models['catboost'] = cat_final

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
# 5. PREDICTIONS SUR VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("5. PREDICTIONS SUR VALIDATION")
print("="*80)

prediction_results = []

for name, model in final_models.items():
    y_pred = model.predict(X_val_final)
    metrics = ModelMetrics.calculate_regression_metrics(y_val_final.values, y_pred)
    metrics['model'] = name
    prediction_results.append(metrics)

    print(f"\n   {name.upper()}:")
    print(f"      RMSE: {metrics['rmse']:.6f}")
    print(f"      MAE:  {metrics['mae']:.6f}")
    print(f"      R2:   {metrics['r2']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE DES MODÈLES ET RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("6. SAUVEGARDE")
print("="*80)

# Sauvegarder les modèles
for name, model in final_models.items():
    filepath = f'./kaggle_evaluation/core/models/{name}_final.pkl'
    model.save(filepath)

# Sauvegarder les features sélectionnées
import json
with open('./kaggle_evaluation/core/models/selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)
print("Features selectionnees sauvegardees")

# Sauvegarder les résultats
results_df = pd.DataFrame(prediction_results)
results_df.to_csv('./kaggle_evaluation/core/files_results/prediction_results.csv', index=False)
print("Resultats predictions sauvegardes")

print("\n" + "="*80)
print("ENTRAINEMENT AVANCE TERMINE")
print("="*80)

# Meilleur modèle
best_model = min(prediction_results, key=lambda x: x['rmse'])
print(f"\nMEILLEUR MODELE: {best_model['model'].upper()}")
print(f"   RMSE: {best_model['rmse']:.6f}")
print(f"   R2:   {best_model['r2']:.4f}")

