"""
Pipeline complet pour Hull Tactical Market Prediction
VERSION FINALE OPTIMISÉE avec Preprocessor V2
Baseline avec XGBoost et feature engineering avancé
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import du preprocessor V2 optimisé
from preprocessor_v2 import HullPreprocessorV2
# LIGNE 17 - Remplacer:
from preprocessor_v2 import HullPreprocessorV2

# Par:
from preprocessor_simple import HullPreprocessorSimple as HullPreprocessorV2
# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CONFIG = {
    'random_state': 42,
    'n_splits': 5,
    'test_size': 0.2,
    'model_path': 'xgb_model.pkl',
    'preprocessor_path': 'preprocessor.pkl',
    'feature_importance_path': 'feature_importance.csv',
    
    # Hyperparamètres XGBoost (optimisés pour Sharpe Ratio)
    'xgb_params': {
        'n_estimators': 700,
        'max_depth': 6,
        'learning_rate': 0.008,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
}

# ==============================================================================
# 2. CALCUL DU SHARPE RATIO
# ==============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculer le Sharpe Ratio annualisé
    
    Args:
        returns: prédictions ou rendements (array-like)
        risk_free_rate: taux sans risque (déjà pris en compte dans excess returns)
    
    Returns:
        Sharpe Ratio annualisé (float)
    """
    # Convertir en array numpy si nécessaire
    returns = np.asarray(returns).flatten()
    
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    
    # Annualiser (252 jours de trading)
    sharpe_annualized = float(sharpe * np.sqrt(252))
    
    return sharpe_annualized


def calculate_sortino_ratio(returns, risk_free_rate=0):
    """Calculer le Sortino Ratio (comme Sharpe mais uniquement downside risk)"""
    returns = np.asarray(returns).flatten()
    
    mean_return = float(np.mean(returns))
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = float(np.std(downside_returns))
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    
    sortino = (mean_return - risk_free_rate) / downside_std
    sortino_annualized = float(sortino * np.sqrt(252))
    
    return sortino_annualized


def calculate_max_drawdown(returns):
    """Calculer le Maximum Drawdown"""
    if isinstance(returns, pd.Series):
        returns_series = returns
    else:
        returns_series = pd.Series(returns)
    
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    return float(drawdown.min())


def calculate_metrics(y_true, y_pred):
    """Calculer toutes les métriques de performance"""
    
    # S'assurer que ce sont des arrays numpy
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Erreurs de prédiction
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Sharpe Ratio (métrique principale) - utiliser y_pred uniquement
    sharpe = calculate_sharpe_ratio(y_pred)
    
    # Sortino Ratio
    sortino = calculate_sortino_ratio(y_pred)
    
    # Hit rate (% de fois où on prédit le bon signe)
    hit_rate = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    
    # Maximum Drawdown
    max_dd = calculate_max_drawdown(y_pred)
    
    # Corrélation
    if len(y_pred) > 1:
        corr_matrix = np.corrcoef(y_pred, y_true)
        correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        correlation = 0.0
    
    # Volatilité
    volatility = float(np.std(y_pred))
    
    return {
        'MSE': mse,
        'RMSE': float(np.sqrt(mse)),
        'MAE': mae,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Hit_Rate': hit_rate,
        'Max_Drawdown': max_dd,
        'Correlation': correlation,
        'Volatility': volatility
    }

# ==============================================================================
# 3. TRAINING PIPELINE
# ==============================================================================

def train_baseline_model():
    """Pipeline complet d'entraînement"""
    
    print("=" * 80)
    print("HULL TACTICAL - TRAINING PIPELINE V2")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Preprocessor: HullPreprocessorV2")
    print(f"  - Model: XGBoost")
    print(f"  - Validation split: {CONFIG['test_size']*100}%")
    print(f"  - Random state: {CONFIG['random_state']}")
    
    # 1. Charger les données
    print("\n" + "=" * 80)
    print("1. CHARGEMENT DES DONNÉES")
    print("=" * 80)
    
    if not Path('train.csv').exists():
        raise FileNotFoundError("train.csv not found. Please download the data first.")
    
    train = pd.read_csv('train.csv')
    print(f"✓ Train shape: {train.shape}")
    print(f"✓ Période: date_id {train['date_id'].min()} à {train['date_id'].max()}")
    print(f"✓ Colonnes: {len(train.columns)}")
    
    # Vérifier la target
    target = 'market_forward_excess_returns'
    if target not in train.columns:
        raise ValueError(f"Target column '{target}' not found in train.csv")
    
    y = train[target].values
    print(f"\n✓ Target: {target}")
    print(f"  - Mean: {y.mean():.6f}")
    print(f"  - Std: {y.std():.6f}")
    print(f"  - Min: {y.min():.6f}")
    print(f"  - Max: {y.max():.6f}")
    
    # 2. Preprocessing avec V2
    print("\n" + "=" * 80)
    print("2. PREPROCESSING (HullPreprocessorV2)")
    print("=" * 80)
    
    preprocessor = HullPreprocessorV2(verbose=True)
    X = preprocessor.fit_transform(train)
    
    print(f"\n✓ Features shape: {X.shape}")
    print(f"✓ Nombre de features: {len(preprocessor.feature_names)}")
    
    # Résumé des features
    summary = preprocessor.get_feature_summary()
    print(f"\nRésumé des features:")
    print(f"  - Original: {summary['original_features']}")
    print(f"  - Lagged: {summary['lagged_features']}")
    print(f"  - Created: {summary['created_features']}")
    print(f"  - Total: {summary['total_features']}")
    
    # 3. Train/Validation split (time-based)
    print("\n" + "=" * 80)
    print("3. TRAIN/VALIDATION SPLIT")
    print("=" * 80)
    
    split_idx = int(len(X) * (1 - CONFIG['test_size']))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✓ Train set: {X_train.shape}")
    print(f"  - date_id range: {train['date_id'].iloc[:split_idx].min()} à {train['date_id'].iloc[:split_idx].max()}")
    print(f"  - Target mean: {y_train.mean():.6f}")
    
    print(f"\n✓ Validation set: {X_val.shape}")
    print(f"  - date_id range: {train['date_id'].iloc[split_idx:].min()} à {train['date_id'].iloc[split_idx:].max()}")
    print(f"  - Target mean: {y_val.mean():.6f}")
    
    # 4. Entraîner le modèle
    print("\n" + "=" * 80)
    print("4. ENTRAÎNEMENT DU MODÈLE XGBOOST")
    print("=" * 80)
    print(f"Hyperparamètres:")
    for key, value in CONFIG['xgb_params'].items():
        print(f"  {key:20s}: {value}")
    
    print(f"\n⏳ Training en cours...")
    print(f"⚠️  Cela peut prendre 5-10 minutes...\n")
    
    # Créer le modèle SANS eval_metric et early_stopping dans le constructeur
    model = xgb.XGBRegressor(**CONFIG['xgb_params'])
    
    # Fit avec eval_set mais SANS early_stopping pour forcer le training complet
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100  # Afficher tous les 100 iterations
    )
    
    print(f"\n✓ Training terminé")
    print(f"✓ Total iterations: {model.n_estimators}")
    
    # 5. Évaluation
    print("\n" + "=" * 80)
    print("5. ÉVALUATION DES PERFORMANCES")
    print("=" * 80)
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Métriques
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    print("\n📊 TRAIN METRICS:")
    for metric, value in train_metrics.items():
        # Convertir en scalaire si c'est un array
        if isinstance(value, np.ndarray):
            value = float(value)
        print(f"  {metric:15s}: {value:.6f}")
    
    print("\n📊 VALIDATION METRICS:")
    for metric, value in val_metrics.items():
        # Convertir en scalaire si c'est un array
        if isinstance(value, np.ndarray):
            value = float(value)
        print(f"  {metric:15s}: {value:.6f}")
    
    # Évaluation du Sharpe Ratio
    sharpe_val = val_metrics['Sharpe']
    print("\n" + "=" * 80)
    print("ÉVALUATION DU SHARPE RATIO")
    print("=" * 80)
    
    if sharpe_val > 1.0:
        print(f"🏆 EXCELLENT ! Sharpe = {sharpe_val:.4f}")
        print("   Vous êtes dans le top 10% !")
    elif sharpe_val > 0.7:
        print(f"🔵 TRÈS BON ! Sharpe = {sharpe_val:.4f}")
        print("   Vous êtes très compétitif !")
    elif sharpe_val > 0.5:
        print(f"🟢 BON ! Sharpe = {sharpe_val:.4f}")
        print("   Vous êtes compétitif, continuez à optimiser !")
    elif sharpe_val > 0.3:
        print(f"🟡 ACCEPTABLE. Sharpe = {sharpe_val:.4f}")
        print("   Baseline fonctionnel, optimisation recommandée.")
    else:
        print(f"⚠️  FAIBLE. Sharpe = {sharpe_val:.4f}")
        print("   Vérifiez les features et les hyperparamètres.")
    
    # 6. Feature importance
    print("\n" + "=" * 80)
    print("6. FEATURE IMPORTANCE")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'feature': preprocessor.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 20 Features importantes:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Vérifier que les lagged features sont importantes
    top_10_features = feature_importance.head(10)['feature'].tolist()
    lagged_in_top10 = sum(1 for f in top_10_features if 'lagged' in f or 'target' in f)
    
    print(f"\n📊 Features lagged/target dans le top 10: {lagged_in_top10}/10")
    if lagged_in_top10 >= 3:
        print("✅ Bon ! Les lagged features sont bien utilisées.")
    else:
        print("⚠️  Warning: Peu de lagged features dans le top 10.")
    
    # Sauvegarder feature importance
    feature_importance.to_csv(CONFIG['feature_importance_path'], index=False)
    print(f"\n✓ Feature importance sauvegardée: {CONFIG['feature_importance_path']}")
    
    # 7. Analyse par régime de marché
    print("\n" + "=" * 80)
    print("7. ANALYSE PAR RÉGIME DE MARCHÉ")
    print("=" * 80)
    
    # Définir les régimes basés sur les returns
    regimes_val = pd.cut(y_val, 
                         bins=[-np.inf, -0.005, 0.005, np.inf],
                         labels=['Bear', 'Neutral', 'Bull'])
    
    print("\n📊 Performance par régime (Validation):\n")
    
    for regime in ['Bear', 'Neutral', 'Bull']:
        mask = regimes_val == regime
        if mask.sum() == 0:
            continue
        
        pred_regime = y_val_pred[mask]
        actual_regime = y_val[mask]
        
        sharpe_regime = calculate_sharpe_ratio(pred_regime)
        hit_rate_regime = np.mean(np.sign(pred_regime) == np.sign(actual_regime))
        mae_regime = np.mean(np.abs(pred_regime - actual_regime))
        
        print(f"  {regime:8s} ({mask.sum():4d} jours):")
        print(f"    Sharpe    : {sharpe_regime:7.4f}")
        print(f"    Hit Rate  : {hit_rate_regime:7.2%}")
        print(f"    MAE       : {mae_regime:.6f}")
        print()
    
    # 8. Sauvegarder le modèle et le preprocessor
    print("=" * 80)
    print("8. SAUVEGARDE DU MODÈLE")
    print("=" * 80)
    
    # Sauvegarder le modèle
    with open(CONFIG['model_path'], 'wb') as f:
        pickle.dump(model, f)
    
    # Sauvegarder le preprocessor
    with open(CONFIG['preprocessor_path'], 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"✓ Modèle sauvegardé: {CONFIG['model_path']}")
    print(f"✓ Preprocessor sauvegardé: {CONFIG['preprocessor_path']}")
    
    # Vérifier la taille des fichiers
    model_size = Path(CONFIG['model_path']).stat().st_size / (1024 * 1024)
    prep_size = Path(CONFIG['preprocessor_path']).stat().st_size / (1024 * 1024)
    
    print(f"\nTailles des fichiers:")
    print(f"  - Modèle: {model_size:.2f} MB")
    print(f"  - Preprocessor: {prep_size:.2f} MB")
    print(f"  - Total: {model_size + prep_size:.2f} MB")
    
    if model_size + prep_size > 100:
        print("\n⚠️  Warning: Taille totale > 100MB. Optimisation recommandée.")
    else:
        print("\n✅ Taille OK pour Kaggle (<100MB)")
    
    # 9. Résumé final
    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    
    print(f"\n📊 Métriques clés (Validation):")
    print(f"  - Sharpe Ratio  : {val_metrics['Sharpe']:.4f}")
    print(f"  - Hit Rate      : {val_metrics['Hit_Rate']:.2%}")
    print(f"  - Correlation   : {val_metrics['Correlation']:.4f}")
    print(f"  - Max Drawdown  : {val_metrics['Max_Drawdown']:.2%}")
    print(f"  - MAE           : {val_metrics['MAE']:.6f}")
    
    print(f"\n✓ Features utilisées: {summary['total_features']}")
    print(f"✓ Lagged features: {summary['lagged_features']}")
    print(f"✓ Created features: {summary['created_features']}")
    
    print("\n" + "=" * 80)
    print("TRAINING TERMINÉ AVEC SUCCÈS")
    print("=" * 80)
    
    # Recommandations
    print("\n💡 PROCHAINES ÉTAPES:")
    
    if sharpe_val > 0.7:
        print("  1. ✅ Tester localement: python test_local.py --mode full")
        print("  2. ✅ Analyser les résultats: python analyze_results.py")
        print("  3. 🚀 SOUMETTRE À KAGGLE !")
    elif sharpe_val > 0.5:
        print("  1. ✅ Tester localement: python test_local.py --mode full")
        print("  2. 📊 Analyser les résultats: python analyze_results.py")
        print("  3. ⚡ Optimiser les hyperparamètres (optionnel)")
        print("  4. 🚀 Soumettre à Kaggle")
    else:
        print("  1. 📊 Analyser les résultats: python analyze_results.py")
        print("  2. 🔧 Vérifier les features (lagged features utilisées ?)")
        print("  3. ⚡ Optimiser les hyperparamètres")
        print("  4. 🧪 Re-tester et re-entraîner")
    
    return model, preprocessor, val_metrics

# ==============================================================================
# 4. TEST SUR LE FICHIER TEST
# ==============================================================================

def test_on_test_file(model, preprocessor):
    """Tester le modèle sur test.csv"""
    
    print("\n" + "=" * 80)
    print("TEST SUR test.csv")
    print("=" * 80)
    
    if not Path('test.csv').exists():
        print("⚠️  test.csv not found. Skipping test predictions.")
        return None
    
    test = pd.read_csv('test.csv')
    print(f"✓ Test shape: {test.shape}")
    
    # Preprocessing
    print("\n⏳ Preprocessing test data...")
    X_test = preprocessor.transform(test)
    print(f"✓ Test features shape: {X_test.shape}")
    
    # Prédictions
    print("\n⏳ Generating predictions...")
    predictions = model.predict(X_test)
    
    print("\n📊 Prédictions:")
    print(f"  - Nombre: {len(predictions)}")
    print(f"  - Min: {predictions.min():.6f}")
    print(f"  - Max: {predictions.max():.6f}")
    print(f"  - Mean: {predictions.mean():.6f}")
    print(f"  - Std: {predictions.std():.6f}")
    
    # Vérifier les valeurs anormales
    nan_count = np.isnan(predictions).sum()
    inf_count = np.isinf(predictions).sum()
    
    if nan_count > 0:
        print(f"\n⚠️  Warning: {nan_count} NaN predictions detected")
    if inf_count > 0:
        print(f"\n⚠️  Warning: {inf_count} Inf predictions detected")
    
    if nan_count == 0 and inf_count == 0:
        print("\n✅ Toutes les prédictions sont valides")
    
    # Créer un DataFrame de résultats
    results = pd.DataFrame({
        'date_id': test['date_id'],
        'prediction': predictions
    })
    
    print("\n📋 Premières prédictions:")
    print(results.head(10).to_string(index=False))
    
    # Sauvegarder les prédictions
    results.to_csv('test_predictions.csv', index=False)
    print(f"\n✓ Prédictions sauvegardées: test_predictions.csv")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    
    import time
    start_time = time.time()
    
    try:
        # Entraîner le modèle
        model, preprocessor, metrics = train_baseline_model()
        
        # Tester sur test.csv
        results = test_on_test_file(model, preprocessor)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        print(f"⏱️  Temps total: {elapsed_time/60:.1f} minutes")
        print(f"🎯 Sharpe Ratio: {metrics['Sharpe']:.4f}")
        print(f"📁 Fichiers générés:")
        print(f"   - {CONFIG['model_path']}")
        print(f"   - {CONFIG['preprocessor_path']}")
        print(f"   - {CONFIG['feature_importance_path']}")
        if results is not None:
            print(f"   - test_predictions.csv")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERREUR LORS DE L'EXÉCUTION")
        print("=" * 80)
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Suggestions:")
        print("  1. Vérifiez que train.csv existe")
        print("  2. Vérifiez que preprocessor_v2.py est accessible")
        print("  3. Vérifiez les packages installés (xgboost, pandas, etc.)")