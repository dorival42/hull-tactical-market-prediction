"""
═══════════════════════════════════════════════════════════════════════════════
HULL TACTICAL - SUBMISSION SCRIPT
═══════════════════════════════════════════════════════════════════════════════

Ce script implémente l'InferenceServer pour la soumission Kaggle.
Basé sur la meilleure stratégie baseline : Momentum (5 jours)

Auteur: Submission Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle

# Ajouter le chemin du module kaggle_evaluation
sys.path.append('/home/claude/kaggle_evaluation')

from default_inference_server import DefaultInferenceServer

print("="*80)
print("HULL TACTICAL - SUBMISSION SCRIPT")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

# Choisir la stratégie à utiliser
STRATEGY = 'momentum'  # Options: 'momentum', 'lightgbm', 'xgboost', 'buy_hold'
MOMENTUM_WINDOW = 5

print(f"\n📊 Configuration:")
print(f"   Stratégie: {STRATEGY}")
if STRATEGY == 'momentum':
    print(f"   Momentum Window: {MOMENTUM_WINDOW} jours")

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE D'ÉTAT GLOBALE (pour gérer l'historique)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalState:
    """Classe pour maintenir l'état entre les appels à predict()."""
    
    def __init__(self):
        self.history = []  # Historique des rendements
        self.model = None
        self.feature_cols = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle si nécessaire."""
        if STRATEGY == 'lightgbm':
            try:
                with open('/home/claude/lgb_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                print("   ✓ Modèle LightGBM chargé")
            except Exception as e:
                print(f"     Erreur chargement LightGBM: {e}")
        
        elif STRATEGY == 'xgboost':
            try:
                with open('/home/claude/xgb_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                print("   ✓ Modèle XGBoost chargé")
            except Exception as e:
                print(f"     Erreur chargement XGBoost: {e}")
    
    def update_history(self, value):
        """Ajoute une valeur à l'historique."""
        self.history.append(value)
        # Garder seulement les dernières valeurs nécessaires
        max_window = max(MOMENTUM_WINDOW, 20)
        if len(self.history) > max_window:
            self.history = self.history[-max_window:]

# Initialiser l'état global
STATE = GlobalState()

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE STRATÉGIE
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_momentum(test_batch, window=5):
    """
    Strategie Momentum Simple - retourne la prediction du momentum.
    """
    # Convertir en pandas si c'est un polars DataFrame
    if hasattr(test_batch, 'to_pandas'):
        test_batch = test_batch.to_pandas()

    # Recuperer les rendements lagges
    if 'lagged_market_forward_excess_returns' in test_batch.columns:
        current_return = test_batch['lagged_market_forward_excess_returns'].iloc[0]
        STATE.update_history(current_return)

    # Calculer le momentum comme prediction
    if len(STATE.history) >= window:
        prediction = np.mean(STATE.history[-window:])
    else:
        prediction = 0.0

    return prediction

def strategy_lightgbm(test_batch):
    """
    Prediction basee sur LightGBM.
    """
    if STATE.model is None:
        return 0.0

    # Convertir en pandas si c'est un polars DataFrame
    if hasattr(test_batch, 'to_pandas'):
        test_batch = test_batch.to_pandas()

    # Preparer les features
    exclude_cols = ['date_id', 'is_scored',
                    'lagged_forward_returns', 'lagged_risk_free_rate',
                    'lagged_market_forward_excess_returns']
    feature_cols = [col for col in test_batch.columns if col not in exclude_cols]

    X = test_batch[feature_cols]

    # Predire
    prediction = STATE.model.predict(X)[0]

    return float(prediction)

def strategy_xgboost(test_batch):
    """
    Prediction basee sur XGBoost.
    """
    if STATE.model is None:
        return 0.0

    # Convertir en pandas si c'est un polars DataFrame
    if hasattr(test_batch, 'to_pandas'):
        test_batch = test_batch.to_pandas()

    # Preparer les features
    exclude_cols = ['date_id', 'is_scored',
                    'lagged_forward_returns', 'lagged_risk_free_rate',
                    'lagged_market_forward_excess_returns']
    feature_cols = [col for col in test_batch.columns if col not in exclude_cols]

    X = test_batch[feature_cols]

    # Predire
    prediction = STATE.model.predict(X)[0]

    return float(prediction)

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION PREDICT (APPELÉE PAR L'API)
# ═══════════════════════════════════════════════════════════════════════════════

def predict(test_batch):
    """
    Fonction principale appelee par l'API Kaggle pour chaque batch.

    Args:
        test_batch: DataFrame contenant les features pour un date_id

    Returns:
        float: Prediction de market_forward_excess_returns
    """
    try:
        # Selectionner la strategie
        if STRATEGY == 'momentum':
            prediction = strategy_momentum(test_batch, window=MOMENTUM_WINDOW)
        elif STRATEGY == 'lightgbm':
            prediction = strategy_lightgbm(test_batch)
        elif STRATEGY == 'xgboost':
            prediction = strategy_xgboost(test_batch)
        else:
            prediction = 0.0

        return float(prediction)

    except Exception as e:
        print(f"Erreur dans predict(): {e}")
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n🚀 Démarrage du serveur d'inférence...")
    
    # Créer le serveur d'inférence
    inference_server = DefaultInferenceServer(predict)
    
    # Vérifier si on est en mode test local ou soumission Kaggle
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("   Mode: Soumission Kaggle (rerun)")
        inference_server.serve()
    else:
        print("   Mode: Test local")
        print("\n   Exécution du test local avec le mock test set...")
        
        try:
            # Tester localement
            inference_server.run_local_gateway()
            
            print("\n" + "="*80)
            print("✓ TEST LOCAL TERMINÉ AVEC SUCCÈS")
            print("="*80)
            
            # Vérifier le fichier de soumission
            if os.path.exists('submission.parquet'):
                submission = pd.read_parquet('submission.parquet')
                print(f"\n📊 Fichier de soumission créé:")
                print(f"   Lignes: {len(submission)}")
                print(f"   Colonnes: {list(submission.columns)}")
                print(f"\n   Aperçu:")
                print(submission.head(10))
                
                # Statistiques sur les predictions
                if 'prediction' in submission.columns:
                    preds = submission['prediction']
                    print(f"\nStatistiques des predictions:")
                    print(f"   Mean:   {preds.mean():.6f}")
                    print(f"   Std:    {preds.std():.6f}")
                    print(f"   Min:    {preds.min():.6f}")
                    print(f"   Max:    {preds.max():.6f}")
                    print(f"   Median: {preds.median():.6f}")
            else:
                print("\n⚠️  Fichier submission.parquet non trouvé!")
        
        except Exception as e:
            print(f"\n❌ ERREUR LORS DU TEST LOCAL:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print("SCRIPT TERMINÉ")
print("="*80)