"""
═══════════════════════════════════════════════════════════════════════════════
BASE MODEL CLASS - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

Classe de base abstraite pour tous les modèles de prédiction.
Définit l'interface commune et les méthodes partagées.

Auteur: Advanced Modeling Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import pickle
import json
from pathlib import Path


class BaseModel(ABC):
    """
    Classe abstraite de base pour tous les modèles.
    
    Tous les modèles doivent implémenter:
    - fit(): Entraîner le modèle
    - predict(): Faire des prédictions
    - get_feature_importance(): Obtenir l'importance des features
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialisation du modèle.
        
        Args:
            name: Nom du modèle
            params: Dictionnaire de paramètres
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'BaseModel':
        """
        Entraîner le modèle.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faire des prédictions.
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            Array de prédictions
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtenir l'importance des features.
        
        Returns:
            DataFrame avec colonnes ['feature', 'importance']
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarder le modèle.
        
        Args:
            filepath: Chemin du fichier
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        model_data = {
            'name': self.name,
            'params': self.params,
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str) -> 'BaseModel':
        """
        Charger le modèle.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            self
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.name = model_data['name']
        self.params = model_data['params']
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_fitted = True
        
        print(f"✓ Modèle chargé: {filepath}")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Obtenir les paramètres du modèle."""
        return self.params
    
    def set_params(self, **params) -> 'BaseModel':
        """Définir les paramètres du modèle."""
        self.params.update(params)
        return self
    
    def get_metrics(self) -> Dict[str, float]:
        """Obtenir les métriques d'entraînement."""
        return self.training_metrics
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class ModelConfig:
    """Classe pour gérer la configuration des modèles."""
    
    # Paramètres par défaut pour chaque type de modèle
    DEFAULT_PARAMS = {
        'lightgbm': {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'xgboost': {
            'objective': 'reg:squarederror',
            'n_estimators': 700,
            'learning_rate': 0.01,
            'max_depth': 5,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        },
        'random_forest': {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        },
        'catboost': {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 5,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False
        },
        'lstm': {
            'units': 64,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 20
        },
        'sarimax': {
            'order': (1, 0, 1),
            'seasonal_order': (1, 0, 1, 5),
            'trend': 'c'
        }
    }
    
    @classmethod
    def get_default_params(cls, model_type: str) -> Dict[str, Any]:
        """
        Obtenir les paramètres par défaut pour un type de modèle.
        
        Args:
            model_type: Type de modèle ('lightgbm', 'xgboost', etc.)
            
        Returns:
            Dictionnaire de paramètres
        """
        return cls.DEFAULT_PARAMS.get(model_type, {}).copy()
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], filepath: str) -> None:
        """Sauvegarder une configuration."""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration sauvegardée: {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> Dict[str, Any]:
        """Charger une configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"✓ Configuration chargée: {filepath}")
        return config


class ModelMetrics:
    """Classe pour calculer les métriques de performance."""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculer les métriques de régression.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            Dictionnaire de métriques
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Filtrer les NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan
            }
        
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # MAPE (éviter division par zéro)
        mask_nonzero = y_true_clean != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero]) / 
                                  y_true_clean[mask_nonzero])) * 100
        else:
            mape = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    @staticmethod
    def calculate_sharpe_ratio(allocations: np.ndarray,
                              target_returns: np.ndarray) -> Dict[str, float]:
        """
        Calcul Sharpe  sur market_forward_excess_returns.
        
        Args:
            allocations: array [0, 2]
            target_returns: market_forward_excess_returns (déjà excess)
        
        Returns:
            dict avec métriques Sharpe
        """
        # Portfolio excess returns = allocation × target
        # target est déjà excess (forward_returns - risk_free_rate)
        portfolio_excess = allocations * target_returns
        
        # Annualiser (252 jours de trading)
        annual_return = portfolio_excess.mean() * 252
        annual_volatility = portfolio_excess.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return / annual_volatility 
                        if annual_volatility > 0 else 0.0)
        
        # Contrainte Kaggle: Volatility Ratio ≤ 1.5
        market_volatility = target_returns.std() * np.sqrt(252)
        volatility_ratio = (annual_volatility / market_volatility 
                            if market_volatility > 0 else 0.0)
        exceeds_constraint = volatility_ratio > 1.5
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annual_return,
            'annualized_volatility': annual_volatility,
            'volatility_ratio': volatility_ratio,
            'exceeds_constraint': exceeds_constraint
        }

