"""
═══════════════════════════════════════════════════════════════════════════════
GRADIENT BOOSTING MODELS - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

Implémentations des modèles Gradient Boosting:
- LightGBM
- XGBoost  
- CatBoost

Auteur: Advanced Modeling Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
sys.path.append('/home/claude')

import pandas as pd
import numpy as np
from typing import Optional
import lightgbm as lgb
import xgboost as xgb
from models.base_model import BaseModel, ModelConfig, ModelMetrics


class LightGBMModel(BaseModel):
    """Modèle LightGBM pour la prédiction de rendements."""
    
    def __init__(self, name: str = 'LightGBM', params: Optional[dict] = None):
        if params is None:
            params = ModelConfig.get_default_params('lightgbm')
        super().__init__(name, params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'LightGBMModel':
        """Entraîner le modèle LightGBM."""
        
        self.feature_names = X_train.columns.tolist()
        
        # Créer le modèle
        self.model = lgb.LGBMRegressor(**self.params)
        
        # Entraîner
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Métriques de validation
            y_pred_val = self.model.predict(X_val)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_val.values, y_pred_val
            )
        else:
            self.model.fit(X_train, y_train)
            
            # Métriques sur train
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train.values, y_pred_train
            )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtenir l'importance des features."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class XGBoostModel(BaseModel):
    """Modèle XGBoost pour la prédiction de rendements."""
    
    def __init__(self, name: str = 'XGBoost', params: Optional[dict] = None):
        if params is None:
            params = ModelConfig.get_default_params('xgboost')
        super().__init__(name, params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'XGBoostModel':
        """Entraîner le modèle XGBoost."""
        
        self.feature_names = X_train.columns.tolist()
        
        # Créer le modèle
        self.model = xgb.XGBRegressor(**self.params)
        
        # Entraîner
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Métriques de validation
            y_pred_val = self.model.predict(X_val)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_val.values, y_pred_val
            )
        else:
            self.model.fit(X_train, y_train)
            
            # Métriques sur train
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train.values, y_pred_train
            )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtenir l'importance des features."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class CatBoostModel(BaseModel):
    """Modèle CatBoost pour la prédiction de rendements."""
    
    def __init__(self, name: str = 'CatBoost', params: Optional[dict] = None):
        if params is None:
            params = ModelConfig.get_default_params('catboost')
        super().__init__(name, params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'CatBoostModel':
        """Entraîner le modèle CatBoost."""
        
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost n'est pas installé. Installer avec: pip install catboost")
        
        self.feature_names = X_train.columns.tolist()
        
        # Créer le modèle
        self.model = CatBoostRegressor(**self.params)
        
        # Entraîner
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=50
            )
            
            # Métriques de validation
            y_pred_val = self.model.predict(X_val)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_val.values, y_pred_val
            )
        else:
            self.model.fit(X_train, y_train)
            
            # Métriques sur train
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train.values, y_pred_train
            )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtenir l'importance des features."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class RandomForestModel(BaseModel):
    """Modèle Random Forest pour la prédiction de rendements."""
    
    def __init__(self, name: str = 'RandomForest', params: Optional[dict] = None):
        if params is None:
            params = ModelConfig.get_default_params('random_forest')
        super().__init__(name, params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'RandomForestModel':
        """Entraîner le modèle Random Forest."""
        
        from sklearn.ensemble import RandomForestRegressor
        
        self.feature_names = X_train.columns.tolist()
        
        # Créer le modèle
        self.model = RandomForestRegressor(**self.params)
        
        # Entraîner
        self.model.fit(X_train, y_train)
        
        # Métriques
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_val.values, y_pred_val
            )
        else:
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train.values, y_pred_train
            )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtenir l'importance des features."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


if __name__ == '__main__':
    print("="*80)
    print("GRADIENT BOOSTING MODELS - TEST")
    print("="*80)
    
    # Créer des données de test
    np.random.seed(42)
    n_train = 800
    n_val = 200
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randn(n_train) * 0.01)
    
    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.randn(n_val) * 0.01)
    
    # Test LightGBM
    print("\n1. Test LightGBM")
    print("-" * 40)
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    
    y_pred = lgb_model.predict(X_val)
    print(f"✓ Modèle entraîné")
    print(f"  RMSE: {lgb_model.training_metrics['rmse']:.6f}")
    print(f"  R²:   {lgb_model.training_metrics['r2']:.4f}")
    
    importance = lgb_model.get_feature_importance()
    print(f"\n  Top 3 features:")
    for i, row in importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.2f}")
    
    # Test XGBoost
    print("\n2. Test XGBoost")
    print("-" * 40)
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    
    print(f"✓ Modèle entraîné")
    print(f"  RMSE: {xgb_model.training_metrics['rmse']:.6f}")
    print(f"  R²:   {xgb_model.training_metrics['r2']:.4f}")
    
    # Test Random Forest
    print("\n3. Test Random Forest")
    print("-" * 40)
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train, X_val, y_val)
    
    print(f"✓ Modèle entraîné")
    print(f"  RMSE: {rf_model.training_metrics['rmse']:.6f}")
    print(f"  R²:   {rf_model.training_metrics['r2']:.4f}")
    
    print("\n" + "="*80)
    print("✓ Tests Gradient Boosting réussis")
    print("="*80)