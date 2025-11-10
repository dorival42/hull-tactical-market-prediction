"""
═══════════════════════════════════════════════════════════════════════════════
DEEP LEARNING MODELS - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

Implémentations des modèles Deep Learning:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Simple Neural Network
- Attention Mechanism (optionnel)

Auteur: Deep Learning Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import sys,  os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from base_model import BaseModel, ModelConfig, ModelMetrics


class SequencePreprocessor:
    """Préprocesseur pour créer des séquences temporelles."""
    
    def __init__(self, sequence_length: int = 20):
        """
        Initialisation.
        
        Args:
            sequence_length: Longueur des séquences
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Créer des séquences pour les modèles récurrents.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
            
        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i - self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None, 
                     y_val: Optional[pd.Series] = None) -> dict:
        """
        Préparer les données pour l'entraînement.
        
        Returns:
            Dictionnaire avec X_train_seq, y_train_seq, X_val_seq, y_val_seq
        """
        # Normaliser
        X_train_scaled = self.scaler.fit_transform(X_train.fillna(0))
        y_train_array = y_train.values
        
        # Créer séquences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_array)
        
        result = {
            'X_train': X_train_seq,
            'y_train': y_train_seq
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val.fillna(0))
            y_val_array = y_val.values
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_array)
            result['X_val'] = X_val_seq
            result['y_val'] = y_val_seq
        
        return result


class LSTMModel(BaseModel):
    """Modèle LSTM pour prédiction de séries temporelles."""
    
    def __init__(self, name: str = 'LSTM', params: Optional[dict] = None):
        if params is None:
            params = ModelConfig.get_default_params('lstm')
        super().__init__(name, params)
        self.preprocessor = SequencePreprocessor(sequence_length=params.get('sequence_length', 20))
        self.history = None
    
    def _build_model(self, input_shape: tuple) -> Model:
        """
        Construire le modèle LSTM.
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Modèle compilé
        """
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = layers.LSTM(
            self.params.get('units', 64),
            return_sequences=True,
            dropout=self.params.get('dropout', 0.2)
        )(inputs)
        
        x = layers.LSTM(
            self.params.get('units', 64) // 2,
            dropout=self.params.get('dropout', 0.2)
        )(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.params.get('dropout', 0.2))(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compiler
        model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'LSTMModel':
        """Entraîner le modèle LSTM."""
        
        print(f"   Préparation des séquences...")
        
        # Préparer les données
        data = self.preprocessor.fit_transform(X_train, y_train, X_val, y_val)
        
        X_train_seq = data['X_train']
        y_train_seq = data['y_train']
        
        print(f"   Shape des séquences: {X_train_seq.shape}")
        
        # Construire le modèle
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        print(f"   Entraînement du modèle LSTM...")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.params.get('patience', 20),
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )
        
        # Entraîner
        if X_val is not None:
            self.history = self.model.fit(
                X_train_seq, y_train_seq,
                validation_data=(data['X_val'], data['y_val']),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Métriques sur validation
            y_pred_val = self.model.predict(data['X_val'], verbose=0).flatten()
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                data['y_val'], y_pred_val
            )
        else:
            self.history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Métriques sur train
            y_pred_train = self.model.predict(X_train_seq, verbose=0).flatten()
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train_seq, y_pred_train
            )
        
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True
        
        print(f"   ✓ LSTM entraîné - RMSE: {self.training_metrics['rmse']:.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        # Normaliser
        X_scaled = self.preprocessor.scaler.transform(X.fillna(0))
        
        # Créer séquences (utiliser padding pour les premières valeurs)
        sequence_length = self.preprocessor.sequence_length
        
        if len(X_scaled) < sequence_length:
            # Padding si pas assez de données
            padding = np.zeros((sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        predictions = []
        for i in range(sequence_length, len(X_scaled) + 1):
            seq = X_scaled[i - sequence_length:i].reshape(1, sequence_length, -1)
            pred = self.model.predict(seq, verbose=0)[0, 0]
            predictions.append(pred)
        
        # Compléter avec des valeurs moyennes pour les premières prédictions
        if len(predictions) < len(X):
            mean_pred = np.mean(predictions) if len(predictions) > 0 else 0
            predictions = [mean_pred] * (len(X) - len(predictions)) + predictions
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """LSTM n'a pas d'importance de features directe."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': [1.0] * len(self.feature_names)
        })


class GRUModel(BaseModel):
    """Modèle GRU (plus rapide que LSTM)."""
    
    def __init__(self, name: str = 'GRU', params: Optional[dict] = None):
        if params is None:
            params = ModelConfig.get_default_params('lstm')
        super().__init__(name, params)
        self.preprocessor = SequencePreprocessor(sequence_length=params.get('sequence_length', 20))
        self.history = None
    
    def _build_model(self, input_shape: tuple) -> Model:
        """Construire le modèle GRU."""
        inputs = layers.Input(shape=input_shape)
        
        # GRU layers (plus rapide que LSTM)
        x = layers.GRU(
            self.params.get('units', 64),
            return_sequences=True,
            dropout=self.params.get('dropout', 0.2)
        )(inputs)
        
        x = layers.GRU(
            self.params.get('units', 64) // 2,
            dropout=self.params.get('dropout', 0.2)
        )(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.params.get('dropout', 0.2))(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'GRUModel':
        """Entraîner le modèle GRU."""
        
        print(f"   Préparation des séquences...")
        
        data = self.preprocessor.fit_transform(X_train, y_train, X_val, y_val)
        
        X_train_seq = data['X_train']
        y_train_seq = data['y_train']
        
        print(f"   Shape des séquences: {X_train_seq.shape}")
        
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        print(f"   Entraînement du modèle GRU...")
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.params.get('patience', 20),
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )
        
        if X_val is not None:
            self.history = self.model.fit(
                X_train_seq, y_train_seq,
                validation_data=(data['X_val'], data['y_val']),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            y_pred_val = self.model.predict(data['X_val'], verbose=0).flatten()
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                data['y_val'], y_pred_val
            )
        else:
            self.history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            y_pred_train = self.model.predict(X_train_seq, verbose=0).flatten()
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train_seq, y_pred_train
            )
        
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True
        
        print(f"   ✓ GRU entraîné - RMSE: {self.training_metrics['rmse']:.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        X_scaled = self.preprocessor.scaler.transform(X.fillna(0))
        sequence_length = self.preprocessor.sequence_length
        
        if len(X_scaled) < sequence_length:
            padding = np.zeros((sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        predictions = []
        for i in range(sequence_length, len(X_scaled) + 1):
            seq = X_scaled[i - sequence_length:i].reshape(1, sequence_length, -1)
            pred = self.model.predict(seq, verbose=0)[0, 0]
            predictions.append(pred)
        
        if len(predictions) < len(X):
            mean_pred = np.mean(predictions) if len(predictions) > 0 else 0
            predictions = [mean_pred] * (len(X) - len(predictions)) + predictions
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """GRU n'a pas d'importance de features directe."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': [1.0] * len(self.feature_names)
        })


class SimpleNNModel(BaseModel):
    """Simple Neural Network (Feed-Forward)."""
    
    def __init__(self, name: str = 'SimpleNN', params: Optional[dict] = None):
        if params is None:
            params = {
                'layers': [128, 64, 32],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'patience': 20
            }
        super().__init__(name, params)
        self.scaler = StandardScaler()
        self.history = None
    
    def _build_model(self, input_dim: int) -> Model:
        """Construire le réseau de neurones."""
        inputs = layers.Input(shape=(input_dim,))
        
        x = inputs
        for units in self.params.get('layers', [128, 64, 32]):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.params.get('dropout', 0.3))(x)
            x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'SimpleNNModel':
        """Entraîner le modèle."""
        
        self.feature_names = X_train.columns.tolist()
        
        # Normaliser
        X_train_scaled = self.scaler.fit_transform(X_train.fillna(0))
        y_train_array = y_train.values
        
        # Construire le modèle
        self.model = self._build_model(X_train_scaled.shape[1])
        
        print(f"   Entraînement du Simple NN...")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.params.get('patience', 20),
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )
        
        # Entraîner
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val.fillna(0))
            y_val_array = y_val.values
            
            self.history = self.model.fit(
                X_train_scaled, y_train_array,
                validation_data=(X_val_scaled, y_val_array),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            y_pred_val = self.model.predict(X_val_scaled, verbose=0).flatten()
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_val_array, y_pred_val
            )
        else:
            self.history = self.model.fit(
                X_train_scaled, y_train_array,
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            y_pred_train = self.model.predict(X_train_scaled, verbose=0).flatten()
            self.training_metrics = ModelMetrics.calculate_regression_metrics(
                y_train_array, y_pred_train
            )
        
        self.is_fitted = True
        
        print(f"   ✓ SimpleNN entraîné - RMSE: {self.training_metrics['rmse']:.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Calculer importance approximative via poids."""
        # Approximation: moyenne des poids absolus de la première couche
        first_layer_weights = self.model.layers[1].get_weights()[0]
        importance = np.abs(first_layer_weights).mean(axis=1)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


if __name__ == '__main__':
    print("="*80)
    print("DEEP LEARNING MODELS - TEST")
    print("="*80)
    
    # Créer des données de test
    np.random.seed(42)
    n_train = 500
    n_val = 100
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
    
    # Test LSTM
    print("\n1. Test LSTM")
    print("-" * 40)
    lstm_model = LSTMModel(params={'epochs': 10, 'sequence_length': 10})
    lstm_model.fit(X_train, y_train, X_val, y_val)
    print(f"   R²: {lstm_model.training_metrics['r2']:.4f}")
    
    # Test GRU
    print("\n2. Test GRU")
    print("-" * 40)
    gru_model = GRUModel(params={'epochs': 10, 'sequence_length': 10})
    gru_model.fit(X_train, y_train, X_val, y_val)
    print(f"   R²: {gru_model.training_metrics['r2']:.4f}")
    
    # Test Simple NN
    print("\n3. Test Simple NN")
    print("-" * 40)
    nn_model = SimpleNNModel(params={'epochs': 10})
    nn_model.fit(X_train, y_train, X_val, y_val)
    print(f"   R²: {nn_model.training_metrics['r2']:.4f}")
    
    print("\n" + "="*80)
    print("✓ Tests Deep Learning réussis")
    print("="*80)
