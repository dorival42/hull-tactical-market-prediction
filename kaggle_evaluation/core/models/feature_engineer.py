"""
═══════════════════════════════════════════════════════════════════════════════
FEATURE ENGINEERING - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

Classe pour créer et gérer les features avancées.

Auteur: Advanced Modeling Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Classe pour créer et gérer les features.
    
    Fonctionnalités:
    - Lag features
    - Rolling statistics
    - Technical indicators
    - Interaction features
    - Feature selection
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialisation.
        
        Args:
            verbose: Afficher les messages
        """
        self.verbose = verbose
        self.scaler = None
        self.feature_names = []
        self.selected_features = []
        
    def create_lag_features(self, df: pd.DataFrame, 
                          columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
        """
        Créer des features de lag.
        
        Args:
            df: DataFrame
            columns: Colonnes à lagguer
            lags: Liste des lags (ex: [1, 5, 10])
            
        Returns:
            DataFrame avec nouvelles features
        """
        if self.verbose:
            print(f"   Création de {len(columns) * len(lags)} lag features...")
        
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    feature_name = f'{col}_lag_{lag}'
                    df[feature_name] = df[col].shift(lag)
                    self.feature_names.append(feature_name)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               columns: List[str], 
                               windows: List[int],
                               functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Créer des rolling statistics.
        
        Args:
            df: DataFrame
            columns: Colonnes pour rolling stats
            windows: Fenêtres (ex: [5, 10, 20])
            functions: Fonctions ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            DataFrame avec nouvelles features
        """
        if self.verbose:
            print(f"   Création de {len(columns) * len(windows) * len(functions)} rolling features...")
        
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for func in functions:
                        feature_name = f'{col}_rolling_{func}_{window}'
                        
                        if func == 'mean':
                            df[feature_name] = df[col].rolling(window).mean()
                        elif func == 'std':
                            df[feature_name] = df[col].rolling(window).std()
                        elif func == 'min':
                            df[feature_name] = df[col].rolling(window).min()
                        elif func == 'max':
                            df[feature_name] = df[col].rolling(window).max()
                        elif func == 'median':
                            df[feature_name] = df[col].rolling(window).median()
                        
                        self.feature_names.append(feature_name)
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame, 
                                columns: List[str], 
                                windows: List[int]) -> pd.DataFrame:
        """
        Créer des features de momentum.
        
        Args:
            df: DataFrame
            columns: Colonnes pour momentum
            windows: Fenêtres
            
        Returns:
            DataFrame avec nouvelles features
        """
        if self.verbose:
            print(f"   Création de {len(columns) * len(windows)} momentum features...")
        
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Simple momentum (différence)
                    feature_name = f'{col}_momentum_{window}'
                    df[feature_name] = df[col] - df[col].shift(window)
                    self.feature_names.append(feature_name)
                    
                    # Rate of change
                    feature_name = f'{col}_roc_{window}'
                    df[feature_name] = (df[col] - df[col].shift(window)) / (df[col].shift(window).abs() + 1e-8)
                    self.feature_names.append(feature_name)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame, 
                                  return_col: str, 
                                  windows: List[int]) -> pd.DataFrame:
        """
        Créer des features de volatilité.
        
        Args:
            df: DataFrame
            return_col: Colonne des rendements
            windows: Fenêtres
            
        Returns:
            DataFrame avec nouvelles features
        """
        if self.verbose:
            print(f"   Création de {len(windows) * 2} volatility features...")
        
        df = df.copy()
        
        if return_col in df.columns:
            for window in windows:
                # Volatilité historique
                feature_name = f'volatility_{window}'
                df[feature_name] = df[return_col].rolling(window).std()
                self.feature_names.append(feature_name)
                
                # Volatilité réalisée
                feature_name = f'realized_vol_{window}'
                df[feature_name] = df[return_col].rolling(window).apply(
                    lambda x: np.sqrt(np.sum(x**2))
                )
                self.feature_names.append(feature_name)
        
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame, 
                                   price_col: Optional[str] = None) -> pd.DataFrame:
        """
        Créer des indicateurs techniques.
        
        Args:
            df: DataFrame
            price_col: Colonne de prix (optionnel)
            
        Returns:
            DataFrame avec nouvelles features
        """
        if self.verbose:
            print(f"   Création d'indicateurs techniques...")
        
        df = df.copy()
        
        # RSI (Relative Strength Index)
        if price_col and price_col in df.columns:
            for window in [14, 21]:
                delta = df[price_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / (loss + 1e-8)
                feature_name = f'rsi_{window}'
                df[feature_name] = 100 - (100 / (1 + rs))
                self.feature_names.append(feature_name)
        
        # MACD (Moving Average Convergence Divergence)
        target_col = 'market_forward_excess_returns'
        if target_col in df.columns:
            ema_12 = df[target_col].ewm(span=12).mean()
            ema_26 = df[target_col].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Créer des features d'interaction.
        
        Args:
            df: DataFrame
            feature_pairs: Liste de paires de features
            
        Returns:
            DataFrame avec nouvelles features
        """
        if self.verbose:
            print(f"   Création de {len(feature_pairs)} interaction features...")
        
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                feature_name = f'{feat1}_x_{feat2}'
                df[feature_name] = df[feat1] * df[feat2]
                self.feature_names.append(feature_name)
                
                # Division (éviter division par zéro)
                feature_name = f'{feat1}_div_{feat2}'
                df[feature_name] = df[feat1] / (df[feat2].abs() + 1e-8)
                self.feature_names.append(feature_name)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, 
                          target_col: str = 'market_forward_excess_returns') -> pd.DataFrame:
        """
        Créer toutes les features en une fois.
        
        Args:
            df: DataFrame
            target_col: Colonne target
            
        Returns:
            DataFrame avec toutes les nouvelles features
        """
        if self.verbose:
            print("\n" + "="*80)
            print("CRÉATION DES FEATURES")
            print("="*80)
        
        self.feature_names = []
        
        # Colonnes de base (exclure les colonnes non-numériques et la target)
        exclude_cols = ['date_id', target_col, 'forward_returns', 'risk_free_rate']
        base_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 1. Lag features pour les top features
        top_features = ['V13', 'M4', 'S5', 'P6', 'P4'] if any(f in df.columns for f in ['V13', 'M4']) else base_cols[:5]
        df = self.create_lag_features(df, top_features, lags=[1, 2, 3, 5, 10])
        
        # 2. Rolling statistics pour la target
        if target_col in df.columns:
            df = self.create_rolling_features(df, [target_col], 
                                             windows=[5, 10, 20, 60],
                                             functions=['mean', 'std'])
        
        # 3. Momentum features
        if target_col in df.columns:
            df = self.create_momentum_features(df, [target_col], windows=[5, 10, 20])
        
        # 4. Volatility features
        if 'forward_returns' in df.columns:
            df = self.create_volatility_features(df, 'forward_returns', windows=[5, 10, 20, 60])
        
        # 5. Technical indicators
        df = self.create_technical_indicators(df)
        
        # 6. Interaction features (top paires basées sur EDA)
        if all(f in df.columns for f in ['V13', 'M4']):
            interactions = [('V13', 'M4'), ('S5', 'P6'), ('M4', 'P4')]
            df = self.create_interaction_features(df, interactions)
        
        if self.verbose:
            print(f"\n✓ Total de {len(self.feature_names)} nouvelles features créées")
            print(f"✓ DataFrame: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        return df
    
    def select_features(self, df: pd.DataFrame, 
                       target_col: str,
                       method: str = 'correlation',
                       n_features: int = 50) -> List[str]:
        """
        Sélectionner les meilleures features.
        
        Args:
            df: DataFrame
            target_col: Colonne target
            method: Méthode ('correlation', 'mutual_info', 'variance')
            n_features: Nombre de features à sélectionner
            
        Returns:
            Liste de noms de features
        """
        if self.verbose:
            print(f"\n   Sélection des {n_features} meilleures features (méthode: {method})...")
        
        # Exclure les colonnes non-features
        exclude_cols = ['date_id', target_col, 'forward_returns', 'risk_free_rate']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if method == 'correlation':
            # Corrélation avec la target
            correlations = []
            for col in feature_cols:
                corr_data = df[[col, target_col]].dropna()
                if len(corr_data) > 100:
                    corr = abs(corr_data.corr().iloc[0, 1])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
            
            # Trier par corrélation absolue
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected = [feat for feat, _ in correlations[:n_features]]
        
        elif method == 'variance':
            # Variance des features
            variances = df[feature_cols].var().sort_values(ascending=False)
            selected = variances.head(n_features).index.tolist()
        
        elif method == 'mutual_info':
            # Mutual information
            from sklearn.feature_selection import mutual_info_regression
            
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)
            
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores})
            mi_df = mi_df.sort_values('mi_score', ascending=False)
            selected = mi_df.head(n_features)['feature'].tolist()
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        self.selected_features = selected
        
        if self.verbose:
            print(f"   ✓ {len(selected)} features sélectionnées")
        
        return selected
    
    def normalize_features(self, df_train: pd.DataFrame, 
                          df_test: pd.DataFrame,
                          feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normaliser les features.
        
        Args:
            df_train: DataFrame train
            df_test: DataFrame test
            feature_cols: Colonnes à normaliser
            
        Returns:
            Tuple (df_train_normalized, df_test_normalized)
        """
        if self.verbose:
            print(f"\n   Normalisation de {len(feature_cols)} features...")
        
        self.scaler = StandardScaler()
        
        # Fit sur train
        df_train = df_train.copy()
        df_test = df_test.copy()
        
        df_train[feature_cols] = self.scaler.fit_transform(df_train[feature_cols].fillna(0))
        df_test[feature_cols] = self.scaler.transform(df_test[feature_cols].fillna(0))
        
        if self.verbose:
            print(f"   ✓ Normalisation complétée")
        
        return df_train, df_test
    
    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtenir les statistiques des features.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame de statistiques
        """
        stats = []
        
        for col in df.columns:
            if col not in ['date_id']:
                stats.append({
                    'feature': col,
                    'missing_pct': df[col].isnull().sum() / len(df) * 100,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
        
        return pd.DataFrame(stats)


if __name__ == '__main__':
    print("="*80)
    print("FEATURE ENGINEERING - TEST")
    print("="*80)
    
    # Créer des données de test
    np.random.seed(42)
    n = 1000
    
    df_test = pd.DataFrame({
        'date_id': range(n),
        'market_forward_excess_returns': np.random.randn(n) * 0.01,
        'forward_returns': np.random.randn(n) * 0.01 + 0.0001,
        'V13': np.random.randn(n),
        'M4': np.random.randn(n),
        'S5': np.random.randn(n)
    })
    
    # Test Feature Engineer
    print("\nTest FeatureEngineer:")
    print("-" * 40)
    
    fe = FeatureEngineer(verbose=True)
    
    # Créer toutes les features
    df_enhanced = fe.create_all_features(df_test)
    
    print(f"\n✓ DataFrame final: {df_enhanced.shape}")
    print(f"✓ Nouvelles features: {len(fe.feature_names)}")
    
    # Sélection de features
    selected = fe.select_features(df_enhanced, 'market_forward_excess_returns', 
                                  method='correlation', n_features=20)
    print(f"\n✓ Top 10 features sélectionnées:")
    for i, feat in enumerate(selected[:10], 1):
        print(f"   {i}. {feat}")
    
    print("\n" + "="*80)
    print("✓ Tests Feature Engineering réussis")
    print("="*80)
