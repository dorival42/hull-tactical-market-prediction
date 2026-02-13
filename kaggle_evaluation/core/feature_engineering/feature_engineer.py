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
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


"""
═══════════════════════════════════════════════════════════════════════════════
FEATURE ENGINEERING - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
"""


class FeatureEngineer:
  
    def __init__(self, verbose: bool = True):

        self.feature_names = None
        self.numeric_features = []
        self.verbose = verbose
        self.scaler = None
        self.selected_features = []
        self._fit_stats = {}  # Pour stocker les stats d'imputation
        
    def _log(self, message):
       
        if self.verbose:
            print(message)
        
    def fit(self, df):
    
        
        # Identifier les features numériques (exclure target et metadata)
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        
        self.numeric_features = [col for col in df.columns 
                                if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        self._log(f" Features numériques de base: {len(self.numeric_features)}")
        
        # Calculer les statistiques pour l'imputation
        self._calculate_imputation_stats(df)
        
        return self
    

    def hull_transform(self, df):
        
        self._log("TRANSFORMING DATA")
       
        self._log(f"Input shape: {df.shape}")
        
        df = df.copy()
        
        # 1. Créer les lagged features si nécessaire (TRAIN)
        df = self._create_lagged_features(df)
        
        # 2. Feature engineering
        df = self._create_features(df)
        
        # 3. Gérer les valeurs manquantes (CRITIQUE)
        df = self._handle_missing_values(df)
        return df
    
    def _calculate_imputation_stats(self, df):
        """Calculer les stats nécessaires pour l'imputation"""
        self._log("\n Calcul des statistiques d'imputation...")
        
        # Pour chaque feature, calculer médiane/moyenne
        for col in self.numeric_features:
            if col in df.columns:
                self._fit_stats[col] = {
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'nan_count': df[col].isna().sum(),
                    'nan_pct': df[col].isna().sum() / len(df) * 100
                }
        
        # Identifier les features très problématiques
        high_nan_features = [col for col, stats in self._fit_stats.items() 
                            if stats['nan_pct'] > 30]
        
        if high_nan_features:
            self._log(f"\n  Features avec > 30% NaN:")
            for col in high_nan_features:
                pct = self._fit_stats[col]['nan_pct']
                self._log(f"   {col}: {pct:.1f}%")
    
    def _create_lagged_features(self, df):
 
   
 
        if 'lagged_market_forward_excess_returns' not in df.columns:
            self._log("\n Création des lagged features pour le train set...")
            
     
            if 'market_forward_excess_returns' in df.columns:
                df['lagged_market_forward_excess_returns'] = df['market_forward_excess_returns'].shift(1)
                self._log("    lagged_market_forward_excess_returns créé")
            
            if 'forward_returns' in df.columns:
                df['lagged_forward_returns'] = df['forward_returns'].shift(1)
                self._log("    lagged_forward_returns créé")
            
            if 'risk_free_rate' in df.columns:
                df['lagged_risk_free_rate'] = df['risk_free_rate'].shift(1)
                self._log("    lagged_risk_free_rate créé")
        else:
            self._log("\n Lagged features déjà présentes (TEST set)")
        
        return df
    
    def _create_features(self, df):
       
        self._log("\n Feature engineering...")
        
        initial_cols = len(df.columns)
        
        # ========================================
        # 1. LAGGED FEATURES (TRÈS IMPORTANTES !)
        # ========================================
        
        if 'lagged_market_forward_excess_returns' in df.columns:
            self._log("    Creating lagged target features...")
            
            # La target d'hier (LA PLUS IMPORTANTE)
            df['feat_lagged_target'] = df['lagged_market_forward_excess_returns']
            
            # Signe du mouvement d'hier
            df['feat_lagged_target_sign'] = (df['lagged_market_forward_excess_returns'] > 0).astype(int)
            
            # Magnitude du mouvement d'hier
            df['feat_lagged_target_abs'] = df['lagged_market_forward_excess_returns'].abs()
            
            # Lags multiples de la target (2, 3, 5 jours)
            df['feat_lagged_target_2'] = df['lagged_market_forward_excess_returns'].shift(1)
            df['feat_lagged_target_3'] = df['lagged_market_forward_excess_returns'].shift(2)
           # df['feat_lagged_target_5'] = df['lagged_market_forward_excess_returns'].shift(4)
            
            # Rolling statistics de la target
            df['feat_target_rolling_mean_5'] = df['lagged_market_forward_excess_returns'].rolling(5, min_periods=1).mean()
            df['feat_target_rolling_std_5'] = df['lagged_market_forward_excess_returns'].rolling(5, min_periods=1).std()
            df['feat_target_rolling_mean_20'] = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).mean()
            df['feat_target_rolling_std_20'] = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).std()
            
            # Volatilité de la target
            df['feat_target_volatility_5'] = df['lagged_market_forward_excess_returns'].rolling(5, min_periods=1).std()
            df['feat_target_volatility_20'] = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).std()
            
            # Mean reversion
            rolling_mean = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).mean()
            df['feat_mean_reversion'] = df['lagged_market_forward_excess_returns'] - rolling_mean
            
            # Autocorrélation
            df['feat_autocorr'] = df['lagged_market_forward_excess_returns'] * df['feat_lagged_target_2']
            
            # Momentum (produit des 2 derniers mouvements)
            df['feat_momentum_2days'] = df['lagged_market_forward_excess_returns'] * df['feat_lagged_target_2']
            
            # Z-score de la target
            df['feat_target_zscore'] = (df['lagged_market_forward_excess_returns'] - rolling_mean) / (df['feat_target_rolling_std_20'] + 1e-8)
            
            
        
        if 'lagged_forward_returns' in df.columns:
            df['feat_lagged_returns'] = df['lagged_forward_returns']
            
        if 'lagged_risk_free_rate' in df.columns:
            df['feat_lagged_rfr'] = df['lagged_risk_free_rate']

        # ========================================
        #  MOMENTUM FEATURES SIMPLES
        # ========================================


        target_col = 'lagged_market_forward_excess_returns'
        key_cols = ['lagged_forward_returns', 'lagged_risk_free_rate', target_col]
        for col in key_cols:
            if col in df.columns:
                self._log(f"\n    Momentum Features for {col}")
                for window in [5, 10, 20, 60]:
                    # Rolling Mean
                    new_col_mean = f'feat_{col}_rolling_mean_{window}d'
                    df[new_col_mean] = df[col].rolling(window, min_periods=1).mean()
                    
                    # Rolling Std
                    new_col_std = f'feat_{col}_rolling_std_{window}d'
                    df[new_col_std] = df[col].rolling(window, min_periods=1).std()
                    
                    # Rolling Max
                    new_col_max = f'feat_{col}_rolling_max_{window}d'
                    df[new_col_max] = df[col].rolling(window, min_periods=1).max()
                    
                    # Rolling Min
                    new_col_min = f'feat_{col}_rolling_min_{window}d'
                    df[new_col_min] = df[col].rolling(window, min_periods=1).min()
                    
                    # Rolling Skew
                    new_col_skew = f'feat_{col}_rolling_skew_{window}d'
                    df[new_col_skew] = df[col].rolling(window, min_periods=1).skew()
                    
                    # Rolling Kurtosis
                    new_col_kurt = f'feat_{col}_rolling_kurtosis_{window}d'
                    df[new_col_kurt] = df[col].rolling(window, min_periods=1).kurt()
                    

        
        # ========================================
        # 2. VOLATILITY FEATURES (V*)
        # ========================================
        
           
 
        for window in [20, 60]:
            vol_col = f'lagged_forward_returns_rolling_std_{window}d'
            if vol_col in df.columns:
                new_col = f'feat_vol_of_vol_{window}'
                df[new_col] = df[vol_col].rolling(20).std()
        
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) > 1 and col[1:].isdigit()]
        if v_cols:
            
            # Statistiques de base
            df['feat_v_mean'] = df[v_cols].mean(axis=1)
            df['feat_v_std'] = df[v_cols].std(axis=1)
            df['feat_v_max'] = df[v_cols].max(axis=1)
            df['feat_v_min'] = df[v_cols].min(axis=1)
            df['feat_v_range'] = df['feat_v_max'] - df['feat_v_min']
            df['feat_v_median'] = df[v_cols].median(axis=1)
            
            # Volatility regimes
            df['feat_high_vol'] = (df['feat_v_mean'] > df['feat_v_mean'].quantile(0.75)).astype(int)
            df['feat_low_vol'] = (df['feat_v_mean'] < df['feat_v_mean'].quantile(0.25)).astype(int)
            df['feat_extreme_vol'] = (df['feat_high_vol'] | df['feat_low_vol']).astype(int)
            
            # Volatility percentile (position relative)
            df['feat_v_percentile'] = df['feat_v_mean'].rolling(252, min_periods=1).rank(pct=True)
            
            # Changement de volatilité
            df['feat_v_change'] = df['feat_v_mean'] - df['feat_v_mean'].shift(5).mean()
            df['feat_v_pct_change'] = df['feat_v_mean'].pct_change(5).mean()
            
           
        # ========================================
        # 3. MOMENTUM FEATURES (M*)
        # ========================================
        
        m_cols = [col for col in df.columns if col.startswith('M') and len(col) > 1 and col[1:].isdigit()]
        if m_cols:
           
            # Statistiques de base
            df['feat_m_mean'] = df[m_cols].mean(axis=1)
            df['feat_m_sum'] = df[m_cols].sum(axis=1)
            df['feat_m_max'] = df[m_cols].max(axis=1)
            df['feat_m_min'] = df[m_cols].min(axis=1)
            df['feat_m_std'] = df[m_cols].std(axis=1)
            
            # Momentum strength
            df['feat_momentum_strength'] = df[m_cols].abs().sum(axis=1)
            
            # Positive vs negative momentum
            df['feat_positive_momentum'] = (df[m_cols] > 0).sum(axis=1)
            df['feat_negative_momentum'] = (df[m_cols] < 0).sum(axis=1)
            df['feat_momentum_balance'] = df['feat_positive_momentum'] - df['feat_negative_momentum']
            
            # Momentum consistency
            df['feat_momentum_consistency'] = df['feat_momentum_balance'] / (len(m_cols) + 1e-8)
            
            # Momentum change
            df['feat_m_change'] = df['feat_m_mean'] - df['feat_m_mean'].shift(5).mean()
            
            
        
        # ========================================
        # 4. SENTIMENT FEATURES (S*)
        # ========================================
        
        s_cols = [col for col in df.columns if col.startswith('S') and len(col) > 1 and col[1:].isdigit()]
        if s_cols:
            
            
            # Statistiques de base
            df['feat_s_mean'] = df[s_cols].mean(axis=1)
            df['feat_s_std'] = df[s_cols].std(axis=1)
            df['feat_s_max'] = df[s_cols].max(axis=1)
            df['feat_s_min'] = df[s_cols].min(axis=1)
            
            # Sentiment regime
            df['feat_positive_sentiment'] = (df['feat_s_mean'] > 0).astype(int)
            df['feat_extreme_sentiment'] = (df['feat_s_mean'].abs() > df['feat_s_mean'].abs().quantile(0.9)).astype(int)
            
            # Sentiment change
            df['feat_s_change'] = df['feat_s_mean'] - df['feat_s_mean'].shift(5).mean()
            df['feat_s_pct_change'] = df['feat_s_mean'].pct_change(5).mean()
            
           
        
        # ========================================
        # 5. PRICE/VALUATION FEATURES (P*)
        # ========================================
        
        p_cols = [col for col in df.columns if col.startswith('P') and len(col) > 1 and col[1:].isdigit()]
        if p_cols:
           
            
            df['feat_p_mean'] = df[p_cols].mean(axis=1)
            df['feat_p_std'] = df[p_cols].std(axis=1)
            df['feat_p_max'] = df[p_cols].max(axis=1)
            df['feat_p_min'] = df[p_cols].min(axis=1)
            
            # Price change
            df['feat_p_change'] = df['feat_p_mean'] - df['feat_p_mean'].shift(5).mean()
            
            
        
        # ========================================
        # 6. INTEREST RATE FEATURES (I*)
        # ========================================
        
        i_cols = [col for col in df.columns if col.startswith('I') and len(col) > 1 and col[1:].isdigit()]
        if i_cols:
            
            
            df['feat_i_mean'] = df[i_cols].mean(axis=1)
            df['feat_i_spread'] = df[i_cols].max(axis=1) - df[i_cols].min(axis=1)
            df['feat_i_std'] = df[i_cols].std(axis=1)
            
            # Yield curve shape
            if len(i_cols) >= 3:
                df['feat_i_slope'] = df[i_cols[-1]] - df[i_cols[0]]  # Long - Short
            
            # Interest rate change
            df['feat_i_change'] = df['feat_i_mean'] - df['feat_i_mean'].shift(20).mean()
            
            
        
        # ========================================
        # 7. ECONOMIC FEATURES (E*)
        # ========================================
        
        e_cols = [col for col in df.columns if col.startswith('E') and len(col) > 1 and col[1:].isdigit()]
        if e_cols:
            
            
            df['feat_e_mean'] = df[e_cols].mean(axis=1)
            df['feat_e_std'] = df[e_cols].std(axis=1)
            
            # Economic change
            df['feat_e_change'] = df['feat_e_mean'] - df['feat_e_mean'].shift(20).mean()
            
           
        
        # ========================================
        # 8. INTERACTIONS ENTRE GROUPES
        # ========================================

    
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
#
        for col in ['lagged_forward_returns', target_col]:
            if col in df.columns:
                for window in [7, 14, 21]:
                    new_col = f'feat_{col}_rsi_{window}'
                    df[new_col] = calculate_rsi(df[col], window)

     
        for col in ['lagged_forward_returns', target_col]:
            if col in df.columns:
                # MACD = EMA(12) - EMA(26)
                ema_12 = df[col].ewm(span=12, adjust=False).mean()
                ema_26 = df[col].ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                
                new_col = f'feat_{col}_macd'
                df[new_col] = macd
                
                # Signal line = EMA(9) of MACD
                signal = macd.ewm(span=9, adjust=False).mean()
                signal_col = f'feat_{col}_macd_signal'
                df[signal_col] = signal
                
                # MACD Histogram
                hist_col = f'feat_{col}_macd_hist'
                df[hist_col] = macd - signal

      
        
        # Volatilité × Momentum
        if 'feat_v_mean' in df.columns and 'feat_m_mean' in df.columns:
            df['feat_vol_momentum_interaction'] = df['feat_v_mean'] * df['feat_m_mean']
            df['feat_vol_momentum_ratio'] = df['feat_v_mean'] / (df['feat_m_mean'].abs() + 1e-8)
        
        # Sentiment × Volatilité
        if 'feat_s_mean' in df.columns and 'feat_v_mean' in df.columns:
            df['feat_sentiment_vol_interaction'] = df['feat_s_mean'] * df['feat_v_mean']
        
        # Momentum × Target lagged
        if 'feat_m_mean' in df.columns and 'feat_lagged_target' in df.columns:
            df['feat_momentum_target_alignment'] = (np.sign(df['feat_m_mean']) == np.sign(df['feat_lagged_target'])).astype(int)
            df['feat_momentum_divergence'] = (np.sign(df['feat_m_mean']) != np.sign(df['feat_lagged_target'])).astype(int)
        
        # Price × Volatility
        if 'feat_p_mean' in df.columns and 'feat_v_mean' in df.columns:
            df['feat_price_vol_ratio'] = df['feat_p_mean'] / (df['feat_v_mean'] + 1e-8)
        
        
        
       
        # ═══════════════════════════════════════════════════════════════════════════════
        # 9. TIME-BASED FEATURES
        # ═══════════════════════════════════════════════════════════════════════════════

       
       
        print("="*80)

        time_features = []

        print("\n    Periodic features")
        # Approximer une notion de jour de la semaine / mois
        # En supposant 252 jours de trading par an
        df['feat_day_of_year'] = df['date_id'] % 252
        df['feat_week_of_year'] = (df['date_id'] % 252) // 5
        df['feat_month_of_year'] = (df['date_id'] % 252) // 21

        # Features cycliques
        df['feat_day_sin'] = np.sin(2 * np.pi * df['feat_day_of_year'] / 252)
        df['feat_day_cos'] = np.cos(2 * np.pi * df['feat_day_of_year'] / 252)
        df['feat_month_sin'] = np.sin(2 * np.pi * df['feat_month_of_year'] / 12)
        df['feat_month_cos'] = np.cos(2 * np.pi * df['feat_month_of_year'] / 12)
        
        final_cols = len(df.columns)
        self._log(f"\n Feature engineering terminé: {final_cols - initial_cols} nouvelles features")
        
        return df
    
    #=======================================================================
    # FUNCTIONS MISSING VALUE HANDLING
    #=======================================================================
    def select_features(self, df_train: pd.DataFrame, 
                                    target_col: str,
                                    method: str = "lgb_importance", 
                                    n_features: int = 150) -> List[str]:
        """
        Sélectionne features en utilisant SEULEMENT les données d'entraînement.
        
        Args:
            df_train: DataFrame d'entraînement (subset temporel)
            target_col: Nom de la colonne target
            method: "lgb_importance" ou "correlation"
            n_features: Nombre de features à sélectionner
        
        Returns:
            Liste des features sélectionnées
        """
        # ✅ CORRECTION: Exclure TOUTES les colonnes interdites
        exclude_cols = [
            "date_id", 
            target_col,
            "forward_returns",           # ❌ Future data
            "risk_free_rate",            # ❌ Future data
            "market_forward_excess_returns",  # ❌ Target
            "is_scored"
        ]
        
        candidate_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        X = df_train[candidate_cols].fillna(0)
        y = df_train[target_col].values

        if method == "lgb_importance":
            model = lgb.LGBMRegressor(
                random_state=42, 
                n_estimators=300, 
                learning_rate=0.05,
                verbosity=-1
            )
            model.fit(X, y)
            
            imp = pd.DataFrame({
                "feature": candidate_cols, 
                "importance": model.feature_importances_
            })
            imp = imp.sort_values("importance", ascending=False)
            selected = imp.head(n_features)["feature"].tolist()
            
        elif method == "correlation":
            corrs = []
            for col in candidate_cols:
                s = df_train[[col, target_col]].dropna()
                if len(s) > 50:
                    corr = abs(s[col].corr(s[target_col]))
                    corrs.append((col, corr))
            
            corrs.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in corrs[:n_features]]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ Selected {len(selected)} features using {method}")
        
        return selected



  
    



    def _handle_missing_values(self, df, nan_threshold=0.30):
        # 1. Drop colonnes trop vides
        nan_ratio = df.isna().mean()
        cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index
        df = df.drop(columns=cols_to_drop)
        
        # 2. Colonnes numériques uniquement
        num_cols = df.select_dtypes(include=['float64', 'float32', 'int64']).columns
        
        # 3. Remplacer inf / -inf
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        
        # 4. Imputation par médiane
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        return df
   


    
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
                    'missing_pct': df[col].isna().sum() / len(df) * 100,
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
    
     # Charger les données
    train = pd.read_csv('train.csv')
    print(f"\nTrain shape: {train.shape}")
    test = pd.read_csv('test.csv')

    print(f"Test shape: {test.shape}")


    # Test Feature Engineer
    print("\nTest FeatureEngineer:")
    print("-" * 40)
    
    fe = FeatureEngineer(verbose=True)
    
    # Créer toutes les features
    df_enhanced = fe.hull_transform(train)
    
    print(f"\n✓ DataFrame final: {df_enhanced.shape}")
    #print(f"✓ Nouvelles features: {len(fe.feature_names)}")
    
    # Sélection de features
    selected = fe.select_features(df_enhanced, 'market_forward_excess_returns', 
                                  method='correlation', n_features=120)
    print(f"\n✓ Top 10 features sélectionnées:")
    for i, feat in enumerate(selected, 1):
        print(f"   {i}. {feat}")
    
    print("\n" + "="*80)
    print("✓ Tests Feature Engineering réussis")
    print("="*80)
