# preprocessor_simple.py - VERSION COMPLÈTE
import pandas as pd
import numpy as np

class HullPreprocessorSimple:
    """Version simplifiée et robuste du preprocessor"""
    
    def __init__(self, verbose=True):
        self.feature_names = None
        self.numeric_features = None
        self.verbose = verbose
        
    def fit(self, df):
        if self.verbose:
            print("\n✓ Fitting SimplePreprocessor")
        
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        
        self.numeric_features = [col for col in df.columns 
                                if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        if self.verbose:
            print(f"✓ {len(self.numeric_features)} features de base")
        
        return self
    
    def transform(self, df):
        df = df.copy()
        
        # 1. Créer lagged features si nécessaire (TRAIN)
        if 'lagged_market_forward_excess_returns' not in df.columns:
            if 'market_forward_excess_returns' in df.columns:
                df['lagged_market_forward_excess_returns'] = df['market_forward_excess_returns'].shift(1)
            if 'forward_returns' in df.columns:
                df['lagged_forward_returns'] = df['forward_returns'].shift(1)
            if 'risk_free_rate' in df.columns:
                df['lagged_risk_free_rate'] = df['risk_free_rate'].shift(1)
        
        # 2. Features simples mais efficaces
        all_features = []
        
        # Features de base
        for col in self.numeric_features:
            if col in df.columns:
                all_features.append(col)
        
        # Lagged features (LES PLUS IMPORTANTES)
        if 'lagged_market_forward_excess_returns' in df.columns:
            all_features.append('lagged_market_forward_excess_returns')
            df['feat_lagged_target'] = df['lagged_market_forward_excess_returns']
            all_features.append('feat_lagged_target')
        
        if 'lagged_forward_returns' in df.columns:
            all_features.append('lagged_forward_returns')
        
        if 'lagged_risk_free_rate' in df.columns:
            all_features.append('lagged_risk_free_rate')
        
        # Moyennes par groupe
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) > 1 and col[1:].isdigit()]
        if v_cols:
            df['feat_v_mean'] = df[v_cols].mean(axis=1)
            all_features.append('feat_v_mean')
        
        m_cols = [col for col in df.columns if col.startswith('M') and len(col) > 1 and col[1:].isdigit()]
        if m_cols:
            df['feat_m_mean'] = df[m_cols].mean(axis=1)
            all_features.append('feat_m_mean')
        
        s_cols = [col for col in df.columns if col.startswith('S') and len(col) > 1 and col[1:].isdigit()]
        if s_cols:
            df['feat_s_mean'] = df[s_cols].mean(axis=1)
            all_features.append('feat_s_mean')
        
        p_cols = [col for col in df.columns if col.startswith('P') and len(col) > 1 and col[1:].isdigit()]
        if p_cols:
            df['feat_p_mean'] = df[p_cols].mean(axis=1)
            all_features.append('feat_p_mean')
        
        # Indicateur V9 missing
        if 'V9' in df.columns:
            df['feat_v9_missing'] = df['V9'].isnull().astype(int)
            all_features.append('feat_v9_missing')
        
        # Mémoriser les features
        if self.feature_names is None:
            self.feature_names = all_features
        
        # Sélectionner et gérer les NaN
        X = df[self.feature_names].copy()
        
        # Remplir les NaN
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(method='ffill').fillna(0)
        
        # Remplacer inf
        X = X.replace([np.inf, -np.inf], 0)
        
        if self.verbose:
            print(f"✓ Output: {X.shape}, {(X.std() > 0).sum()} features avec variance > 0")
        
        return X
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
    def get_feature_summary(self):
        """Retourner un résumé des features créées"""
        if self.feature_names is None:
            return "Preprocessor not fitted yet"
        
        lagged_count = len([f for f in self.feature_names if 'lagged' in f])
        created_count = len([f for f in self.feature_names if f.startswith('feat_')])
        original_count = len(self.feature_names) - lagged_count - created_count
        
        summary = {
            'total_features': len(self.feature_names),
            'lagged_features': lagged_count,
            'created_features': created_count,
            'original_features': original_count,
            'feature_groups': {}
        }
        
        for prefix in ['V', 'M', 'S', 'P', 'I', 'E', 'D']:
            count = len([f for f in self.feature_names if f.startswith(prefix) and len(f) > 1])
            if count > 0:
                summary['feature_groups'][prefix] = count
        
        return summary