
"""
 Hull Tactical Challenge
Integrates the features tagged CRITIQUES + intelligent management of NaNs
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class HullPreprocessorV2:
    """
    Preprocessing pour Hull Tactical Challenge
    
    Key Features:
- Automatic creation of lagged features for TRAIN
- Intelligent management of NaNs by feature type
- Advanced feature engineering
- Automatic validation and logging
    """
    
    def __init__(self, verbose=True):
        self.feature_names = None
        self.numeric_features = None
        self.verbose = verbose
        self._fit_stats = {}  # To store imputation stats
        
    def _log(self, message):
        """Logger if verbose=True"""
        if self.verbose:
            print(message)
        
    def fit(self, df):
        """Learn about columns and types"""
        self._log("\n" + "="*80)
        self._log("FITTING PREPROCESSOR")
        self._log("="*80)
        
        # Identify numeric features (exclude target and metadata)
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        
        self.numeric_features = [col for col in df.columns 
                                if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        self._log(f"✓ feature numeric: {len(self.numeric_features)}")
        
        # Calculate statistics for imputation
        self._calculate_imputation_stats(df)
        
        
        return self
    
    def _calculate_imputation_stats(self, df):
        """Calculate the stats needed for imputation"""
        self._log("\n Calculate the stats for imputation")
        
       # For each feature, calculate median/mean
        for col in self.numeric_features:
            if col in df.columns:
                self._fit_stats[col] = {
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'nan_count': df[col].isnull().sum(),
                    'nan_pct': df[col].isnull().sum() / len(df) * 100
                }
        
        # Identify highly problematic features
        high_nan_features = [col for col, stats in self._fit_stats.items() 
                            if stats['nan_pct'] > 40]
        
        if high_nan_features:
            self._log(f"\n  Features with > 40% NaN:")
            for col in high_nan_features:
                pct = self._fit_stats[col]['nan_pct']
                self._log(f"   {col}: {pct:.1f}%")
    

    def summarize(self, df):

        self._log("\n Summary of dataset after transformation")
        df.info()
        self._log(f"\n Sample data:")
        self._log(df.head())
        self._log(df.describe().T)

    def transform(self, df):
        """Transforming  data"""
        self._log("\n" + "="*80)
        self._log("TRANSFORMING DATA")
        self._log("="*80)
        self._log(f"Input shape: {df.shape}")
        
        df = df.copy()
        
        # 1. Create lagged features if necessary (TRAIN)
        df = self._create_lagged_features(df)
        
        # 2. Feature engineering
        df = self._create_features(df)
        
        # 3. Handle missing values ​​(CRITICAL)
        df = self._handle_missing_values(df)
        
        # 4. Select the final features
        if self.feature_names is None:
            # Première fois : mémoriser les features
            feature_cols = [col for col in df.columns 
                          if col in self.numeric_features or col.startswith('feat_') or col.startswith('lagged_')]
            self.feature_names = feature_cols
            self._log(f"\n✓ Features finales créées: {len(self.feature_names)}")
        
        # Make sure we have all the expected features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        
        X = df[self.feature_names].copy()
        
        # 5.Managing inf and -inf
        inf_count = np.isinf(X.values).sum()
        if inf_count > 0:
            self._log(f"\n {inf_count} Inf values ​​detected, replaced by NaN")
            X = X.replace([np.inf, -np.inf], np.nan)
        
        # 6. Final check
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            self._log(f"\n {nan_count} Remaining NaNs, replaced by 0")
            X = X.fillna(0)
        
        self._log(f"\n✓ Output shape: {X.shape}")
        self._log("="*80)
        
        return X
    
    def _create_lagged_features(self, df):
        """
        Create the lagged features for the TRAIN
             (They already exist in TEST)
        """
        
        if 'lagged_market_forward_excess_returns' not in df.columns:
            self._log("\n Creating lagged features for the train set...")
            
            # WARNING: We can only create lags if we have the source columns
            if 'market_forward_excess_returns' in df.columns:
                df['lagged_market_forward_excess_returns'] = df['market_forward_excess_returns'].shift(1)
                self._log("**    lagged_market_forward_excess_returns created")
            
            if 'forward_returns' in df.columns:
                df['lagged_forward_returns'] = df['forward_returns'].shift(1)
                self._log("**    lagged_forward_returns created")
            
            if 'risk_free_rate' in df.columns:
                df['lagged_risk_free_rate'] = df['risk_free_rate'].shift(1)
                self._log("**    lagged_risk_free_rate créé")
        else:
            self._log("\n Lagged features already present (TEST set)")
        
        return df
    
    def _create_features(self, df):
        """Create additional features based on existing ones"""
        self._log("\n🔧 Feature engineering...")
        
        initial_cols = len(df.columns)
        
        # ========================================
        # 1. LAGGED FEATURES
        # ========================================
        
        if 'lagged_market_forward_excess_returns' in df.columns:
            self._log("***    Creating lagged target features...")
            
            # Yesterday's target (THE MOST IMPORTANT)
            df['feat_lagged_target'] = df['lagged_market_forward_excess_returns']
             
            # Sign of yesterday's movement
            df['feat_lagged_target_sign'] = (df['lagged_market_forward_excess_returns'] > 0).astype(int)
            
            # Magnitude of yesterday's movementr
            df['feat_lagged_target_abs'] = df['lagged_market_forward_excess_returns'].abs()
            
            # Multiple target lags (2, 3, 5 days)
            df['feat_lagged_target_2'] = df['lagged_market_forward_excess_returns'].shift(1)
            df['feat_lagged_target_3'] = df['lagged_market_forward_excess_returns'].shift(2)
            df['feat_lagged_target_5'] = df['lagged_market_forward_excess_returns'].shift(4)
            
            # Target Rolling statistics (5-day, 20-day)
            df['feat_target_rolling_mean_5'] = df['lagged_market_forward_excess_returns'].rolling(5, min_periods=1).mean()
            df['feat_target_rolling_std_5'] = df['lagged_market_forward_excess_returns'].rolling(5, min_periods=1).std()
            df['feat_target_rolling_mean_20'] = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).mean()
            df['feat_target_rolling_std_20'] = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).std()
            
            # Target volatility
            df['feat_target_volatility_5'] = df['lagged_market_forward_excess_returns'].rolling(5, min_periods=1).std()
            df['feat_target_volatility_20'] = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).std()
            
            # Mean reversion
            rolling_mean = df['lagged_market_forward_excess_returns'].rolling(20, min_periods=1).mean()
            df['feat_mean_reversion'] = df['lagged_market_forward_excess_returns'] - rolling_mean
            
            #Autocorrelation effect between past returns and previous signals.
            df['feat_autocorr'] = df['lagged_market_forward_excess_returns'] * df['feat_lagged_target_2']
            
            # Momentum (product of the last 2 movements)
            df['feat_momentum_2days'] = df['lagged_market_forward_excess_returns'] * df['feat_lagged_target_2']
            
            # Z-score of the target
            df['feat_target_zscore'] = (df['lagged_market_forward_excess_returns'] - rolling_mean) / (df['feat_target_rolling_std_20'] + 1e-8)
            
            self._log(f"     16 lagged target features created")
        
        if 'lagged_forward_returns' in df.columns:
            df['feat_lagged_returns'] = df['lagged_forward_returns']
            
        if 'lagged_risk_free_rate' in df.columns:
            df['feat_lagged_rfr'] = df['lagged_risk_free_rate']
        
        # ========================================
        # 2. VOLATILITY FEATURES (V*)
        # ========================================
        
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) > 1 and col[1:].isdigit()]
        if v_cols:
            self._log(f"    Creating volatility features from {len(v_cols)} V* columns...")
            
            # Basic Statistics
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
            
            # Volatility percentile.
            df['feat_v_percentile'] = df['feat_v_mean'].rolling(252, min_periods=20).rank(pct=True)
            
            # Change in volatility
            df['feat_v_change'] = df['feat_v_mean'] - df['feat_v_mean'].shift(5)
            df['feat_v_pct_change'] = df['feat_v_mean'].pct_change(5)
            
            self._log(f"       13 volatility features created")
        
        # ========================================
        # 3. MOMENTUM FEATURES (M*)
        # ========================================
        
        m_cols = [col for col in df.columns if col.startswith('M') and len(col) > 1 and col[1:].isdigit()]
        if m_cols:
            self._log(f"    Creating momentum features from {len(m_cols)} M* columns...")
            
            # Statistiques de base
            df['feat_m_mean'] = df[m_cols].mean(axis=1)
            df['feat_m_sum'] = df[m_cols].sum(axis=1)
            df['feat_m_max'] = df[m_cols].max(axis=1)
            df['feat_m_min'] = df[m_cols].min(axis=1)
            df['feat_m_std'] = df[m_cols].std(axis=1)
            
            # Momentum strength
            df['feat_momentum_strength'] = df[m_cols].abs().sum(axis=1)
            
            # Positive vs. negative momentum
            df['feat_positive_momentum'] = (df[m_cols] > 0).sum(axis=1)
            df['feat_negative_momentum'] = (df[m_cols] < 0).sum(axis=1)
            df['feat_momentum_balance'] = df['feat_positive_momentum'] - df['feat_negative_momentum']
            
            # Momentum consistency
            df['feat_momentum_consistency'] = df['feat_momentum_balance'] / (len(m_cols) + 1e-8)
            
            # Momentum change
            df['feat_m_change'] = df['feat_m_mean'] - df['feat_m_mean'].shift(5)
            
            self._log(f"       11 momentum features created")
        
        # ========================================
        # 4. SENTIMENT FEATURES (S*)
        # ========================================
        
        s_cols = [col for col in df.columns if col.startswith('S') and len(col) > 1 and col[1:].isdigit()]
        if s_cols:
            self._log(f"   📊 Creating sentiment features from {len(s_cols)} S* columns...")
            
            # SBasic statistics
            df['feat_s_mean'] = df[s_cols].mean(axis=1)
            df['feat_s_std'] = df[s_cols].std(axis=1)
            df['feat_s_max'] = df[s_cols].max(axis=1)
            df['feat_s_min'] = df[s_cols].min(axis=1)
            
            # Sentiment regime
            df['feat_positive_sentiment'] = (df['feat_s_mean'] > 0).astype(int)
            df['feat_extreme_sentiment'] = (df['feat_s_mean'].abs() > df['feat_s_mean'].abs().quantile(0.9)).astype(int)
            
            # Sentiment change
            df['feat_s_change'] = df['feat_s_mean'] - df['feat_s_mean'].shift(5)
            df['feat_s_pct_change'] = df['feat_s_mean'].pct_change(5)
            
            self._log(f"       8 sentiment features created")
        
        # ========================================
        # 5. PRICE/VALUATION FEATURES (P*)
        # ========================================
        
        p_cols = [col for col in df.columns if col.startswith('P') and len(col) > 1 and col[1:].isdigit()]
        if p_cols:
            self._log(f"    Creating price features from {len(p_cols)} P* columns...")
            
            df['feat_p_mean'] = df[p_cols].mean(axis=1)
            df['feat_p_std'] = df[p_cols].std(axis=1)
            df['feat_p_max'] = df[p_cols].max(axis=1)
            df['feat_p_min'] = df[p_cols].min(axis=1)
            
            # Price change
            df['feat_p_change'] = df['feat_p_mean'] - df['feat_p_mean'].shift(5)
            
            self._log(f"      ✓ 5 price features created")
        
        # ========================================
        # 6. INTEREST RATE FEATURES (I*)
        # ========================================
        
        i_cols = [col for col in df.columns if col.startswith('I') and len(col) > 1 and col[1:].isdigit()]
        if i_cols:
            self._log(f"    Creating interest rate features from {len(i_cols)} I* columns...")
            
            df['feat_i_mean'] = df[i_cols].mean(axis=1)
            df['feat_i_spread'] = df[i_cols].max(axis=1) - df[i_cols].min(axis=1)
            df['feat_i_std'] = df[i_cols].std(axis=1)
            
            # Yield curve shape
            if len(i_cols) >= 3:
                df['feat_i_slope'] = df[i_cols[-1]] - df[i_cols[0]]  # Long - Short
            
            # Interest rate change
            df['feat_i_change'] = df['feat_i_mean'] - df['feat_i_mean'].shift(20)
            
            self._log(f"      ✓ 5 interest rate features created")
        
        # ========================================
        # 7. ECONOMIC FEATURES (E*)
        # ========================================
        
        e_cols = [col for col in df.columns if col.startswith('E') and len(col) > 1 and col[1:].isdigit()]
        if e_cols:
            self._log(f"    Creating economic features from {len(e_cols)} E* columns...")
            
            df['feat_e_mean'] = df[e_cols].mean(axis=1)
            df['feat_e_std'] = df[e_cols].std(axis=1)
            
            # Economic change
            df['feat_e_change'] = df['feat_e_mean'] - df['feat_e_mean'].shift(20)
            
            self._log(f"       3 economic features créées")
        
        # ========================================
        # 8. INTERACTIONS ENTRE GROUPES
        # ========================================
        
        self._log("    Creating interaction features...")
        
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
        
        self._log(f"       6 interaction features créées")
        
        # ========================================
        # 9. MISSING VALUE INDICATORS
        # ========================================
        
        # V9 has 50% NaN - create an indicator
        if 'V9' in df.columns:
            df['feat_v9_was_missing'] = df['V9'].isnull().astype(int)
            self._log("       V9 missing indicator created")
        
        # Indicators for economic features (often missing)
        for col in e_cols:
            if col in df.columns:
                df[f'feat_{col}_was_missing'] = df[col].isnull().astype(int)
        
        final_cols = len(df.columns)
        self._log(f"\n Feature engineering completed: {final_cols - initial_cols} new features created")
        
        return df
    
    def _handle_missing_values(self, df):
        """
         Intelligently manage missing values
         Different strategies depending on the feature type
        """
        self._log("\n Handling missing values...")
        
        initial_nans = df.isnull().sum().sum()
        self._log(f"  Initial NaNs: {initial_nans}")
        
        # ==========================================
        # 1. FEATURES ÉCONOMIQUES (E*) - FORWARD FILL
        # ==========================================
        e_cols = [col for col in df.columns if col.startswith('E') and len(col) > 1 and col[1:].isdigit()]
        
        if e_cols:
            self._log("    Economic features (E*): forward fill...")
            for col in e_cols:
                if col in df.columns and df[col].isnull().any():
                    #Economic data has a publication delay, so forward fill is appropriate
                    df[col] = df[col].fillna(method='ffill')
                    
                    #If still NaNs at the beginning, use the mean of the fit
                    if df[col].isnull().any():
                        fill_value = self._fit_stats.get(col, {}).get('mean', df[col].mean())
                        df[col] = df[col].fillna(fill_value)
        
        # ==========================================
        # 2. FEATURES DE VOLATILITÉ (V*)
        # ==========================================
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) > 1 and col[1:].isdigit()]
        
        # Special processing for V9 (50% NaN)
        if 'V9' in df.columns and df['V9'].isnull().any():
            self._log("   V9 (50% NaN): imputation by average of other V*...")
            
            # Impute with the average of the other V*
            other_v_cols = [col for col in v_cols if col != 'V9' and col in df.columns and not df[col].isnull().all()]
            if other_v_cols:
                df['V9'] = df['V9'].fillna(df[other_v_cols].mean(axis=1))
            
            # If still NaNs, use the median of the fit
            if df['V9'].isnull().any():
                fill_value = self._fit_stats.get('V9', {}).get('median', df['V9'].median())
                df['V9'] = df['V9'].fillna(fill_value)
        
        # Others V*
        if v_cols:
            self._log("    Other V* features: forward fill + median...")
            for col in v_cols:
                if col != 'V9' and col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(method='ffill')
                    
                    if df[col].isnull().any():
                        fill_value = self._fit_stats.get(col, {}).get('median', df[col].median())
                        df[col] = df[col].fillna(fill_value)
        
        # ==========================================
        # 3. OTHERS FEATURES - STANDARD STRATEGY
        # ==========================================
        other_prefixes = ['M', 'P', 'S', 'I', 'D']
        
        for prefix in other_prefixes:
            cols = [col for col in df.columns if col.startswith(prefix) and len(col) > 1 and col[1:].isdigit()]
            
            if cols:
                self._log(f"    {prefix}* features: forward/backward fill + median...")
                for col in cols:
                    if col in df.columns and df[col].isnull().any():
                        # Forward fill
                        df[col] = df[col].fillna(method='ffill')
                        
                        # Backward fill (for leading NaNs)
                        df[col] = df[col].fillna(method='bfill')
                        
                        # Fallback: median of fit or current median
                        if df[col].isnull().any():
                            fill_value = self._fit_stats.get(col, {}).get('median', df[col].median())
                            if pd.isna(fill_value):
                                fill_value = 0
                            df[col] = df[col].fillna(fill_value)
        
        # ==========================================
        # 4.FEATURES CREATED (feat_*) - ADAPTED STRATEGY
        # ==========================================
        feat_cols = [col for col in df.columns if col.startswith('feat_')]
        
        if feat_cols:
            self._log(f"    Created features (feat_*): intelligent fill...")
            for col in feat_cols:
                if col in df.columns and df[col].isnull().any():
                    #Rolling features can have NaNs at the beginning
                    if 'rolling' in col or 'volatility' in col:
                        # Forward fill for rolling features
                        df[col] = df[col].fillna(method='ffill')
                    
                    #Difference features can have NaNs at the beginning
                    if 'change' in col or 'diff' in col:
                        df[col] = df[col].fillna(0)
                    
                    # General fallback: median of fit or 0
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(0)
        
        # ==========================================
        # 5. LAGGED FEATURES - FORWARD FILL
        # ==========================================
        lagged_cols = [col for col in df.columns if col.startswith('lagged_')]
        
        if lagged_cols:
            self._log(f"    Lagged features: forward fill...")
            for col in lagged_cols:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(method='ffill')
                    
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(0)
        
        # ==========================================
        # 6. Final check and forced fill
        # ==========================================
        remaining_nans = df.isnull().sum().sum()
        
        if remaining_nans > 0:
            self._log(f"\n {remaining_nans} NaN remaining, forced replacement by 0...")
            
            # Show problematic columns
            nan_cols = df.columns[df.isnull().any()].tolist()
            if len(nan_cols) <= 10:
                self._log(f"   Columns: {nan_cols}")
            else:
                self._log(f"   {len(nan_cols)} columns concerned")
            
            # Force all remaining NaNs to 0
            df = df.fillna(0)
        
        final_nans = df.isnull().sum().sum()
        self._log(f"\n✓ Final NaNs: {final_nans} (reduced by {initial_nans})")
        
        return df
    
    def fit_transform(self, df):
        """Fit then transform"""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_summary(self):
        """Return a summary of created features"""
        if self.feature_names is None:
            return "Preprocessor not fitted yet"
        
        summary = {
            'total_features': len(self.feature_names),
            'lagged_features': len([f for f in self.feature_names if f.startswith('lagged_')]),
            'created_features': len([f for f in self.feature_names if f.startswith('feat_')]),
            'original_features': len([f for f in self.feature_names if not f.startswith(('feat_', 'lagged_'))]),
            'feature_groups': {}
        }
        
        # Count by group
        for prefix in ['V', 'M', 'S', 'P', 'I', 'E', 'D']:
            count = len([f for f in self.feature_names if f.startswith(prefix)])
            if count > 0:
                summary['feature_groups'][prefix] = count
        
        return summary




        