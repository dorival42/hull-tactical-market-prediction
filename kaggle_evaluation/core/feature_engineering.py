"""
═══════════════════════════════════════════════════════════════════════════════
HULL TACTICAL - ADVANCED FEATURE ENGINEERING
═══════════════════════════════════════════════════════════════════════════════

Ce script crée des features avancées pour améliorer les prédictions :
1. Lag Features (1, 2, 3, 5, 10, 20 jours)
2. Rolling Statistics (mean, std, min, max sur différentes fenêtres)
3. Momentum Indicators (simple, exponential, rate of change)
4. Volatility Features (realized vol, GARCH-like, range-based)
5. Trend Features (moving average crossovers, trend strength)
6. Interaction Features (produits, ratios entre features)
7. Technical Indicators (RSI, MACD, Bollinger Bands)

Auteur: Feature Engineering Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HULL TACTICAL - ADVANCED FEATURE ENGINEERING")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("1. CHARGEMENT DES DONNÉES")
print("="*80)

train = pd.read_csv('train.csv')
print(f"\n✓ Données chargées: {train.shape[0]:,} lignes × {train.shape[1]} colonnes")

# Utiliser les données récentes (date_id >= 7000)
CUTOFF_DATE = 7000
train_clean = train[train['date_id'] >= CUTOFF_DATE].copy()
train_clean = train_clean.sort_values('date_id').reset_index(drop=True)
print(f"✓ Données après cutoff: {len(train_clean):,} lignes (date_id >= {CUTOFF_DATE})")

# Colonnes originales
original_cols = train_clean.columns.tolist()
print(f"✓ Features originales: {len(original_cols)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LAG FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("2. CRÉATION DES LAG FEATURES")
print("="*80)

target_col = 'market_forward_excess_returns'
key_cols = ['forward_returns', 'risk_free_rate', target_col]

# Lags à créer
lags = [1, 2, 3, 5, 10, 20]

print(f"\n   Création de lags pour les colonnes clés: {key_cols}")
print(f"   Lags: {lags}")

lag_features = []
for col in key_cols:
    if col in train_clean.columns:
        for lag in lags:
            new_col = f'{col}_lag_{lag}'
            train_clean[new_col] = train_clean[col].shift(lag)
            lag_features.append(new_col)

print(f"\n✓ {len(lag_features)} lag features créées")
print(f"   Exemples: {lag_features[:5]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROLLING STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("3. ROLLING STATISTICS")
print("="*80)

windows = [5, 10, 20, 60]
stats = ['mean', 'std', 'min', 'max']

print(f"\n   Fenêtres: {windows}")
print(f"   Statistiques: {stats}")

rolling_features = []

for col in key_cols:
    if col in train_clean.columns:
        for window in windows:
            for stat in stats:
                new_col = f'{col}_rolling_{stat}_{window}d'
                
                if stat == 'mean':
                    train_clean[new_col] = train_clean[col].rolling(window).mean()
                elif stat == 'std':
                    train_clean[new_col] = train_clean[col].rolling(window).std()
                elif stat == 'min':
                    train_clean[new_col] = train_clean[col].rolling(window).min()
                elif stat == 'max':
                    train_clean[new_col] = train_clean[col].rolling(window).max()
                
                rolling_features.append(new_col)

print(f"\n✓ {len(rolling_features)} rolling features créées")
print(f"   Exemples: {rolling_features[:5]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("4. MOMENTUM INDICATORS")
print("="*80)

momentum_features = []

print("\n   a) Simple Momentum (différence de moyennes)")
for col in key_cols:
    if col in train_clean.columns:
        # Momentum = MA court terme - MA long terme
        for short, long in [(5, 20), (10, 60), (20, 60)]:
            new_col = f'{col}_momentum_{short}_{long}'
            ma_short = train_clean[col].rolling(short).mean()
            ma_long = train_clean[col].rolling(long).mean()
            train_clean[new_col] = ma_short - ma_long
            momentum_features.append(new_col)

print(f"   ✓ {len([f for f in momentum_features if 'momentum_' in f])} momentum simples")

print("\n   b) Rate of Change (ROC)")
for col in key_cols:
    if col in train_clean.columns:
        for period in [5, 10, 20]:
            new_col = f'{col}_roc_{period}'
            train_clean[new_col] = train_clean[col].pct_change(periods=period)
            momentum_features.append(new_col)

print(f"   ✓ {len([f for f in momentum_features if '_roc_' in f])} ROC features")

print("\n   c) Exponential Moving Average (EMA)")
for col in key_cols:
    if col in train_clean.columns:
        for span in [5, 10, 20]:
            new_col = f'{col}_ema_{span}'
            train_clean[new_col] = train_clean[col].ewm(span=span, adjust=False).mean()
            momentum_features.append(new_col)

print(f"   ✓ {len([f for f in momentum_features if '_ema_' in f])} EMA features")

print(f"\n✓ Total: {len(momentum_features)} momentum indicators créés")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. VOLATILITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("5. VOLATILITY FEATURES")
print("="*80)

volatility_features = []

print("\n   a) Realized Volatility")
for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        for window in [5, 10, 20, 60]:
            # Realized volatility = sqrt(sum of squared returns)
            new_col = f'{col}_realized_vol_{window}'
            train_clean[new_col] = train_clean[col].rolling(window).apply(
                lambda x: np.sqrt((x**2).sum())
            )
            volatility_features.append(new_col)

print(f"   ✓ {len([f for f in volatility_features if 'realized_vol' in f])} realized vol features")

print("\n   b) Parkinson Volatility (range-based)")
# Approximation: utiliser min/max des rolling windows
for col in ['forward_returns']:
    if col in train_clean.columns:
        for window in [5, 10, 20]:
            new_col = f'{col}_parkinson_vol_{window}'
            high = train_clean[col].rolling(window).max()
            low = train_clean[col].rolling(window).min()
            # Parkinson estimator
            train_clean[new_col] = np.sqrt((np.log(high/low)**2) / (4 * np.log(2)))
            # Gérer les valeurs infinies
            train_clean[new_col] = train_clean[new_col].replace([np.inf, -np.inf], np.nan)
            volatility_features.append(new_col)

print(f"   ✓ {len([f for f in volatility_features if 'parkinson' in f])} Parkinson vol features")

print("\n   c) Volatility of Volatility")
for window in [20, 60]:
    vol_col = f'forward_returns_rolling_std_{window}d'
    if vol_col in train_clean.columns:
        new_col = f'vol_of_vol_{window}'
        train_clean[new_col] = train_clean[vol_col].rolling(20).std()
        volatility_features.append(new_col)

print(f"   ✓ {len([f for f in volatility_features if 'vol_of_vol' in f])} vol-of-vol features")

print(f"\n✓ Total: {len(volatility_features)} volatility features créées")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TREND FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("6. TREND FEATURES")
print("="*80)

trend_features = []

print("\n   a) Moving Average Crossovers")
for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        # MA Crossover: 1 si MA courte > MA longue, 0 sinon
        for short, long in [(5, 20), (10, 60)]:
            new_col = f'{col}_ma_cross_{short}_{long}'
            ma_short = train_clean[col].rolling(short).mean()
            ma_long = train_clean[col].rolling(long).mean()
            train_clean[new_col] = (ma_short > ma_long).astype(int)
            trend_features.append(new_col)

print(f"   ✓ {len([f for f in trend_features if '_ma_cross_' in f])} MA crossover features")

print("\n   b) Trend Strength")
for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        for window in [10, 20, 60]:
            new_col = f'{col}_trend_strength_{window}'
            # Linear regression slope
            def calc_slope(y):
                if len(y) < 2:
                    return 0
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0] if not np.isnan(y).all() else 0
                return slope
            
            train_clean[new_col] = train_clean[col].rolling(window).apply(calc_slope)
            trend_features.append(new_col)

print(f"   ✓ {len([f for f in trend_features if 'trend_strength' in f])} trend strength features")

print("\n   c) Distance from Moving Averages")
for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        for window in [10, 20, 60]:
            new_col = f'{col}_dist_from_ma_{window}'
            ma = train_clean[col].rolling(window).mean()
            train_clean[new_col] = (train_clean[col] - ma) / ma
            # Gérer les divisions par zéro
            train_clean[new_col] = train_clean[new_col].replace([np.inf, -np.inf], np.nan)
            trend_features.append(new_col)

print(f"   ✓ {len([f for f in trend_features if 'dist_from_ma' in f])} distance features")

print(f"\n✓ Total: {len(trend_features)} trend features créées")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("7. TECHNICAL INDICATORS")
print("="*80)

technical_features = []

print("\n   a) RSI (Relative Strength Index)")
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        for window in [7, 14, 21]:
            new_col = f'{col}_rsi_{window}'
            train_clean[new_col] = calculate_rsi(train_clean[col], window)
            technical_features.append(new_col)

print(f"   ✓ {len([f for f in technical_features if '_rsi_' in f])} RSI features")

print("\n   b) MACD (Moving Average Convergence Divergence)")
for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        # MACD = EMA(12) - EMA(26)
        ema_12 = train_clean[col].ewm(span=12, adjust=False).mean()
        ema_26 = train_clean[col].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        
        new_col = f'{col}_macd'
        train_clean[new_col] = macd
        technical_features.append(new_col)
        
        # Signal line = EMA(9) of MACD
        signal = macd.ewm(span=9, adjust=False).mean()
        signal_col = f'{col}_macd_signal'
        train_clean[signal_col] = signal
        technical_features.append(signal_col)
        
        # MACD Histogram
        hist_col = f'{col}_macd_hist'
        train_clean[hist_col] = macd - signal
        technical_features.append(hist_col)

print(f"   ✓ {len([f for f in technical_features if 'macd' in f])} MACD features")

print("\n   c) Bollinger Bands")
for col in ['forward_returns', target_col]:
    if col in train_clean.columns:
        for window in [20]:
            ma = train_clean[col].rolling(window).mean()
            std = train_clean[col].rolling(window).std()
            
            # Upper Band
            upper_col = f'{col}_bb_upper_{window}'
            train_clean[upper_col] = ma + (2 * std)
            technical_features.append(upper_col)
            
            # Lower Band
            lower_col = f'{col}_bb_lower_{window}'
            train_clean[lower_col] = ma - (2 * std)
            technical_features.append(lower_col)
            
            # BB Width
            width_col = f'{col}_bb_width_{window}'
            train_clean[width_col] = (train_clean[upper_col] - train_clean[lower_col]) / ma
            train_clean[width_col] = train_clean[width_col].replace([np.inf, -np.inf], np.nan)
            technical_features.append(width_col)
            
            # %B (position within bands)
            pct_b_col = f'{col}_bb_pctb_{window}'
            train_clean[pct_b_col] = (train_clean[col] - train_clean[lower_col]) / (train_clean[upper_col] - train_clean[lower_col])
            train_clean[pct_b_col] = train_clean[pct_b_col].replace([np.inf, -np.inf], np.nan)
            technical_features.append(pct_b_col)

print(f"   ✓ {len([f for f in technical_features if '_bb_' in f])} Bollinger Bands features")

print(f"\n✓ Total: {len(technical_features)} technical indicators créés")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("8. INTERACTION FEATURES")
print("="*80)

interaction_features = []

print("\n   a) Produits entre top features")
# Basé sur l'analyse baseline, top features: M4, P6, P4, V13, S5
top_original_features = ['M4', 'P6', 'P4', 'V13', 'S5']
available_top = [f for f in top_original_features if f in train_clean.columns]

if len(available_top) >= 2:
    # Créer quelques interactions clés
    interactions = [
        ('M4', 'V13'),  # Market Dynamics × Volatility
        ('P6', 'S5'),   # Price × Sentiment
        ('P4', 'M4'),   # Price × Market Dynamics
    ]
    
    for feat1, feat2 in interactions:
        if feat1 in train_clean.columns and feat2 in train_clean.columns:
            # Produit
            prod_col = f'{feat1}_x_{feat2}'
            train_clean[prod_col] = train_clean[feat1] * train_clean[feat2]
            interaction_features.append(prod_col)
            
            # Ratio
            ratio_col = f'{feat1}_div_{feat2}'
            train_clean[ratio_col] = train_clean[feat1] / (train_clean[feat2] + 1e-8)
            train_clean[ratio_col] = train_clean[ratio_col].replace([np.inf, -np.inf], np.nan)
            interaction_features.append(ratio_col)

print(f"   ✓ {len(interaction_features)} interaction features créées")

print("\n   b) Interactions avec momentum")
for mom_col in ['forward_returns_momentum_5_20', 'forward_returns_momentum_10_60']:
    if mom_col in train_clean.columns:
        for feat in available_top[:3]:  # Top 3 features
            if feat in train_clean.columns:
                new_col = f'{mom_col}_x_{feat}'
                train_clean[new_col] = train_clean[mom_col] * train_clean[feat]
                interaction_features.append(new_col)

print(f"   ✓ {len([f for f in interaction_features if 'momentum' in f])} momentum interactions")

print(f"\n✓ Total: {len(interaction_features)} interaction features créées")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. TIME-BASED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("9. TIME-BASED FEATURES")
print("="*80)

time_features = []

print("\n   a) Periodic features")
# Approximer une notion de jour de la semaine / mois
# En supposant 252 jours de trading par an
train_clean['day_of_year'] = train_clean['date_id'] % 252
train_clean['week_of_year'] = (train_clean['date_id'] % 252) // 5
train_clean['month_of_year'] = (train_clean['date_id'] % 252) // 21

# Features cycliques
train_clean['day_sin'] = np.sin(2 * np.pi * train_clean['day_of_year'] / 252)
train_clean['day_cos'] = np.cos(2 * np.pi * train_clean['day_of_year'] / 252)
train_clean['month_sin'] = np.sin(2 * np.pi * train_clean['month_of_year'] / 12)
train_clean['month_cos'] = np.cos(2 * np.pi * train_clean['month_of_year'] / 12)

time_features.extend(['day_of_year', 'week_of_year', 'month_of_year', 
                      'day_sin', 'day_cos', 'month_sin', 'month_cos'])

print(f"   ✓ {len(time_features)} time features créées")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RÉSUMÉ ET SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("10. RÉSUMÉ ET SAUVEGARDE")
print("="*80)

# Compter les nouvelles features
all_new_features = (lag_features + rolling_features + momentum_features + 
                   volatility_features + trend_features + technical_features + 
                   interaction_features + time_features)

print(f"\n📊 RÉCAPITULATIF DES FEATURES CRÉÉES:")
print(f"   1. Lag Features:          {len(lag_features):4d}")
print(f"   2. Rolling Statistics:    {len(rolling_features):4d}")
print(f"   3. Momentum Indicators:   {len(momentum_features):4d}")
print(f"   4. Volatility Features:   {len(volatility_features):4d}")
print(f"   5. Trend Features:        {len(trend_features):4d}")
print(f"   6. Technical Indicators:  {len(technical_features):4d}")
print(f"   7. Interaction Features:  {len(interaction_features):4d}")
print(f"   8. Time Features:         {len(time_features):4d}")
print(f"   " + "-"*40)
print(f"   TOTAL NOUVELLES FEATURES: {len(all_new_features):4d}")

print(f"\n📊 DIMENSIONS:")
print(f"   Features originales:  {len(original_cols)}")
print(f"   Features nouvelles:   {len(all_new_features)}")
print(f"   Total:                {train_clean.shape[1]}")
print(f"   Lignes:               {train_clean.shape[0]:,}")

# Vérifier les valeurs manquantes
missing_pct = (train_clean.isnull().sum().sum() / (train_clean.shape[0] * train_clean.shape[1]) * 100)
print(f"\n⚠️  Valeurs manquantes après feature engineering: {missing_pct:.2f}%")

# Sauvegarder
output_file = 'train_with_features.csv'
train_clean.to_csv(output_file, index=False)
print(f"\n💾 Fichier sauvegardé: {output_file}")
print(f"   Taille: {train_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Sauvegarder la liste des features
feature_dict = {
    'lag_features': lag_features,
    'rolling_features': rolling_features,
    'momentum_features': momentum_features,
    'volatility_features': volatility_features,
    'trend_features': trend_features,
    'technical_features': technical_features,
    'interaction_features': interaction_features,
    'time_features': time_features,
    'all_new_features': all_new_features
}

import pickle
with open('feature_dict.pkl', 'wb') as f:
    pickle.dump(feature_dict, f)
print(f"✓ Feature dictionary sauvegardé: feature_dict.pkl")

# Créer un rapport texte
with open('feature_engineering_report.txt', 'w') as f:
    f.write("HULL TACTICAL - FEATURE ENGINEERING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total features créées: {len(all_new_features)}\n\n")
    
    f.write("1. LAG FEATURES ({} features):\n".format(len(lag_features)))
    for feat in lag_features[:10]:
        f.write(f"   - {feat}\n")
    if len(lag_features) > 10:
        f.write(f"   ... et {len(lag_features)-10} autres\n")
    f.write("\n")
    
    f.write("2. ROLLING FEATURES ({} features):\n".format(len(rolling_features)))
    for feat in rolling_features[:10]:
        f.write(f"   - {feat}\n")
    if len(rolling_features) > 10:
        f.write(f"   ... et {len(rolling_features)-10} autres\n")
    f.write("\n")
    
    f.write("3. MOMENTUM FEATURES ({} features):\n".format(len(momentum_features)))
    for feat in momentum_features[:10]:
        f.write(f"   - {feat}\n")
    if len(momentum_features) > 10:
        f.write(f"   ... et {len(momentum_features)-10} autres\n")
    f.write("\n")
    
    f.write("4. VOLATILITY FEATURES ({} features):\n".format(len(volatility_features)))
    for feat in volatility_features:
        f.write(f"   - {feat}\n")
    f.write("\n")
    
    f.write("5. TREND FEATURES ({} features):\n".format(len(trend_features)))
    for feat in trend_features:
        f.write(f"   - {feat}\n")
    f.write("\n")
    
    f.write("6. TECHNICAL FEATURES ({} features):\n".format(len(technical_features)))
    for feat in technical_features:
        f.write(f"   - {feat}\n")
    f.write("\n")
    
    f.write("7. INTERACTION FEATURES ({} features):\n".format(len(interaction_features)))
    for feat in interaction_features:
        f.write(f"   - {feat}\n")
    f.write("\n")
    
    f.write("8. TIME FEATURES ({} features):\n".format(len(time_features)))
    for feat in time_features:
        f.write(f"   - {feat}\n")

print(f"✓ Rapport sauvegardé: feature_engineering_report.txt")

print("\n" + "="*80)
print("FEATURE ENGINEERING TERMINÉ AVEC SUCCÈS")
print("="*80)
print("\n🎯 PROCHAINES ÉTAPES:")
print("   1. Entraîner des modèles avec les nouvelles features")
print("   2. Faire une feature selection")
print("   3. Comparer les performances avec la baseline")