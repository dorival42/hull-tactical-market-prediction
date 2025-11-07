"""
═══════════════════════════════════════════════════════════════════════════════
HULL TACTICAL - MARKET PREDICTION
EXPLORATORY DATA ANALYSIS (EDA) COMPLET
═══════════════════════════════════════════════════════════════════════════════

Ce notebook contient une analyse exploratoire complète des données :
1. Vue d'ensemble des données
2. Analyse de la série temporelle
3. Analyse des valeurs manquantes
4. Analyse statistique par catégorie de features
5. Analyse de la target (market_forward_excess_returns)
6. Corrélations et relations entre variables
7. Détection d'anomalies
8. Insights et recommandations

Auteur: Analyse EDA Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("HULL TACTICAL - MARKET PREDICTION - EDA COMPLET")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET VUE D'ENSEMBLE DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("1. CHARGEMENT ET VUE D'ENSEMBLE DES DONNÉES")
print("="*80)

# Charger les données
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\n DIMENSIONS DES DONNÉES:")
print(f"   Train set: {train.shape[0]:,} lignes × {train.shape[1]} colonnes")
print(f"   Test set:  {test.shape[0]:,} lignes × {test.shape[1]} colonnes")

# Informations générales
print(f"\n PÉRIODE COUVERTE:")
print(f"   Premier date_id: {train['date_id'].min()}")
print(f"   Dernier date_id: {train['date_id'].max()}")
print(f"   Nombre de jours de trading: {train.shape[0]:,}")

# Catégories de features
feature_categories = {
    'D': 'Dummy/Binary',
    'E': 'Macro Economic',
    'I': 'Interest Rate',
    'M': 'Market Dynamics',
    'P': 'Price/Valuation',
    'S': 'Sentiment',
    'V': 'Volatility',
    'MOM': 'Momentum'
}

print(f"\n CATÉGORIES DE FEATURES:")
for prefix, category in feature_categories.items():
    features = [col for col in train.columns if col.startswith(prefix) and col not in ['date_id']]
    print(f"   {prefix:4s} ({category:20s}): {len(features):2d} features")

# Colonnes target
target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
print(f"\n COLONNES TARGET:")
for col in target_cols:
    if col in train.columns:
        print(f"   ✓ {col}")

# Premiers et derniers échantillons
print(f"\n APERÇU DES PREMIÈRES LIGNES:")
print(train.head(3))

print(f"\n APERÇU DES DERNIÈRES LIGNES:")
print(train.tail(3))

# Types de données
print(f"\n TYPES DE DONNÉES:")
print(train.dtypes.value_counts())

# Mémoire utilisée
print(f"\n UTILISATION MÉMOIRE:")
print(f"   Train set: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"   Test set:  {test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSE DES VALEURS MANQUANTES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("2. ANALYSE DES VALEURS MANQUANTES")
print("="*80)

# Calculer les valeurs manquantes
missing_stats = pd.DataFrame({
    'Missing_Count': train.isnull().sum(),
    'Missing_Percent': (train.isnull().sum() / len(train) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print(f"\n  STATISTIQUES DES VALEURS MANQUANTES:")
print(f"   Colonnes avec valeurs manquantes: {len(missing_stats)}/{train.shape[1]}")
print(f"   Total de valeurs manquantes: {train.isnull().sum().sum():,}")

if len(missing_stats) > 0:
    print(f"\n   Top 10 colonnes avec le plus de valeurs manquantes:")
    print(missing_stats.head(10))

# Évolution des valeurs manquantes dans le temps
print(f"\n ÉVOLUTION TEMPORELLE DES VALEURS MANQUANTES:")
missing_by_date = train.isnull().sum(axis=1)
print(f"   Moyenne par date: {missing_by_date.mean():.2f}")
print(f"   Médiane par date: {missing_by_date.median():.2f}")
print(f"   Maximum par date: {missing_by_date.max()}")
print(f"   Minimum par date: {missing_by_date.min()}")

# Analyser par périodes
n_periods = 10
period_size = len(train) // n_periods
print(f"\n   Analyse par périodes (divisée en {n_periods} blocs):")

for i in range(n_periods):
    start_idx = i * period_size
    end_idx = min((i + 1) * period_size, len(train))
    period_data = train.iloc[start_idx:end_idx]
    missing_pct = (period_data.isnull().sum().sum() / (period_data.shape[0] * period_data.shape[1]) * 100)
    date_range = f"date_id {period_data['date_id'].min()}-{period_data['date_id'].max()}"
    print(f"   Période {i+1:2d} ({date_range:25s}): {missing_pct:5.2f}% manquant")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSE DE LA TARGET (market_forward_excess_returns)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("3. ANALYSE DE LA TARGET: market_forward_excess_returns")
print("="*80)

target = 'market_forward_excess_returns'
target_data = train[target].dropna()

print(f"\n STATISTIQUES DESCRIPTIVES:")
print(f"   Count:        {target_data.count():,}")
print(f"   Mean:         {target_data.mean():.6f}")
print(f"   Std:          {target_data.std():.6f}")
print(f"   Min:          {target_data.min():.6f}")
print(f"   25%:          {target_data.quantile(0.25):.6f}")
print(f"   50% (Median): {target_data.quantile(0.50):.6f}")
print(f"   75%:          {target_data.quantile(0.75):.6f}")
print(f"   Max:          {target_data.max():.6f}")

print(f"\n ANALYSE DE DISTRIBUTION:")
print(f"   Skewness:     {target_data.skew():.4f}")
print(f"   Kurtosis:     {target_data.kurtosis():.4f}")

# Test de normalité
statistic, p_value = stats.shapiro(target_data.sample(min(5000, len(target_data))))
print(f"\n TEST DE NORMALITÉ (Shapiro-Wilk sur échantillon):")
print(f"   Statistic:    {statistic:.6f}")
print(f"   P-value:      {p_value:.6f}")
print(f"   Distribution normale? {'OUI' if p_value > 0.05 else 'NON'}")

# Analyse des valeurs extrêmes
print(f"\n VALEURS EXTRÊMES:")
print(f"   Valeurs > 2 std:  {(target_data > target_data.mean() + 2*target_data.std()).sum()} ({(target_data > target_data.mean() + 2*target_data.std()).sum()/len(target_data)*100:.2f}%)")
print(f"   Valeurs < -2 std: {(target_data < target_data.mean() - 2*target_data.std()).sum()} ({(target_data < target_data.mean() - 2*target_data.std()).sum()/len(target_data)*100:.2f}%)")

# Périodes positives vs négatives
print(f"\n RENDEMENTS POSITIFS VS NÉGATIFS:")
positive = (target_data > 0).sum()
negative = (target_data < 0).sum()
neutral = (target_data == 0).sum()
print(f"   Positifs: {positive:,} ({positive/len(target_data)*100:.2f}%)")
print(f"   Négatifs: {negative:,} ({negative/len(target_data)*100:.2f}%)")
print(f"   Neutres:  {neutral:,} ({neutral/len(target_data)*100:.2f}%)")

# Analyse forward_returns et risk_free_rate
print(f"\n ANALYSE DES COMPOSANTES:")
if 'forward_returns' in train.columns:
    fr = train['forward_returns'].dropna()
    print(f"   forward_returns:")
    print(f"      Mean: {fr.mean():.6f}")
    print(f"      Std:  {fr.std():.6f}")

if 'risk_free_rate' in train.columns:
    rfr = train['risk_free_rate'].dropna()
    print(f"   risk_free_rate:")
    print(f"      Mean: {rfr.mean():.6f}")
    print(f"      Std:  {rfr.std():.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYSE PAR CATÉGORIE DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("4. ANALYSE PAR CATÉGORIE DE FEATURES")
print("="*80)

for prefix, category in feature_categories.items():
    features = [col for col in train.columns if col.startswith(prefix) and col not in ['date_id']]
    
    if len(features) == 0:
        continue
    
    print(f"\n{'─'*80}")
    print(f"📊 CATÉGORIE: {category} ({prefix}*) - {len(features)} features")
    print(f"{'─'*80}")
    
    category_data = train[features]
    
    # Statistiques générales
    print(f"\n   Statistiques générales:")
    print(f"      Valeurs manquantes: {category_data.isnull().sum().sum():,} ({category_data.isnull().sum().sum()/(len(train)*len(features))*100:.2f}%)")
    print(f"      Features complètes: {(category_data.isnull().sum() == 0).sum()}/{len(features)}")
    
    # Statistiques descriptives pour les features numériques
    numeric_features = category_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) > 0:
        print(f"\n   Statistiques descriptives (features numériques):")
        stats_df = category_data[numeric_features].describe().T
        print(f"      Mean range:  [{stats_df['mean'].min():.4f}, {stats_df['mean'].max():.4f}]")
        print(f"      Std range:   [{stats_df['std'].min():.4f}, {stats_df['std'].max():.4f}]")
        print(f"      Min value:   {stats_df['min'].min():.4f}")
        print(f"      Max value:   {stats_df['max'].max():.4f}")
    
    # Pour les features binaires (D*)
    if prefix == 'D':
        print(f"\n   Distribution des valeurs (features binaires):")
        for feat in features[:5]:  # Limiter à 5 features
            value_counts = train[feat].value_counts()
            print(f"      {feat}: {dict(value_counts)}")
    
    # Corrélation avec la target
    if target in train.columns:
        correlations = []
        for feat in numeric_features:
            if feat in train.columns:
                corr = train[[feat, target]].dropna().corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations.append((feat, corr))
        
        if len(correlations) > 0:
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n   Top 5 corrélations avec {target}:")
            for feat, corr in correlations[:5]:
                print(f"      {feat:10s}: {corr:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANALYSE DE SÉRIES TEMPORELLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("5. ANALYSE DE SÉRIES TEMPORELLES")
print("="*80)

# Créer une copie avec index temporel
train_ts = train.copy()
train_ts['date_id'] = train_ts['date_id'].astype(int)
train_ts = train_ts.sort_values('date_id')

print(f"\n ANALYSE TEMPORELLE DE LA TARGET:")

# Rolling statistics
windows = [5, 20, 60, 252]  # ~1 semaine, 1 mois, 3 mois, 1 an
print(f"\n   Rolling statistics (fenêtres: {windows}):")

for window in windows:
    if len(train_ts) >= window:
        rolling_mean = train_ts[target].rolling(window).mean()
        rolling_std = train_ts[target].rolling(window).std()
        
        print(f"\n   Window {window:3d} jours:")
        print(f"      Mean of rolling means: {rolling_mean.mean():.6f}")
        print(f"      Std of rolling means:  {rolling_mean.std():.6f}")
        print(f"      Mean of rolling stds:  {rolling_std.mean():.6f}")

# Autocorrélation
print(f"\n AUTOCORRÉLATION:")
lags = [1, 5, 10, 20, 60]
print(f"   Lags analysés: {lags}")

for lag in lags:
    if len(train_ts) > lag:
        autocorr = train_ts[target].autocorr(lag=lag)
        print(f"   Lag {lag:2d}: {autocorr:+.4f}")

# Stationnarité (ADF test)
from scipy import stats as sp_stats

print(f"\n TEST DE STATIONNARITÉ:")
target_clean = train_ts[target].dropna()
if len(target_clean) > 50:
    # Test simple: variance dans le temps
    n_splits = 5
    split_size = len(target_clean) // n_splits
    variances = []
    
    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size
        var = target_clean.iloc[start:end].var()
        variances.append(var)
    
    print(f"   Variance par période:")
    for i, var in enumerate(variances):
        print(f"      Période {i+1}: {var:.6f}")
    
    var_of_vars = np.var(variances)
    print(f"   Variance des variances: {var_of_vars:.6f}")
    print(f"   Série stationnaire? {'Probablement OUI' if var_of_vars < 0.0001 else 'Probablement NON'}")

# Trends
print(f"\n ANALYSE DE TENDANCES:")
# Diviser en quartiles temporels
n_quartiles = 4
quartile_size = len(train_ts) // n_quartiles

for i in range(n_quartiles):
    start = i * quartile_size
    end = (i + 1) * quartile_size if i < n_quartiles - 1 else len(train_ts)
    quartile_data = train_ts[target].iloc[start:end]
    
    print(f"   Quartile {i+1} (date_id {train_ts['date_id'].iloc[start]}-{train_ts['date_id'].iloc[end-1]}):")
    print(f"      Mean: {quartile_data.mean():+.6f}")
    print(f"      Std:  {quartile_data.std():.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALYSE DE CORRÉLATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("6. ANALYSE DE CORRÉLATION GLOBALE")
print("="*80)

# Sélectionner un échantillon de features pour la corrélation
all_features = [col for col in train.columns if col not in ['date_id'] + target_cols]

# Prendre un échantillon si trop de features
max_features_for_corr = 30
if len(all_features) > max_features_for_corr:
    # Sélectionner les features avec le moins de valeurs manquantes
    missing_counts = train[all_features].isnull().sum().sort_values()
    sample_features = missing_counts.head(max_features_for_corr).index.tolist()
else:
    sample_features = all_features

print(f"\n MATRICE DE CORRÉLATION:")
print(f"   Nombre de features analysées: {len(sample_features)}")
print(f"   (Échantillon des features avec le moins de valeurs manquantes)")

# Calculer la matrice de corrélation
corr_data = train[sample_features + [target]].dropna()
print(f"   Données disponibles pour corrélation: {len(corr_data):,} lignes")

if len(corr_data) > 100:
    corr_matrix = corr_data.corr()
    
    # Top corrélations avec la target
    target_corr = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
    
    print(f"\n   Top 15 corrélations absolues avec {target}:")
    for i, (feat, corr) in enumerate(target_corr.head(15).items(), 1):
        original_corr = corr_matrix[target][feat]
        print(f"   {i:2d}. {feat:15s}: {original_corr:+.4f} (|r| = {corr:.4f})")
    
    # Features fortement corrélées entre elles
    print(f"\n   Features fortement corrélées entre elles (|r| > 0.8):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if len(high_corr_pairs) > 0:
        for feat1, feat2, corr in high_corr_pairs[:10]:
            print(f"      {feat1:15s} <-> {feat2:15s}: {corr:+.4f}")
    else:
        print(f"      Aucune paire de features avec |r| > 0.8")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. DÉTECTION D'ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("7. DÉTECTION D'ANOMALIES")
print("="*80)

print(f"\n ANALYSE DES VALEURS EXTRÊMES (Target):")

# Z-score method
z_scores = np.abs(stats.zscore(target_data))
outliers_zscore = (z_scores > 3).sum()
print(f"   Outliers (|Z-score| > 3): {outliers_zscore} ({outliers_zscore/len(target_data)*100:.2f}%)")

# IQR method
Q1 = target_data.quantile(0.25)
Q3 = target_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = ((target_data < lower_bound) | (target_data > upper_bound)).sum()
print(f"   Outliers (IQR method):    {outliers_iqr} ({outliers_iqr/len(target_data)*100:.2f}%)")

# Valeurs extrêmes
print(f"\n   Top 5 valeurs les plus élevées:")
top_5_high = target_data.nlargest(5)
for idx, val in top_5_high.items():
    date_id = train.loc[idx, 'date_id']
    print(f"      date_id {date_id:5d}: {val:+.6f}")

print(f"\n   Top 5 valeurs les plus basses:")
top_5_low = target_data.nsmallest(5)
for idx, val in top_5_low.items():
    date_id = train.loc[idx, 'date_id']
    print(f"      date_id {date_id:5d}: {val:+.6f}")

# Anomalies dans les features
print(f"\n ANOMALIES DANS LES FEATURES:")
anomaly_summary = []

for col in all_features[:20]:  # Limiter aux 20 premières features
    col_data = train[col].dropna()
    if len(col_data) > 0 and col_data.dtype in [np.float64, np.int64]:
        z_scores = np.abs(stats.zscore(col_data))
        n_outliers = (z_scores > 3).sum()
        if n_outliers > 0:
            anomaly_summary.append((col, n_outliers, n_outliers/len(col_data)*100))

if len(anomaly_summary) > 0:
    anomaly_summary.sort(key=lambda x: x[1], reverse=True)
    print(f"   Features avec le plus d'outliers (top 10):")
    for feat, count, pct in anomaly_summary[:10]:
        print(f"      {feat:15s}: {count:5d} outliers ({pct:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ANALYSE DES PATTERNS TEMPORELS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("8. ANALYSE DES PATTERNS TEMPORELS")
print("="*80)

# Volatility clustering
print(f"\n VOLATILITY CLUSTERING:")
abs_returns = train_ts[target].abs()
windows = [5, 20, 60]

for window in windows:
    if len(train_ts) >= window:
        rolling_vol = abs_returns.rolling(window).std()
        
        # Autocorrélation de la volatilité
        vol_autocorr = rolling_vol.autocorr(lag=1)
        print(f"   Window {window:2d} - Autocorr(1) of volatility: {vol_autocorr:+.4f}")

# Périodes de forte volatilité
print(f"\n PÉRIODES DE FORTE VOLATILITÉ:")
rolling_vol_20 = abs_returns.rolling(20).std()
high_vol_threshold = rolling_vol_20.quantile(0.95)
high_vol_periods = rolling_vol_20[rolling_vol_20 > high_vol_threshold]

print(f"   Seuil (95e percentile): {high_vol_threshold:.6f}")
print(f"   Nombre de périodes à haute volatilité: {len(high_vol_periods)}")

if len(high_vol_periods) > 0:
    print(f"   Exemples de date_ids avec haute volatilité:")
    for idx in high_vol_periods.index[:5]:
        date_id = train_ts.loc[idx, 'date_id']
        vol = rolling_vol_20.loc[idx]
        print(f"      date_id {date_id:5d}: volatilité = {vol:.6f}")

# Momentum
print(f"\n📈 ANALYSE DE MOMENTUM:")
momentum_windows = [5, 10, 20]

for window in momentum_windows:
    if len(train_ts) >= window:
        momentum = train_ts[target].rolling(window).mean()
        
        positive_momentum = (momentum > 0).sum()
        negative_momentum = (momentum < 0).sum()
        
        print(f"   Window {window:2d}:")
        print(f"      Périodes avec momentum positif: {positive_momentum} ({positive_momentum/(positive_momentum+negative_momentum)*100:.1f}%)")
        print(f"      Périodes avec momentum négatif: {negative_momentum} ({negative_momentum/(positive_momentum+negative_momentum)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. RÉSUMÉ ET INSIGHTS CLÉS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("9. RÉSUMÉ ET INSIGHTS CLÉS")
print("="*80)

print(f"\n🎯 INSIGHTS PRINCIPAUX:")

# 1. Valeurs manquantes
missing_pct_total = train.isnull().sum().sum() / (train.shape[0] * train.shape[1]) * 100
print(f"\n1. VALEURS MANQUANTES:")
print(f"    {missing_pct_total:.2f}% des données sont manquantes")
print(f"    Les premières périodes ont beaucoup plus de données manquantes")
print(f"    Recommandation: Utiliser seulement les données récentes ou imputer intelligemment")

# 2. Target
print(f"\n2. TARGET (market_forward_excess_returns):")
print(f"    Distribution proche de zéro (mean ≈ {target_data.mean():.6f})")
print(f"   {'NON' if abs(target_data.skew()) > 0.5 else 'RELATIVEMENT'} symétrique (skewness = {target_data.skew():.2f})")
print(f"    Présence d'outliers: {outliers_zscore} valeurs extrêmes")
print(f"    {positive/len(target_data)*100:.1f}% des périodes sont positives")

# 3. Corrélations
if len(corr_data) > 100:
    max_corr = target_corr.iloc[0]
    max_corr_feat = target_corr.index[0]
    print(f"\n3. CORRÉLATIONS:")
    print(f"    Feature la plus corrélée: {max_corr_feat} (|r| = {max_corr:.4f})")
    print(f"    {'Beaucoup' if len(high_corr_pairs) > 10 else 'Peu'} de features fortement corrélées entre elles")
    print(f"    Recommandation: {'Attention à la multicolinéarité' if len(high_corr_pairs) > 10 else 'Multicolinéarité limitée'}")

# 4. Séries temporelles
print(f"\n4. PROPRIÉTÉS TEMPORELLES:")
autocorr_lag1 = train_ts[target].autocorr(lag=1)
print(f"    Autocorrélation lag-1: {autocorr_lag1:+.4f}")
print(f"    {'Forte' if abs(autocorr_lag1) > 0.1 else 'Faible'} dépendance temporelle")
print(f"    Volatility clustering: {'OUI' if vol_autocorr > 0.3 else 'NON'}")

# 5. Recommandations
print(f"\n5. RECOMMANDATIONS POUR LA MODÉLISATION:")
print(f"    Utiliser une validation walk-forward (respecter l'ordre temporel)")
print(f"    Créer des features de lag pour capturer la dépendance temporelle")
print(f"    Ajouter des rolling statistics (mean, std, etc.)")
print(f"    Gérer les outliers (winsorization ou transformation)")
print(f"    Considérer les modèles robustes aux valeurs manquantes (LightGBM, CatBoost)")
print(f"    Tester des modèles de séries temporelles (ARIMA, GARCH)")

print("\n" + "="*80)
print("FIN DE L'ANALYSE EXPLORATOIRE")
print("="*80)

print("\n📊 L'analyse est terminée!")
print("   Les statistiques complètes ont été calculées.")
print("   Consultez les résultats ci-dessus pour vos insights.")