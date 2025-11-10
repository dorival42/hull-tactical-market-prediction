"""
═══════════════════════════════════════════════════════════════════════════════
HULL TACTICAL - VISUALISATIONS EDA
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Chargement des données...")
train = pd.read_csv('train.csv')

# Créer un dossier pour les graphiques
import os
os.makedirs('visualizations', exist_ok=True)

print("Génération des visualisations...")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUTION DE LA TARGET
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution de la Target: market_forward_excess_returns', fontsize=16, fontweight='bold')

target = 'market_forward_excess_returns'
target_data = train[target].dropna()

# Histogramme
axes[0, 0].hist(target_data, bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(target_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target_data.mean():.6f}')
axes[0, 0].axvline(target_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target_data.median():.6f}')
axes[0, 0].set_xlabel('market_forward_excess_returns')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Histogramme')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(target_data, vert=True)
axes[0, 1].set_ylabel('market_forward_excess_returns')
axes[0, 1].set_title('Box Plot')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(target_data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normalité)')
axes[1, 0].grid(True, alpha=0.3)

# Distribution cumulative
sorted_data = np.sort(target_data)
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axes[1, 1].plot(sorted_data, cumulative, linewidth=2)
axes[1, 1].set_xlabel('market_forward_excess_returns')
axes[1, 1].set_ylabel('Probabilité cumulative')
axes[1, 1].set_title('Distribution Cumulative')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/01_target_distribution.png', dpi=150, bbox_inches='tight')
print(" Graphique 1: Distribution de la target")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SÉRIE TEMPORELLE DE LA TARGET
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(18, 12))
fig.suptitle('Série Temporelle: market_forward_excess_returns', fontsize=16, fontweight='bold')

# Série brute
axes[0].plot(train['date_id'], train[target], linewidth=0.5, alpha=0.7)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('date_id')
axes[0].set_ylabel('Rendements excédentaires')
axes[0].set_title('Série Temporelle Brute')
axes[0].grid(True, alpha=0.3)

# Rolling mean (20 jours)
rolling_mean_20 = train[target].rolling(window=20).mean()
axes[1].plot(train['date_id'], train[target], linewidth=0.3, alpha=0.4, label='Valeurs brutes')
axes[1].plot(train['date_id'], rolling_mean_20, linewidth=2, color='red', label='Rolling Mean (20 jours)')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('date_id')
axes[1].set_ylabel('Rendements excédentaires')
axes[1].set_title('Rolling Mean (20 jours)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Rolling std (20 jours) - Volatilité
rolling_std_20 = train[target].rolling(window=20).std()
axes[2].plot(train['date_id'], rolling_std_20, linewidth=1, color='purple')
axes[2].fill_between(train['date_id'], 0, rolling_std_20, alpha=0.3, color='purple')
axes[2].set_xlabel('date_id')
axes[2].set_ylabel('Volatilité (Rolling Std)')
axes[2].set_title('Volatilité sur 20 jours')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_time_series.png', dpi=150, bbox_inches='tight')
print(" Graphique 2: Série temporelle")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALEURS MANQUANTES
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('Analyse des Valeurs Manquantes', fontsize=16, fontweight='bold')

# Valeurs manquantes par date
missing_by_date = train.isnull().sum(axis=1)
axes[0].plot(train['date_id'], missing_by_date, linewidth=1)
axes[0].fill_between(train['date_id'], 0, missing_by_date, alpha=0.3)
axes[0].set_xlabel('date_id')
axes[0].set_ylabel('Nombre de valeurs manquantes')
axes[0].set_title('Valeurs Manquantes par Date')
axes[0].grid(True, alpha=0.3)

# Heatmap des valeurs manquantes (échantillon)
sample_size = min(500, len(train))
sample_indices = np.linspace(0, len(train)-1, sample_size, dtype=int)
sample_data = train.iloc[sample_indices]

# Prendre seulement les colonnes avec des valeurs manquantes
cols_with_missing = [col for col in train.columns if train[col].isnull().any()]
missing_matrix = sample_data[cols_with_missing].isnull().astype(int)

im = axes[1].imshow(missing_matrix.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
axes[1].set_xlabel('Échantillon de dates')
axes[1].set_ylabel('Features')
axes[1].set_title(f'Heatmap des Valeurs Manquantes (échantillon de {sample_size} dates)')
plt.colorbar(im, ax=axes[1], label='Manquant (1) / Présent (0)')

plt.tight_layout()
plt.savefig('visualizations/03_missing_values.png', dpi=150, bbox_inches='tight')
print(" Graphique 3: Valeurs manquantes")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CORRÉLATIONS AVEC LA TARGET
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle('Top 30 Corrélations avec market_forward_excess_returns', fontsize=16, fontweight='bold')

# Calculer les corrélations
all_features = [col for col in train.columns if col not in ['date_id', target, 'forward_returns', 'risk_free_rate']]
correlations = []

for feat in all_features:
    corr_data = train[[feat, target]].dropna()
    if len(corr_data) > 100:
        corr = corr_data.corr().iloc[0, 1]
        if not np.isnan(corr):
            correlations.append((feat, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
top_30 = correlations[:30]

features = [x[0] for x in top_30]
corr_values = [x[1] for x in top_30]
colors = ['red' if x < 0 else 'green' for x in corr_values]

y_pos = np.arange(len(features))
ax.barh(y_pos, corr_values, color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel('Corrélation')
ax.set_title('Top 30 Features par Corrélation Absolue')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/04_correlations.png', dpi=150, bbox_inches='tight')
print(" Graphique 4: Corrélations")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DISTRIBUTIONS PAR CATÉGORIE
# ═══════════════════════════════════════════════════════════════════════════════

feature_categories = {
    'D': 'Dummy/Binary',
    'E': 'Macro Economic',
    'I': 'Interest Rate',
    'M': 'Market Dynamics',
    'P': 'Price/Valuation',
    'S': 'Sentiment',
    'V': 'Volatility'
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Statistiques par Catégorie de Features', fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, (prefix, category) in enumerate(feature_categories.items()):
    features = [col for col in train.columns if col.startswith(prefix) and col not in ['date_id']]
    
    if len(features) > 0:
        # Calculer les statistiques
        missing_pcts = []
        for feat in features:
            missing_pct = train[feat].isnull().sum() / len(train) * 100
            missing_pcts.append(missing_pct)
        
        axes[idx].bar(range(len(missing_pcts)), missing_pcts, alpha=0.7)
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_ylabel('% Manquant')
        axes[idx].set_title(f'{category} ({prefix}*)\n{len(features)} features')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim([0, 100])

# Masquer le dernier subplot vide
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('visualizations/05_categories_missing.png', dpi=150, bbox_inches='tight')
print(" Graphique 5: Catégories de features")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. AUTOCORRÉLATION
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Analyse d\'Autocorrélation', fontsize=16, fontweight='bold')

# ACF
max_lag = 60
lags = range(1, max_lag + 1)
acf_values = [train[target].autocorr(lag=lag) for lag in lags]

axes[0].stem(lags, acf_values, basefmt=' ')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0].axhline(y=1.96/np.sqrt(len(train)), color='red', linestyle='--', linewidth=1, label='Seuil de significativité')
axes[0].axhline(y=-1.96/np.sqrt(len(train)), color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Autocorrélation')
axes[0].set_title('Fonction d\'Autocorrélation (ACF)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Autocorrélation de la volatilité (valeurs absolues)
abs_returns = train[target].abs()
acf_vol_values = [abs_returns.autocorr(lag=lag) for lag in lags]

axes[1].stem(lags, acf_vol_values, basefmt=' ')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].axhline(y=1.96/np.sqrt(len(train)), color='red', linestyle='--', linewidth=1, label='Seuil de significativité')
axes[1].axhline(y=-1.96/np.sqrt(len(train)), color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Autocorrélation')
axes[1].set_title('ACF de la Volatilité (|returns|)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_autocorrelation.png', dpi=150, bbox_inches='tight')
print(" Graphique 6: Autocorrélation")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. FORWARD RETURNS VS RISK FREE RATE
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analyse: Forward Returns vs Risk-Free Rate', fontsize=16, fontweight='bold')

# Série temporelle des deux composantes
axes[0, 0].plot(train['date_id'], train['forward_returns'], linewidth=0.5, alpha=0.7, label='Forward Returns')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_xlabel('date_id')
axes[0, 0].set_ylabel('Valeur')
axes[0, 0].set_title('Forward Returns')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(train['date_id'], train['risk_free_rate'], linewidth=0.5, alpha=0.7, color='orange', label='Risk-Free Rate')
axes[0, 1].set_xlabel('date_id')
axes[0, 1].set_ylabel('Valeur')
axes[0, 1].set_title('Risk-Free Rate')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Distribution
axes[1, 0].hist(train['forward_returns'].dropna(), bins=100, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Forward Returns')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title('Distribution des Forward Returns')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(train['risk_free_rate'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_xlabel('Risk-Free Rate')
axes[1, 1].set_ylabel('Fréquence')
axes[1, 1].set_title('Distribution du Risk-Free Rate')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/07_components.png', dpi=150, bbox_inches='tight')
print(" Graphique 7: Composantes de la target")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. RENDEMENTS POSITIFS VS NÉGATIFS DANS LE TEMPS
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('Rendements Positifs vs Négatifs', fontsize=16, fontweight='bold')

# Créer une série binaire
train['is_positive'] = (train[target] > 0).astype(int)

# Rolling proportion de rendements positifs
window = 60
rolling_positive = train['is_positive'].rolling(window).mean()

axes[0].plot(train['date_id'], rolling_positive, linewidth=2)
axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='50% (neutre)')
axes[0].fill_between(train['date_id'], 0.5, rolling_positive, where=(rolling_positive > 0.5), alpha=0.3, color='green', label='Périodes bullish')
axes[0].fill_between(train['date_id'], 0.5, rolling_positive, where=(rolling_positive <= 0.5), alpha=0.3, color='red', label='Périodes bearish')
axes[0].set_xlabel('date_id')
axes[0].set_ylabel('Proportion de rendements positifs')
axes[0].set_title(f'Rolling {window}-day Proportion de Rendements Positifs')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Séquences de rendements consécutifs
consecutive_positive = []
consecutive_negative = []
current_streak = 0
current_type = None

for val in train[target].dropna():
    if val > 0:
        if current_type == 'pos':
            current_streak += 1
        else:
            if current_type == 'neg' and current_streak > 0:
                consecutive_negative.append(current_streak)
            current_streak = 1
            current_type = 'pos'
    else:
        if current_type == 'neg':
            current_streak += 1
        else:
            if current_type == 'pos' and current_streak > 0:
                consecutive_positive.append(current_streak)
            current_streak = 1
            current_type = 'neg'

axes[1].hist([consecutive_positive, consecutive_negative], bins=range(1, 15), alpha=0.7, 
             label=['Séquences positives', 'Séquences négatives'], color=['green', 'red'])
axes[1].set_xlabel('Longueur de la séquence (jours consécutifs)')
axes[1].set_ylabel('Fréquence')
axes[1].set_title('Distribution des Séquences de Rendements Consécutifs')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/08_positive_negative.png', dpi=150, bbox_inches='tight')
print(" Graphique 8: Rendements positifs vs négatifs")
plt.close()

print("\n" + "="*80)
print("VISUALISATIONS COMPLÉTÉES")
print("="*80)
print(f"\nTous les graphiques ont été sauvegardés dans le dossier 'visualizations/':")
print("  1. 01_target_distribution.png    - Distribution de la target")
print("  2. 02_time_series.png             - Série temporelle")
print("  3. 03_missing_values.png          - Valeurs manquantes")
print("  4. 04_correlations.png            - Corrélations avec la target")
print("  5. 05_categories_missing.png      - Valeurs manquantes par catégorie")
print("  6. 06_autocorrelation.png         - Autocorrélation")
print("  7. 07_components.png              - Forward returns vs Risk-free rate")
print("  8. 08_positive_negative.png       - Rendements positifs vs négatifs")
print("\n Visualisations EDA terminées!")