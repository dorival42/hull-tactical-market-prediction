"""
═══════════════════════════════════════════════════════════════════════════════
COMPARAISON COMPLÈTE: GRADIENT BOOSTING vs DEEP LEARNING
═══════════════════════════════════════════════════════════════════════════════

Compare les performances de tous les modèles:
- LightGBM, XGBoost, Random Forest (Gradient Boosting)
- LSTM, GRU, SimpleNN (Deep Learning)

Auteur: Complete Comparison Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import sys,  os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from base_model import ModelMetrics
from feature_engineering.feature_engineer import FeatureEngineer
from gradient_boosting_models import LightGBMModel, XGBoostModel, RandomForestModel
from deep_learning_models import LSTMModel, GRUModel, SimpleNNModel

print("="*80)
print("COMPARAISON COMPLÈTE: GRADIENT BOOSTING vs DEEP LEARNING")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("1. PRÉPARATION DES DONNÉES")
print("="*80)

# Charger données
train = pd.read_csv('train.csv')
CUTOFF_DATE = 7000
train_clean = train[train['date_id'] >= CUTOFF_DATE].copy()

# Feature Engineering
fe = FeatureEngineer(verbose=False)
train_enhanced = fe.create_all_features(train_clean)
train_enhanced = train_enhanced.dropna(subset=['market_forward_excess_returns'])
train_enhanced = train_enhanced.iloc[100:].copy()

# Feature Selection
target_col = 'market_forward_excess_returns'
selected_features = fe.select_features(train_enhanced, target_col, 
                                      method='correlation', n_features=30)

print(f"\n✓ Données préparées: {train_enhanced.shape[0]:,} lignes")
print(f"✓ Features sélectionnées: {len(selected_features)}")

# Split
split_idx = int(len(train_enhanced) * 0.8)
train_df = train_enhanced.iloc[:split_idx]
val_df = train_enhanced.iloc[split_idx:]

X_train = train_df[selected_features].fillna(0)
y_train = train_df[target_col]
X_val = val_df[selected_features].fillna(0)
y_val = val_df[target_col]

print(f"✓ Train: {len(train_df):,} | Val: {len(val_df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("2. ENTRAÎNEMENT DES MODÈLES")
print("="*80)

all_models = {}
all_results = []
training_times = {}

# === GRADIENT BOOSTING MODELS ===
print("\n📊 GRADIENT BOOSTING MODELS")
print("-" * 80)

# 1. LightGBM
print("\n1. LightGBM")
start_time = time.time()
lgb_model = LightGBMModel()
lgb_model.fit(X_train, y_train, X_val, y_val)
lgb_time = time.time() - start_time
all_models['LightGBM'] = lgb_model
training_times['LightGBM'] = lgb_time

print(f"   RMSE: {lgb_model.training_metrics['rmse']:.6f}")
print(f"   R²:   {lgb_model.training_metrics['r2']:.4f}")
print(f"   Temps: {lgb_time:.2f}s")

# 2. XGBoost
print("\n2. XGBoost")
start_time = time.time()
xgb_model = XGBoostModel()
xgb_model.fit(X_train, y_train, X_val, y_val)
xgb_time = time.time() - start_time
all_models['XGBoost'] = xgb_model
training_times['XGBoost'] = xgb_time

print(f"   RMSE: {xgb_model.training_metrics['rmse']:.6f}")
print(f"   R²:   {xgb_model.training_metrics['r2']:.4f}")
print(f"   Temps: {xgb_time:.2f}s")

# 3. Random Forest
print("\n3. Random Forest")
start_time = time.time()
rf_model = RandomForestModel()
rf_model.fit(X_train, y_train, X_val, y_val)
rf_time = time.time() - start_time
all_models['RandomForest'] = rf_model
training_times['RandomForest'] = rf_time

print(f"   RMSE: {rf_model.training_metrics['rmse']:.6f}")
print(f"   R²:   {rf_model.training_metrics['r2']:.4f}")
print(f"   Temps: {rf_time:.2f}s")

# === DEEP LEARNING MODELS ===
print("\n" + "="*80)
print("🧠 DEEP LEARNING MODELS")
print("-" * 80)

# 4. LSTM
print("\n4. LSTM")
start_time = time.time()
lstm_params = {
    'units': 64,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'patience': 15,
    'sequence_length': 20
}
lstm_model = LSTMModel(params=lstm_params)
lstm_model.fit(X_train, y_train, X_val, y_val)
lstm_time = time.time() - start_time
all_models['LSTM'] = lstm_model
training_times['LSTM'] = lstm_time

print(f"   RMSE: {lstm_model.training_metrics['rmse']:.6f}")
print(f"   R²:   {lstm_model.training_metrics['r2']:.4f}")
print(f"   Temps: {lstm_time:.2f}s")

# 5. GRU
print("\n5. GRU")
start_time = time.time()
gru_params = {
    'units': 64,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'patience': 15,
    'sequence_length': 20
}
gru_model = GRUModel(params=gru_params)
gru_model.fit(X_train, y_train, X_val, y_val)
gru_time = time.time() - start_time
all_models['GRU'] = gru_model
training_times['GRU'] = gru_time

print(f"   RMSE: {gru_model.training_metrics['rmse']:.6f}")
print(f"   R²:   {gru_model.training_metrics['r2']:.4f}")
print(f"   Temps: {gru_time:.2f}s")

# 6. Simple NN
print("\n6. Simple Neural Network")
start_time = time.time()
nn_params = {
    'layers': [128, 64, 32],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'patience': 15
}
nn_model = SimpleNNModel(params=nn_params)
nn_model.fit(X_train, y_train, X_val, y_val)
nn_time = time.time() - start_time
all_models['SimpleNN'] = nn_model
training_times['SimpleNN'] = nn_time

print(f"   RMSE: {nn_model.training_metrics['rmse']:.6f}")
print(f"   R²:   {nn_model.training_metrics['r2']:.4f}")
print(f"   Temps: {nn_time:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CALCUL DU SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("3. CALCUL DU SHARPE RATIO")
print("="*80)

def convert_to_allocation(predictions, method='threshold'):
    """Convertir prédictions en allocations."""
    allocations = np.ones_like(predictions)
    allocations[predictions > 0.003] = 1.8
    allocations[(predictions > 0) & (predictions <= 0.003)] = 1.3
    allocations[(predictions < 0) & (predictions >= -0.003)] = 0.7
    allocations[predictions < -0.003] = 0.2
    return np.clip(allocations, 0.0, 2.0)

for name, model in all_models.items():
    print(f"\n{name}")
    print("-" * 40)
    
    # Prédictions
    y_pred = model.predict(X_val)
    
    # Métriques de régression
    metrics = model.get_metrics()
    
    # Allocations
    allocations = convert_to_allocation(y_pred)
    
    # Sharpe ratio
    sharpe_metrics = ModelMetrics.calculate_sharpe_ratio(
        allocations,
        val_df['forward_returns'].values,
        val_df['risk_free_rate'].values
    )
    
    all_results.append({
        'model': name,
        'type': 'Deep Learning' if name in ['LSTM', 'GRU', 'SimpleNN'] else 'Gradient Boosting',
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'sharpe_ratio': sharpe_metrics['sharpe_ratio'],
        'annualized_return': sharpe_metrics['annualized_return'] * 100,
        'annualized_volatility': sharpe_metrics['annualized_volatility'] * 100,
        'volatility_ratio': sharpe_metrics['volatility_ratio'],
        'exceeds_constraint': sharpe_metrics['exceeds_constraint'],
        'training_time': training_times[name]
    })
    
    print(f"   RMSE:              {metrics['rmse']:.6f}")
    print(f"   R²:                {metrics['r2']:.4f}")
    print(f"   Sharpe Ratio:      {sharpe_metrics['sharpe_ratio']:+.4f}")
    print(f"   Annual Return:     {sharpe_metrics['annualized_return']*100:+.2f}%")
    print(f"   Annual Volatility: {sharpe_metrics['annualized_volatility']*100:.2f}%")
    print(f"   Volatility Ratio:  {sharpe_metrics['volatility_ratio']:.2f}x")
    print(f"   Constraint OK:     {'NON ❌' if sharpe_metrics['exceeds_constraint'] else 'OUI ✓'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPARAISON FINALE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("4. COMPARAISON FINALE")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('sharpe_ratio', ascending=False)

print("\n" + "-"*80)
print("CLASSEMENT PAR SHARPE RATIO")
print("-"*80)
print(f"\n{'Rang':<5} {'Modèle':<20} {'Type':<20} {'Sharpe':<10} {'Return':<12} {'R²':<10} {'Temps':<10}")
print("-"*80)

for i, row in results_df.iterrows():
    rank = list(results_df.index).index(i) + 1
    model = row['model']
    model_type = row['type']
    sharpe = row['sharpe_ratio']
    ret = row['annualized_return']
    r2 = row['r2']
    time_sec = row['training_time']
    
    print(f"{rank:<5} {model:<20} {model_type:<20} {sharpe:<+10.4f} {ret:<+12.2f}% {r2:<10.4f} {time_sec:<10.2f}s")

# Meilleur de chaque catégorie
print("\n" + "-"*80)
print("MEILLEUR PAR CATÉGORIE")
print("-"*80)

gb_best = results_df[results_df['type'] == 'Gradient Boosting'].iloc[0]
dl_best = results_df[results_df['type'] == 'Deep Learning'].iloc[0]

print(f"\n🥇 GRADIENT BOOSTING: {gb_best['model']}")
print(f"   Sharpe: {gb_best['sharpe_ratio']:+.4f}")
print(f"   R²:     {gb_best['r2']:.4f}")
print(f"   Temps:  {gb_best['training_time']:.2f}s")

print(f"\n🧠 DEEP LEARNING: {dl_best['model']}")
print(f"   Sharpe: {dl_best['sharpe_ratio']:+.4f}")
print(f"   R²:     {dl_best['r2']:.4f}")
print(f"   Temps:  {dl_best['training_time']:.2f}s")

# Statistiques
print("\n" + "-"*80)
print("STATISTIQUES PAR TYPE")
print("-"*80)

for model_type in ['Gradient Boosting', 'Deep Learning']:
    type_df = results_df[results_df['type'] == model_type]
    
    print(f"\n{model_type}:")
    print(f"   Sharpe moyen:    {type_df['sharpe_ratio'].mean():+.4f}")
    print(f"   Sharpe std:      {type_df['sharpe_ratio'].std():.4f}")
    print(f"   R² moyen:        {type_df['r2'].mean():.4f}")
    print(f"   Temps moyen:     {type_df['training_time'].mean():.2f}s")

# Sauvegarder
results_df.to_csv('../files_results/complete_comparison.csv', index=False)
print("\n✓ Résultats sauvegardés: complete_comparison.csv")

print("\n" + "="*80)
print("🏆 GAGNANT GLOBAL: " + results_df.iloc[0]['model'].upper())
print(f"   Sharpe Ratio: {results_df.iloc[0]['sharpe_ratio']:+.4f}")
print(f"   Type: {results_df.iloc[0]['type']}")
print("="*80)
