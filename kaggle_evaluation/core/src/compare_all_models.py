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
train_clean = train.copy()

# Feature Engineering
fe = FeatureEngineer(verbose=False)
train_enhanced = fe._create_features(train_clean)
train_enhanced = train_enhanced.dropna(subset=['market_forward_excess_returns', 'forward_returns', 'risk_free_rate'])
train_enhanced = train_enhanced.copy()

# Feature Selection
target_col = 'market_forward_excess_returns'
selected_features = fe.select_features(train_enhanced, target_col, 
                                      method='correlation', n_features=100)



important_cols = [col for col in train_enhanced.columns 
                  if not col.startswith('feat_') and col != "forward_returns" and col != "risk_free_rate"]


train_enhanced = train_enhanced[ important_cols + selected_features ].copy()
print(f"\n✓ Données préparées: {train_enhanced.shape[0]:,} lignes")
print(f"✓ Features sélectionnées: {len(selected_features)}")

train_enhanced = train_enhanced.set_index("date_id")
# Split
split_idx = int(len(train_enhanced) * 0.8)
train_df = train_enhanced.iloc[:split_idx]
val_df = train_enhanced.iloc[split_idx:]

X_train = train_df.fillna(0)
y_train = train_df[target_col]
X_val = val_df.fillna(0)
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
# 3. PREDICTIONS ET METRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("3. PREDICTIONS ET METRIQUES")
print("="*80)

for name, model in all_models.items():
    print(f"\n{name}")
    print("-" * 40)

    # Predictions
    y_pred = model.predict(X_val)

    # Metriques de regression
    metrics = model.get_metrics()

    all_results.append({
        'model': name,
        'type': 'Deep Learning' if name in ['LSTM', 'GRU', 'SimpleNN'] else 'Gradient Boosting',
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'training_time': training_times[name]
    })

    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    print(f"   R2:   {metrics['r2']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPARAISON FINALE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("4. COMPARAISON FINALE")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('rmse', ascending=True)

print("\n" + "-"*80)
print("CLASSEMENT PAR RMSE")
print("-"*80)
print(f"\n{'Rang':<5} {'Modele':<20} {'Type':<20} {'RMSE':<12} {'R2':<10} {'Temps':<10}")
print("-"*80)

for i, row in results_df.iterrows():
    rank = list(results_df.index).index(i) + 1
    model = row['model']
    model_type = row['type']
    rmse = row['rmse']
    r2 = row['r2']
    time_sec = row['training_time']

    print(f"{rank:<5} {model:<20} {model_type:<20} {rmse:<12.6f} {r2:<10.4f} {time_sec:<10.2f}s")

# Meilleur de chaque categorie
print("\n" + "-"*80)
print("MEILLEUR PAR CATEGORIE")
print("-"*80)

gb_best = results_df[results_df['type'] == 'Gradient Boosting'].iloc[0]
dl_best = results_df[results_df['type'] == 'Deep Learning'].iloc[0]

print(f"\nGRADIENT BOOSTING: {gb_best['model']}")
print(f"   RMSE:  {gb_best['rmse']:.6f}")
print(f"   R2:    {gb_best['r2']:.4f}")
print(f"   Temps: {gb_best['training_time']:.2f}s")

print(f"\nDEEP LEARNING: {dl_best['model']}")
print(f"   RMSE:  {dl_best['rmse']:.6f}")
print(f"   R2:    {dl_best['r2']:.4f}")
print(f"   Temps: {dl_best['training_time']:.2f}s")

# Statistiques
print("\n" + "-"*80)
print("STATISTIQUES PAR TYPE")
print("-"*80)

for model_type in ['Gradient Boosting', 'Deep Learning']:
    type_df = results_df[results_df['type'] == model_type]

    print(f"\n{model_type}:")
    print(f"   RMSE moyen:      {type_df['rmse'].mean():.6f}")
    print(f"   R2 moyen:        {type_df['r2'].mean():.4f}")
    print(f"   Temps moyen:     {type_df['training_time'].mean():.2f}s")

# Sauvegarder
results_df.to_csv('./kaggle_evaluation/core/files_results/complete_comparison.csv', index=False)
print("\nResultats sauvegardes: complete_comparison.csv")

print("\n" + "="*80)
print("GAGNANT GLOBAL: " + results_df.iloc[0]['model'].upper())
print(f"   RMSE: {results_df.iloc[0]['rmse']:.6f}")
print(f"   Type: {results_df.iloc[0]['type']}")
print("="*80)
