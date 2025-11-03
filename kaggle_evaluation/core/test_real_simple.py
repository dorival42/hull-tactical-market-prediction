# Créez: test_real_sample.py
import pandas as pd
import pickle
import numpy as np

print("Test sur un échantillon RÉEL (milieu du dataset)")

train = pd.read_csv('train.csv')

# Prendre un échantillon au MILIEU (pas au début)
start_idx = 4000
sample = train.iloc[start_idx:start_idx+100].copy()

print(f"\n1. Sample: lignes {start_idx} à {start_idx+100}")
print(f"   Target mean: {sample['market_forward_excess_returns'].mean():.6f}")
print(f"   Target std: {sample['market_forward_excess_returns'].std():.6f}")

# Charger preprocessor
with open('preprocessor.pkl', 'rb') as f:
    prep = pickle.load(f)

# Transform
X = prep.transform(sample)

print(f"\n2. Après preprocessing:")
print(f"   Shape: {X.shape}")
print(f"   Features variance > 0: {(X.std() > 0).sum()}")
print(f"   Features = 0: {(X.std() == 0).sum()}")

# Vérifier quelques colonnes
print(f"\n3. Quelques features:")
print(f"   E1: min={X['E1'].min():.3f}, max={X['E1'].max():.3f}, std={X['E1'].std():.3f}")
print(f"   M1: min={X['M1'].min():.3f}, max={X['M1'].max():.3f}, std={X['M1'].std():.3f}")
print(f"   V1: min={X['V1'].min():.3f}, max={X['V1'].max():.3f}, std={X['V1'].std():.3f}")

if 'lagged_market_forward_excess_returns' in X.columns:
    print(f"   lagged_target: min={X['lagged_market_forward_excess_returns'].min():.6f}, max={X['lagged_market_forward_excess_returns'].max():.6f}")

# Prédire
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

preds = model.predict(X)

print(f"\n4. Prédictions:")
print(f"   Min: {preds.min():.6f}")
print(f"   Max: {preds.max():.6f}")
print(f"   Mean: {preds.mean():.6f}")
print(f"   Std: {preds.std():.10f}")
print(f"   Unique values: {len(np.unique(preds))}")

if preds.std() > 0:
    print("\n✅ LE MODÈLE FONCTIONNE ! Le problème était l'échantillon de test.")
else:
    print("\n❌ Le problème persiste même sur un vrai échantillon.")