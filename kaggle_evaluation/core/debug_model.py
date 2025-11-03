# Créez ce fichier: debug_model.py
import pandas as pd
import pickle
import numpy as np

print("=" * 80)
print("DIAGNOSTIC DU PROBLÈME")
print("=" * 80)

# 1. Charger les données
train = pd.read_csv('train.csv')
print(f"\n1. Train shape: {train.shape}")
print(f"   Target stats:")
print(f"   - Mean: {train['market_forward_excess_returns'].mean():.10f}")
print(f"   - Std: {train['market_forward_excess_returns'].std():.10f}")
print(f"   - Min: {train['market_forward_excess_returns'].min():.6f}")
print(f"   - Max: {train['market_forward_excess_returns'].max():.6f}")

# 2. Charger le preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

print(f"\n2. Preprocessor:")
print(f"   - Feature names: {len(preprocessor.feature_names) if hasattr(preprocessor, 'feature_names') else 'N/A'}")

# 3. Transformer un échantillon
sample = train.head(10)
X_sample = preprocessor.transform(sample)

print(f"\n3. Transformed sample:")
print(f"   - Shape: {X_sample.shape}")
print(f"   - Colonnes avec variance > 0: {(X_sample.std() > 0).sum()}")
print(f"   - Colonnes toutes à 0: {(X_sample.std() == 0).sum()}")
print(f"   - NaN: {X_sample.isnull().sum().sum()}")

print(f"\n4. Sample des features (premières lignes):")
print(X_sample.iloc[:5, :10])

# 4. Charger le modèle et prédire
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_sample)
actuals = sample['market_forward_excess_returns'].values

print(f"\n5. Prédictions sur échantillon:")
print(f"   Predictions: {predictions}")
print(f"   Actuals:     {actuals}")
print(f"   Prediction variance: {predictions.std():.10f}")

# 5. Vérifier feature importance
print(f"\n6. Feature importance (top 10):")
fi = pd.DataFrame({
    'feature': preprocessor.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(fi.head(10))
print(f"\n   Features avec importance > 0: {(fi['importance'] > 0).sum()}")