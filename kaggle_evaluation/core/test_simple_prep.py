# Créez test_simple_prep.py
import pandas as pd
import pickle

from preprocessor_simple import HullPreprocessorSimple

print("Test du preprocessor simple...")

train = pd.read_csv('train.csv')
print(f"Train: {train.shape}")

# Fit transform
prep = HullPreprocessorSimple(verbose=True)
X = prep.fit_transform(train)

print(f"\nRésultats:")
print(f"  Shape: {X.shape}")
print(f"  Features avec variance > 0: {(X.std() > 0).sum()}")
print(f"  Features toutes à 0: {(X.std() == 0).sum()}")
print(f"  NaN: {X.isnull().sum().sum()}")

# Sauvegarder
with open('preprocessor_simple.pkl', 'wb') as f:
    pickle.dump(prep, f)

print("\n✓ preprocessor_simple.pkl sauvegardé")