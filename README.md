
# 🎯 HULL TACTICAL - MARKET PREDICTION
## Analyse Complète du Challenge Kaggle

**Date de l'analyse :** 7 Novembre 2025  
**Participant :** 
- Pierre Chrislin DORIVAL
- Emile STEEVENSON
- Jobed FELIMA
- Sebastien Witchmen ESTANIS
  

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble](#vue-densemble)
2. [Structure des fichiers](#structure-des-fichiers)
3. [Description des données](#description-des-données)
4. [Architecture de l'API](#architecture-de-lapi)
5. [Méthodologie de soumission](#méthodologie-de-soumission)
6. [Stratégie recommandée](#stratégie-recommandée)

---

## 🎯 VUE D'ENSEMBLE

### Objectif du Challenge

Prédire les **rendements excédentaires du S&P 500** (`market_forward_excess_returns`) tout en respectant une **contrainte de volatilité de 120%**.

### Défi intellectuel

Remettre en question l'**Hypothèse des Marchés Efficaces (EMH)** qui stipule qu'il est impossible de battre le marché de manière systématique.

### Récompenses

- **Prize Pool** : $100,000
- **Timeline** :
  - Phase d'entraînement : 16 sept - 15 déc 2025
  - Phase de forecasting : 15 déc 2025 - 16 juin 2026

### Particularité unique

Contrairement à la plupart des compétitions Kaggle, Nos modèles seront **exécutés en temps réel** sur le marché pendant 6 mois après la deadline de soumission.

---

## 📁 STRUCTURE DES FICHIERS

```
hull-tactical-market-prediction/
│
├── train.csv                    # 8,991 lignes (décennies de données historiques)
├── test.csv                     # 11 lignes (mock test set)
│
└── kaggle_evaluation/           # API d'évaluation
    ├── __init__.py
    ├── default_gateway.py       # Gateway par défaut
    ├── default_inference_server.py  # Serveur d'inférence
    │
    └── core/
        ├── __init__.py
        ├── base_gateway.py      # Implémentation de base
        ├── templates.py         # Templates pour Gateway et InferenceServer
        ├── relay.py             # Communication gRPC
        ├── kaggle_evaluation.proto
        │
        └── generated/
            ├── __init__.py
            ├── kaggle_evaluation_pb2.py
            └── kaggle_evaluation_pb2_grpc.py
```

---

## 📊 DESCRIPTION DES DONNÉES

### TRAIN.CSV (8,991 lignes × 98 colonnes)

#### 🔑 Identifiant
- `date_id` : Identifiant unique pour chaque jour de trading (0 à 8990)

#### 📈 FEATURES (95 colonnes de variables prédictives)

| Catégorie | Préfixe | Nombre | Description |
|-----------|---------|--------|-------------|
| **Dummy/Binary** | `D*` | 9 | Variables binaires/catégorielles (D1-D9) |
| **Macro Economic** | `E*` | 20 | Indicateurs macro-économiques (E1-E20) |
| **Interest Rate** | `I*` | 9 | Taux d'intérêt (I1-I9) |
| **Market Dynamics** | `M*` | 18 | Dynamiques de marché (M1-M18) |
| **Price/Valuation** | `P*` | 13 | Prix et valorisation (P1-P13) |
| **Sentiment** | `S*` | 12 | Indicateurs de sentiment (S1-S12) |
| **Volatility** | `V*` | 13 | Indicateurs de volatilité (V1-V13) |
| **Momentum** | `MOM*` | 1 | Indicateur de momentum |

**⚠️ IMPORTANT** : Les premières années contiennent de **nombreuses valeurs manquantes** (coverage incomplet dans les données anciennes).

#### 🎯 TARGETS (3 colonnes - TRAIN UNIQUEMENT)

1. **`forward_returns`**
   - Rendements obtenus en achetant le S&P 500 et en le vendant le lendemain
   - Formule : `(Prix_t+1 - Prix_t) / Prix_t`

2. **`risk_free_rate`**
   - Taux des fonds fédéraux (Federal Funds Rate)
   - Utilisé pour calculer les rendements excédentaires

3. **`market_forward_excess_returns`** ⭐ **CIBLE PRINCIPALE**
   - Rendements excédentaires par rapport aux attentes
   - **Calcul** :
     ```
     1. excess_returns = forward_returns - risk_free_rate
     2. mean_5y = moyenne mobile sur 5 ans de excess_returns
     3. deviation = excess_returns - mean_5y
     4. MAD = Median Absolute Deviation de deviation
     5. market_forward_excess_returns = winsorize(deviation, MAD × 4)
     ```
   - **C'est cette valeur que nous devons prédire**

---

### TEST.CSV (11 lignes × 99 colonnes)

#### Structure pendant la phase d'entraînement
- **Mock test set** : Copie des **180 derniers `date_id`** du train set (8811-8990)
- **10 lignes seulement** dans le fichier mock fourni

#### Colonnes supplémentaires (par rapport au train)

| Colonne | Description |
|---------|-------------|
| **`is_scored`** | Indique si la ligne est incluse dans l'évaluation |
| **`lagged_forward_returns`** | `forward_returns` avec 1 jour de retard |
| **`lagged_risk_free_rate`** | `risk_free_rate` avec 1 jour de retard |
| **`lagged_market_forward_excess_returns`** | `market_forward_excess_returns` avec 1 jour de retard |

**⚠️ Pourquoi le lag ?**  
Simule la réalité : nous ne connaissons les rendements qu'**après la clôture** du marché. Cela évite le "look-ahead bias".

---

## 🔄 PHASES DE LA COMPÉTITION

### Phase 1 : Model Training (16 sept - 15 déc 2025)

```
TRAIN SET
├── Date IDs: 0 → 8990
├── Features: D*, E*, I*, M*, P*, S*, V*, MOM*
└── Targets: forward_returns, risk_free_rate, market_forward_excess_returns

TEST SET (Mock)
├── Date IDs: 8811 → 8990 (copie des derniers 180 jours du train)
├── Features: Identiques au train
├── Lagged targets: Disponibles avec 1 jour de retard
└── is_scored: True pour tous les jours
```

**Public Leaderboard** : ⚠️ **NON SIGNIFICATIF** (données déjà vues dans le train set)

---

### Phase 2 : Forecasting (15 déc 2025 - 16 juin 2026)

```
TEST SET (Real-time)
├── Nouvelles données du marché servies progressivement
├── Vos notebooks s'exécutent AUTOMATIQUEMENT chaque jour
├── is_scored: True uniquement pour les nouveaux jours de trading
└── Durée: ~6 mois = ~180 jours de trading
```

**Private Leaderboard** : Calculé sur les vraies prédictions du marché en temps réel.

---

## 🏗️ ARCHITECTURE DE L'API

### Composants principaux

#### 1. **Gateway** (`default_gateway.py`)
- **Rôle** : Coordonne l'évaluation
- **Responsabilités** :
  - Charger les données test
  - Envoyer les batches au serveur d'inférence
  - Valider les prédictions
  - Générer le fichier de soumission

```python
class DefaultGateway(kaggle_evaluation.core.templates.Gateway):
    def generate_data_batches(self):
        # Lit test.csv
        # Génère des batches par date_id
        # Yield (test_batch, batch_id)
    
    def competition_specific_validation(self, prediction, row_ids, data_batch):
        # Validation spécifique au challenge
        pass
```

#### 2. **InferenceServer** (`default_inference_server.py`)
- **Rôle** : Notre code de prédiction
- **Responsabilités** :
  - Recevoir les batches de données
  - Générer les prédictions
  - Retourner les allocations (0.0 à 2.0)

```python
class DefaultInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def predict(self, test_batch):
        # Notre  CODE ICI
        # Retourner une allocation entre 0.0 et 2.0
        return allocation
```

#### 3. **Communication gRPC**
- Utilise Protocol Buffers pour la communication
- Permet l'échange de DataFrames entre Gateway et InferenceServer

---

## 📤 MÉTHODOLOGIE DE SOUMISSION

### Ce que nous devons soumettre

**UN NOTEBOOK** qui :
1. Définit une fonction `predict(test_batch)`
2. Crée un `InferenceServer` avec cette fonction
3. Démarre le serveur avec `server.serve()`

### Format de la prédiction

Pour chaque `date_id`, retourner une **allocation** :

| Valeur | Signification |
|--------|---------------|
| **0.0** | 0% exposé au marché (cash) |
| **0.5** | 50% exposé |
| **1.0** | 100% exposé (position standard) |
| **1.5** | 150% exposé (levier) |
| **2.0** | 200% exposé (levier maximal autorisé) |

### Exemple minimal

```python
from kaggle_evaluation import default_inference_server

def predict(test_batch):
    """
    Args:
        test_batch: DataFrame avec les features pour un date_id
    
    Returns:
        float ou Series: Allocation entre 0.0 et 2.0
    """
    # Notre modèle de prédiction
    prediction = model.predict(test_batch)
    
    # Convertir en allocation (0.0 à 2.0)
    allocation = convert_to_allocation(prediction)
    
    return allocation

# Créer le serveur avec notre fonction predict
inference_server = default_inference_server.DefaultInferenceServer(predict)

# Tester localement
inference_server.run_local_gateway()

# Pour la soumission Kaggle
inference_server.serve()
```

---

## 📏 MÉTRIQUE D'ÉVALUATION

### Sharpe Ratio Modifié avec Contraintes

La métrique est une **variante du Sharpe Ratio** qui pénalise :

1. **Volatilité excessive** : > 120% de la volatilité du marché
2. **Sous-performance** : Ne pas surperformer le rendement du marché

### Formule (conceptuelle)

```
Score = (Rendement_stratégie - Rendement_marché) / Volatilité_stratégie

Avec pénalités si :
- Volatilité_stratégie > 1.2 × Volatilité_marché
- Rendement_stratégie < Rendement_marché
```

**Le code exact de la métrique est disponible sur Kaggle.**

---

## 🎯 STRATÉGIE RECOMMANDÉE

### 1. Exploration des Données (EDA)

#### A. Analyse temporelle
```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
train = pd.read_csv('train.csv')

# Analyser la couverture des features dans le temps
missing_by_date = train.isnull().sum(axis=1)
plt.plot(train['date_id'], missing_by_date)
plt.title('Valeurs manquantes par date')
plt.show()

# Analyser la distribution de la target
train['market_forward_excess_returns'].hist(bins=100)
plt.title('Distribution des rendements excédentaires')
plt.show()
```

#### B. Analyse des features par catégorie
```python
# Grouper par catégorie
D_features = [col for col in train.columns if col.startswith('D')]
E_features = [col for col in train.columns if col.startswith('E')]
# ... etc

# Analyser les corrélations
correlation_with_target = train[E_features + ['market_forward_excess_returns']].corr()['market_forward_excess_returns']
print(correlation_with_target.sort_values(ascending=False))
```

---

### 2. Feature Engineering

#### A. Gestion des valeurs manquantes
```python
# Stratégies possibles :
# 1. Limiter l'analyse aux années récentes (moins de missing)
# 2. Forward fill pour certaines features (prix, taux)
# 3. Modèles robustes aux missing (LightGBM, CatBoost)
```

#### B. Features dérivées
```python
# Lag features
for lag in [1, 5, 10, 20]:
    train[f'forward_returns_lag_{lag}'] = train['forward_returns'].shift(lag)

# Rolling statistics
for window in [5, 10, 20, 60]:
    train[f'volatility_{window}d'] = train['forward_returns'].rolling(window).std()
    train[f'mean_return_{window}d'] = train['forward_returns'].rolling(window).mean()

# Momentum indicators
train['momentum_5_20'] = (
    train['forward_returns'].rolling(5).mean() - 
    train['forward_returns'].rolling(20).mean()
)
```

---

### 3. Modélisation

#### A. Baseline simple
```python
# Stratégie 1 : Allocation constante
def baseline_constant(test_batch):
    return 1.0  # Toujours 100% investi

# Stratégie 2 : Basée sur la volatilité récente
def baseline_volatility(test_batch):
    recent_vol = test_batch['V1'].iloc[0]  # Exemple
    if recent_vol > threshold_high:
        return 0.5  # Réduire l'exposition
    else:
        return 1.5  # Augmenter l'exposition
```

#### B. Modèles ML

**Option 1 : Régression directe**
```python
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Prédire directement market_forward_excess_returns
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5
)

# Features sélectionnées
features = D_features + E_features + I_features + ['lagged_forward_returns']

# Entraîner
model.fit(train[features], train['market_forward_excess_returns'])

# Convertir prédiction en allocation
def predict(test_batch):
    pred_return = model.predict(test_batch[features])
    
    # Stratégie d'allocation basée sur la prédiction
    if pred_return > 0.005:
        allocation = 1.5  # Bullish
    elif pred_return < -0.005:
        allocation = 0.3  # Bearish
    else:
        allocation = 1.0  # Neutre
    
    return allocation
```

**Option 2 : Classification (Bear/Bull/Neutral)**
```python
from sklearn.ensemble import GradientBoostingClassifier

# Créer des classes
train['signal'] = pd.cut(
    train['market_forward_excess_returns'],
    bins=[-np.inf, -0.003, 0.003, np.inf],
    labels=['bear', 'neutral', 'bull']
)

# Modèle de classification
model = GradientBoostingClassifier()
model.fit(train[features], train['signal'])

# Allocation basée sur la classe prédite
def predict(test_batch):
    signal = model.predict(test_batch[features])[0]
    
    allocation_map = {
        'bear': 0.2,
        'neutral': 1.0,
        'bull': 1.8
    }
    
    return allocation_map[signal]
```

---

### 4. Gestion du Risque (Contrainte de volatilité)

```python
def predict_with_risk_management(test_batch, model, max_vol_ratio=1.2):
    # Prédiction brute
    raw_allocation = model.predict(test_batch)
    
    # Estimer la volatilité anticipée
    estimated_vol = estimate_volatility(test_batch)
    market_vol = test_batch['V1'].iloc[0]  # Exemple
    
    # Ajuster si nécessaire
    if estimated_vol > max_vol_ratio * market_vol:
        # Réduire l'allocation pour respecter la contrainte
        scaling_factor = (max_vol_ratio * market_vol) / estimated_vol
        adjusted_allocation = raw_allocation * scaling_factor
    else:
        adjusted_allocation = raw_allocation
    
    # Assurer que l'allocation reste dans [0, 2]
    return np.clip(adjusted_allocation, 0.0, 2.0)
```

---

### 5. Validation

#### A. Walk-forward validation
```python
# Ne jamais entraîner sur des données futures
# Simuler le processus de prédiction jour par jour

results = []
for i in range(train_size, len(train)):
    # Train sur données passées uniquement
    train_window = train.iloc[max(0, i-lookback):i]
    test_day = train.iloc[i:i+1]
    
    # Entraîner le modèle
    model.fit(train_window[features], train_window['target'])
    
    # Prédire
    prediction = model.predict(test_day[features])
    results.append(prediction)
```

#### B. Calcul du Sharpe ratio
```python
def calculate_sharpe(allocations, returns, risk_free_rates):
    # Portfolio returns
    portfolio_returns = allocations * returns
    
    # Excess returns
    excess_returns = portfolio_returns - risk_free_rates
    
    # Sharpe ratio
    sharpe = excess_returns.mean() / excess_returns.std()
    
    return sharpe
```

---

### 6. Test Local avec l'API

```python
from kaggle_evaluation import default_inference_server

# Votre fonction de prédiction
def predict(test_batch):
    # Votre code ici
    return allocation

# Créer le serveur
inference_server = default_inference_server.DefaultInferenceServer(predict)

# Tester localement sur le mock test set
inference_server.run_local_gateway()

# Vérifier submission.parquet
import pandas as pd
submission = pd.read_parquet('submission.parquet')
print(submission.head())
```

---

## 🎲 APPROCHES AVANCÉES

### 1. Ensemble de modèles
```python
# Combiner plusieurs modèles
predictions = []
predictions.append(model_lgb.predict(test_batch) * 0.4)
predictions.append(model_xgb.predict(test_batch) * 0.3)
predictions.append(model_rf.predict(test_batch) * 0.3)

final_prediction = sum(predictions)
```

### 2. Modèles de séries temporelles
```python
from statsmodels.tsa.arima.model import ARIMA
# ARIMA, GARCH pour la volatilité
```

### 3. Deep Learning
```python
import torch
import torch.nn as nn

class MarketPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

---

## ⚠️ PIÈGES À ÉVITER

### 1. Look-ahead bias
❌ **NE JAMAIS** utiliser des informations futures pour entraîner le modèle

### 2. Overfitting sur le public leaderboard
❌ Le public leaderboard n'est **pas significatif** (mock data)
✅ Se concentrer sur une validation walk-forward robuste

### 3. Ignorer la contrainte de volatilité
❌ Un modèle très performant mais trop volatil sera pénalisé
✅ Toujours vérifier que `Volatilité_stratégie ≤ 1.2 × Volatilité_marché`

### 4. Ne pas gérer les valeurs manquantes
❌ Les premières années ont beaucoup de missing values
✅ Soit les ignorer, soit les imputer intelligemment

### 5. Stratégies trop complexes
❌ Modèles avec des centaines de features peuvent mal généraliser
✅ Commencer simple, ajouter de la complexité progressivement

---

## 📝 CHECKLIST AVANT SOUMISSION

- [ ] Le modèle s'exécute sans erreur sur le test set local
- [ ] La fonction `predict()` retourne des valeurs entre 0.0 et 2.0
- [ ] Le notebook démarre le serveur avec `inference_server.serve()`
- [ ] Le temps de startup est < 5 minutes (limite Kaggle)
- [ ] La prédiction par batch prend < 5 minutes (timeout)
- [ ] Le modèle a été validé avec walk-forward validation
- [ ] La contrainte de volatilité est respectée
- [ ] Les dépendances sont installées correctement
- [ ] Le code ne contient pas de look-ahead bias

---

## 🚀  ÉTAPES DU DEVELOPPEMLENT DU PROJET

### Étape 1 : EDA Approfondie
1. Charger et explorer `train.csv`
2. Analyser les missing values par période
3. Visualiser la distribution de `market_forward_excess_returns`
4. Étudier les corrélations entre features et target

### Étape 2 : Baseline
1. Créer une stratégie baseline simple
2. Tester avec l'API locale
3. Calculer le Sharpe ratio sur validation set

### Étape 3 : Feature Engineering
1. Créer des lag features
2. Calculer des rolling statistics
3. Ajouter des momentum indicators

### Étape 4 : Modélisation
1. Entraîner plusieurs modèles (LightGBM, XGBoost, RF)
2. Walk-forward validation
3. Optimiser les hyperparamètres

### Étape 5 : Gestion du Risque
1. Implémenter la contrainte de volatilité
2. Tester différentes stratégies d'allocation
3. Valider le Sharpe ratio

### Étape 6 : Soumission
1. Créer le notebook de soumission
2. Tester localement avec l'API
3. Soumettre sur Kaggle
4. Surveiller les performances en temps réel

---

## 📚 RESSOURCES

### Documentation Kaggle
- Page de la compétition : https://www.kaggle.com/competitions/hull-tactical-market-prediction
- Notebook d'exemple : Disponible dans la section "Code"
- Forum de discussion : Pour poser des questions

### Concepts clés
- Hypothèse des Marchés Efficaces (EMH)
- Sharpe Ratio
- Contrainte de volatilité
- Walk-forward validation
- Time series forecasting

### Librairies utiles
- `pandas`, `polars` : Manipulation de données
- `numpy` : Calculs numériques
- `scikit-learn` : ML classique
- `lightgbm`, `xgboost`, `catboost` : Boosting
- `statsmodels` : Séries temporelles
- `pytorch`, `tensorflow` : Deep learning

---

## 🏆 OBJECTIFS

### Court terme (1-2 semaines)
- [ ] Comprendre parfaitement les données
- [ ] Créer une baseline fonctionnelle
- [ ] Soumettre une première version

### Moyen terme (1 mois)
- [ ] Développer des features avancées
- [ ] Optimiser le modèle
- [ ] Atteindre un Sharpe ratio > 1.0 en validation

### Long terme (jusqu'au 15 déc)
- [ ] Ensemble de modèles
- [ ] Stratégie de risk management robuste
- [ ] Viser le top 10% du leaderboard

---

**cette compétition est passionnante, elle pourrait remettre en question l'une des théories fondamentales de la finance moderne ! 🚀📈**
** 🚀 Cest une atout pour notre future Carrière dans la Finance, Data science sur Machine learning **
