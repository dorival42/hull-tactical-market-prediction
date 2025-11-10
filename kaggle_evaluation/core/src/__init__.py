import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""
═══════════════════════════════════════════════════════════════════════════════
MODELS PACKAGE - HULL TACTICAL
═══════════════════════════════════════════════════════════════════════════════

Package contenant tous les modèles de prédiction.

Auteur: Advanced Modeling Hull Tactical
Date: 7 Novembre 2025
═══════════════════════════════════════════════════════════════════════════════
"""

from base_model import BaseModel, ModelConfig, ModelMetrics
from feature_engineering.feature_engineer import FeatureEngineer

__all__ = [
    'BaseModel',
    'ModelConfig',
    'ModelMetrics',
    'FeatureEngineer'
]

__version__ = '1.0.0'


