
from data_processing import HullPreprocessorV2

# ==============================================================================
# FONCTION UTILITAIRE
# ==============================================================================

def test_preprocessor():
    """Tester le preprocessor sur les données"""
    import pandas as pd
    
    print("="*80)
    print("TEST DU PREPROCESSOR V2")
    print("="*80)
    
    # Charger les données
    train = pd.read_csv('train.csv')
    print(f"\nTrain shape: {train.shape}")
    
    # Créer et fitter le preprocessor
    preprocessor = HullPreprocessorV2(verbose=True)
    X = preprocessor.fit_transform(train)

    summarize = preprocessor.summarize(X)

    print("\n" + "="*80)
    print(summarize)

    print(f"\n✓ Transformation réussie: {X.shape}")
    
    # Résumé
    summary = preprocessor.get_feature_summary()
    print("\n" + "="*80)
    print("RÉSUMÉ DES FEATURES")
    print("="*80)
    print(f"Total features: {summary['total_features']}")
    print(f"  - Original: {summary['original_features']}")
    print(f"  - Lagged: {summary['lagged_features']}")
    print(f"  - Created: {summary['created_features']}")
    print(f"\nGroups:")
    for group, count in summary['feature_groups'].items():
        print(f"  {group}*: {count}")
    
    # Vérifier les NaN
    nan_count = X.isnull().sum().sum()
    print(f"\nNaN finaux: {nan_count}")
    
    if nan_count == 0:
        print("✅ Aucun NaN - Preprocessor prêt !")
    else:
        print(f"❌ {nan_count} NaN restants - À corriger")
    
    return preprocessor, X


if __name__ == '__main__':
    """Test du preprocessor"""
    try:
        preprocessor, X = test_preprocessor()
        print("\n" + "="*80)
        print("✅ TEST RÉUSSI !")
        print("="*80)
        
        # Sauvegarder le preprocessor de test
        import pickle
        with open('preprocessor_test.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        print("\n✓ Preprocessor sauvegardé: preprocessor_test.pkl")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST ÉCHOUÉ")
        print("="*80)
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
