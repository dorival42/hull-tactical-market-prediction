"""
Script de test local COMPLET pour Hull Tactical
Simule le comportement du gateway Kaggle avec 3 modes de test
VERSION FINALE OPTIMISÉE
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import polars as pl
import time
import traceback


def test_full_pipeline():
    """
    Test complet du pipeline avec le gateway local
    Simule exactement le comportement de Kaggle
    """
    
    print("=" * 80)
    print("TEST COMPLET AVEC GATEWAY LOCAL")
    print("=" * 80 + "\n")
    
    # 1. Vérifier que les fichiers nécessaires existent
    print("1️⃣  VÉRIFICATION DES FICHIERS")
    print("-" * 80)
    
    required_files = [
        'test.csv',
        'xgb_model.pkl',
        'preprocessor.pkl',
        'default_inference_server.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"   ❌ Manquant: {file}")
        else:
            size_mb = Path(file).stat().st_size / (1024 * 1024)
            print(f"   ✓ Trouvé: {file} ({size_mb:.2f} MB)")
    
    if missing_files:
        print(f"\n❌ Fichiers manquants: {', '.join(missing_files)}")
        print("\n💡 Actions requises:")
        if 'xgb_model.pkl' in missing_files or 'preprocessor.pkl' in missing_files:
            print("  1. Exécutez d'abord: python baseline_model.py")
        if 'test.csv' in missing_files:
            print("  2. Téléchargez les données: kaggle competitions download -c hull-tactical-market-prediction")
        if 'default_inference_server.py' in missing_files:
            print("  3. Assurez-vous que default_inference_server.py existe")
        return False
    
    print("\n✅ Tous les fichiers nécessaires sont présents\n")
    
    # 2. Importer l'InferenceServer
    print("2️⃣  INITIALISATION DE L'INFERENCE SERVER")
    print("-" * 80)
    
    try:
        from default_inference_server import DefaultInferenceServer
    except ImportError as e:
        print(f"❌ ERREUR lors de l'import de DefaultInferenceServer: {e}")
        print("\n💡 Vérifiez que default_inference_server.py est dans le même dossier")
        return False
    
    try:
        inference_server = DefaultInferenceServer()
    except Exception as e:
        print(f"❌ ERREUR lors de l'initialisation: {e}")
        traceback.print_exc()
        return False
    
    print("\n✅ InferenceServer initialisé avec succès\n")
    
    # 3. Exécuter le gateway local
    print("3️⃣  EXÉCUTION DU GATEWAY LOCAL")
    print("-" * 80)
    print("   (Cela va traiter tous les batches du test.csv)\n")
    
    start_time = time.time()
    
    try:
        # Le gateway va appeler votre fonction predict() pour chaque batch
        data_paths = (str(Path.cwd()),)  # Dossier actuel
        inference_server.run_local_gateway(data_paths=data_paths)
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'exécution du gateway: {e}")
        traceback.print_exc()
        return False
    
    elapsed_time = time.time() - start_time
    
    # 4. Afficher le résumé
    inference_server.print_summary()
    
    print("=" * 80)
    print("✅ TEST COMPLET RÉUSSI !")
    print("=" * 80)
    print(f"⏱️  Temps total: {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
    
    if elapsed_time > 3600:
        print(f"\n⚠️  WARNING: Temps > 1 heure. Risque de timeout sur Kaggle.")
    elif elapsed_time > 1800:
        print(f"\n⚠️  Temps élevé ({elapsed_time/60:.1f} min). Optimisation recommandée.")
    else:
        print(f"\n✅ Temps OK pour Kaggle")
    
    return True


def test_single_batch():
    """
    Test simple sur un seul batch
    Plus rapide pour le debugging
    """
    
    print("=" * 80)
    print("TEST RAPIDE - SINGLE BATCH")
    print("=" * 80 + "\n")
    
    # 1. Vérifier les fichiers essentiels
    print("1️⃣  VÉRIFICATION DES FICHIERS")
    print("-" * 80)
    
    essential_files = ['test.csv', 'xgb_model.pkl', 'preprocessor.pkl']
    
    for file in essential_files:
        if not Path(file).exists():
            print(f"   ❌ Manquant: {file}")
            print(f"\n❌ Exécutez d'abord: python baseline_model.py")
            return False
        else:
            print(f"   ✓ {file}")
    
    print("\n✅ Fichiers essentiels présents\n")
    
    # 2. Créer l'InferenceServer
    print("2️⃣  INITIALISATION")
    print("-" * 80)
    
    try:
        from default_inference_server import DefaultInferenceServer
        inference_server = DefaultInferenceServer()
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        traceback.print_exc()
        return False
    
    print("\n✅ Initialisation réussie\n")
    
    # 3. Charger un échantillon de test
    print("3️⃣  CHARGEMENT DE test.csv")
    print("-" * 80)
    
    test = pd.read_csv('test.csv')
    print(f"   Shape: {test.shape}")
    print(f"   Colonnes: {test.columns.tolist()[:10]}...")  # Afficher les 10 premières
    
    # 4. Identifier les batch_ids (ou date_ids)
    if 'batch_id' in test.columns:
        batch_col = 'batch_id'
    elif 'date_id' in test.columns:
        batch_col = 'date_id'
    else:
        batch_col = test.columns[0]
    
    print(f"\n   Colonne de batch: {batch_col}")
    
    batch_ids = test[batch_col].unique()
    print(f"   Nombre de batches: {len(batch_ids)}")
    print(f"   Batch IDs: {batch_ids.tolist()}")
    
    if len(batch_ids) == 0:
        print("\n❌ Aucun batch trouvé dans test.csv")
        return False
    
    # 5. Tester sur le premier batch
    print(f"\n4️⃣  TEST SUR LE PREMIER BATCH ({batch_col}={batch_ids[0]})")
    print("-" * 80)
    
    first_batch_id = batch_ids[0]
    test_batch_df = test[test[batch_col] == first_batch_id]
    
    print(f"   Batch {first_batch_id}: {len(test_batch_df)} ligne(s)")
    print(f"   Colonnes: {len(test_batch_df.columns)}")
    
    # Afficher les colonnes importantes
    important_cols = [batch_col, 'is_scored']
    for col in important_cols:
        if col in test_batch_df.columns:
            values = test_batch_df[col].unique()
            print(f"   {col}: {values}")
    
    # Convertir en Polars (comme le fait le gateway)
    test_batch_polars = pl.from_pandas(test_batch_df)
    test_batch = (test_batch_polars,)
    
    # 6. Faire la prédiction
    print(f"\n5️⃣  PRÉDICTION")
    print("-" * 80 + "\n")
    
    try:
        start_time = time.time()
        predictions = inference_server._predict_batch(test_batch)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Prédictions générées avec succès !")
        print(f"⏱️  Temps: {elapsed_time:.3f}s")
        
        # Afficher les résultats
        print(f"\n6️⃣  RÉSULTATS POUR BATCH {first_batch_id}")
        print("-" * 80)
        
        results = pd.DataFrame({
            batch_col: test_batch_df[batch_col].values,
            'prediction': predictions.values
        })
        
        print(results.to_string(index=False))
        
        print(f"\n📊 Statistiques:")
        print(f"   Min    : {predictions.min():.6f}")
        print(f"   Max    : {predictions.max():.6f}")
        print(f"   Mean   : {predictions.mean():.6f}")
        print(f"   Median : {predictions.median():.6f}")
        print(f"   Std    : {predictions.std():.6f}")
        
        # Vérifications
        print(f"\n🔍 Validations:")
        nan_count = predictions.isna().sum()
        print(f"   NaN: {nan_count} {'✅' if nan_count == 0 else '⚠️'}")
        
        extreme_count = (predictions.abs() > 0.1).sum()
        print(f"   Valeurs extrêmes (>|0.1|): {extreme_count} {'✅' if extreme_count == 0 else '⚠️'}")
        
    except Exception as e:
        print(f"❌ ERREUR lors de la prédiction: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ TEST SINGLE BATCH RÉUSSI !")
    print("=" * 80)
    
    return True


def analyze_test_structure():
    """
    Analyser la structure de test.csv
    Utile pour comprendre les données avant de tester
    """
    
    print("=" * 80)
    print("ANALYSE DE TEST.CSV")
    print("=" * 80 + "\n")
    
    if not Path('test.csv').exists():
        print("❌ test.csv n'existe pas")
        print("\n💡 Téléchargez les données:")
        print("   kaggle competitions download -c hull-tactical-market-prediction")
        return False
    
    # Charger les données
    print("📥 Chargement de test.csv...")
    test = pd.read_csv('test.csv')
    
    # Informations de base
    print(f"\n📊 INFORMATIONS GÉNÉRALES")
    print("-" * 80)
    print(f"Shape: {test.shape}")
    print(f"Taille: {Path('test.csv').stat().st_size / (1024*1024):.2f} MB")
    
    # Colonnes
    print(f"\n📋 COLONNES ({len(test.columns)})")
    print("-" * 80)
    
    # Grouper les colonnes par type
    col_groups = {}
    for col in test.columns:
        if col in ['date_id', 'is_scored']:
            prefix = 'Meta'
        elif col.startswith('lagged_'):
            prefix = 'Lagged'
        elif len(col) > 1 and col[0].isalpha() and col[1:].isdigit():
            prefix = col[0]
        else:
            prefix = 'Other'
        
        if prefix not in col_groups:
            col_groups[prefix] = []
        col_groups[prefix].append(col)
    
    for prefix in sorted(col_groups.keys()):
        cols = col_groups[prefix]
        print(f"   {prefix:8s}: {len(cols):3d} colonnes - {cols[:5]}...")
    
    # Identifier la colonne de batch
    print(f"\n🔑 COLONNE DE BATCH")
    print("-" * 80)
    
    if 'batch_id' in test.columns:
        batch_col = 'batch_id'
    elif 'date_id' in test.columns:
        batch_col = 'date_id'
    else:
        batch_col = test.columns[0]
    
    print(f"   Colonne identifiée: {batch_col}")
    
    # Analyser les batches
    batch_ids = test[batch_col].unique()
    print(f"   Nombre de batches uniques: {len(batch_ids)}")
    print(f"   Batch IDs: {sorted(batch_ids.tolist())}")
    
    # Taille de chaque batch
    print(f"\n📦 TAILLE DES BATCHES")
    print("-" * 80)
    
    batch_sizes = test.groupby(batch_col).size()
    print(f"   Min: {batch_sizes.min()} lignes")
    print(f"   Max: {batch_sizes.max()} lignes")
    print(f"   Moyenne: {batch_sizes.mean():.1f} lignes")
    print(f"   Total: {batch_sizes.sum()} lignes")
    
    if len(batch_sizes) <= 20:
        print(f"\n   Détail par batch:")
        for batch_id, size in batch_sizes.items():
            print(f"      Batch {batch_id}: {size} ligne(s)")
    
    # Colonnes is_scored
    print(f"\n🎯 COLONNES SPÉCIALES")
    print("-" * 80)
    
    if 'is_scored' in test.columns:
        scored_count = test['is_scored'].sum()
        print(f"   is_scored: {scored_count}/{len(test)} lignes seront scorées")
        
        if scored_count < len(test):
            print(f"   ⚠️  {len(test) - scored_count} lignes ne seront PAS scorées (public leaderboard)")
    else:
        print(f"   is_scored: Colonne absente")
    
    # Lagged features
    lagged_cols = [col for col in test.columns if col.startswith('lagged_')]
    if lagged_cols:
        print(f"\n   Lagged features ({len(lagged_cols)}):")
        for col in lagged_cols:
            print(f"      - {col}")
    
    # Types de données
    print(f"\n📊 TYPES DE DONNÉES")
    print("-" * 80)
    
    type_counts = test.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"   {str(dtype):12s}: {count:3d} colonnes")
    
    # Valeurs manquantes
    print(f"\n❓ VALEURS MANQUANTES")
    print("-" * 80)
    
    missing = test.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print(f"   {len(missing_cols)} colonnes avec des NaN:")
        
        # Afficher les 10 colonnes avec le plus de NaN
        top_missing = missing_cols.sort_values(ascending=False).head(10)
        for col, count in top_missing.items():
            pct = count / len(test) * 100
            print(f"      {col:30s}: {count:5d} ({pct:5.1f}%)")
        
        if len(missing_cols) > 10:
            print(f"      ... et {len(missing_cols) - 10} autres colonnes")
    else:
        print("   ✅ Aucune valeur manquante")
    
    # Premières lignes
    print(f"\n📋 PREMIÈRES LIGNES")
    print("-" * 80)
    print(test.head(3).to_string())
    
    # Statistiques descriptives (quelques colonnes)
    print(f"\n📈 STATISTIQUES (échantillon)")
    print("-" * 80)
    
    numeric_cols = test.select_dtypes(include=['number']).columns[:5]
    print(test[numeric_cols].describe().to_string())
    
    print("\n" + "=" * 80)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 80)
    
    return True


def print_help():
    """Afficher l'aide détaillée"""
    
    print("\n" + "=" * 80)
    print("AIDE - TEST_LOCAL.PY")
    print("=" * 80)
    
    print("\n📖 DESCRIPTION")
    print("-" * 80)
    print("   Script de test local pour Hull Tactical Market Prediction.")
    print("   Permet de tester l'InferenceServer avant la soumission Kaggle.")
    
    print("\n🎯 MODES DISPONIBLES")
    print("-" * 80)
    
    modes = [
        ("analyze", "Analyser la structure de test.csv", "Rapide", "Découvrir les données"),
        ("single", "Tester sur un seul batch", "Rapide", "Debugging rapide"),
        ("full", "Tester sur tous les batches", "Lent", "Simulation complète Kaggle"),
    ]
    
    for mode, desc, speed, usage in modes:
        print(f"\n   {mode:10s} - {desc}")
        print(f"                Vitesse: {speed}")
        print(f"                Usage: {usage}")
    
    print("\n📝 EXEMPLES D'UTILISATION")
    print("-" * 80)
    print("   # Analyser test.csv")
    print("   python test_local.py --mode analyze")
    print()
    print("   # Test rapide (1 batch)")
    print("   python test_local.py --mode single")
    print()
    print("   # Test complet (tous les batches)")
    print("   python test_local.py --mode full")
    
    print("\n📋 WORKFLOW RECOMMANDÉ")
    print("-" * 80)
    print("   1. python test_local.py --mode analyze")
    print("      → Comprendre la structure des données")
    print()
    print("   2. python baseline_model.py")
    print("      → Entraîner le modèle")
    print()
    print("   3. python test_local.py --mode single")
    print("      → Test rapide du pipeline")
    print()
    print("   4. python test_local.py --mode full")
    print("      → Validation complète avant soumission")
    print()
    print("   5. [Soumettre à Kaggle]")
    
    print("\n💡 PRÉREQUIS")
    print("-" * 80)
    print("   - test.csv (données Kaggle)")
    print("   - xgb_model.pkl (généré par baseline_model.py)")
    print("   - preprocessor.pkl (généré par baseline_model.py)")
    print("   - default_inference_server.py")
    
    print("\n" + "=" * 80 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    
    # Parser les arguments
    parser = argparse.ArgumentParser(
        description='Tester l\'InferenceServer localement pour Hull Tactical',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python test_local.py --mode analyze    # Analyser test.csv
  python test_local.py --mode single     # Test rapide (1 batch)
  python test_local.py --mode full       # Test complet (tous les batches)
  python test_local.py --help            # Afficher l'aide détaillée
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'full', 'analyze', 'help'],
        help='Mode de test: single (rapide), full (complet), analyze (analyser test.csv), help (aide détaillée)'
    )
    
    args = parser.parse_args()
    
    # Exécuter le mode choisi
    print("\n" + "🔍" * 40)
    print(f"MODE: {args.mode.upper()}")
    print("🔍" * 40 + "\n")
    
    success = False
    
    if args.mode == 'help':
        print_help()
        success = True
    
    elif args.mode == 'analyze':
        success = analyze_test_structure()
    
    elif args.mode == 'single':
        success = test_single_batch()
    
    elif args.mode == 'full':
        success = test_full_pipeline()
    
    else:
        print(f"❌ Mode inconnu: {args.mode}")
        print("   Utilisez --help pour voir les modes disponibles.")
        sys.exit(1)
    
    # Exit code
    if success:
        print("\n💡 PROCHAINES ÉTAPES:")
        if args.mode == 'analyze':
            print("   1. Entraîner le modèle: python baseline_model.py")
            print("   2. Tester: python test_local.py --mode single")
        elif args.mode == 'single':
            print("   1. Test complet: python test_local.py --mode full")
            print("   2. Analyser: python analyze_results.py")
        elif args.mode == 'full':
            print("   1. Analyser: python analyze_results.py")
            print("   2. Si Sharpe > 0.5: SOUMETTRE À KAGGLE ! 🚀")
        
        sys.exit(0)
    else:
        print("\n❌ Des problèmes ont été détectés.")
        print("   Corrigez-les avant de continuer.")
        sys.exit(1)