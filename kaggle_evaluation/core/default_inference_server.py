"""
FINAL InferenceServer for Hull Tactical Market Prediction Competition
Features: 

"""

from kaggle_evaluation import default_gateway
from kaggle_evaluation.core import base_gateway
from kaggle_evaluation.core.generated import kaggle_evaluation_pb2
import kaggle_evaluation.core.templates
import default_gateway
import pandas as pd 
import polars as pl
import pickle
import numpy as np
from pathlib import Path
import time
import sys


class DefaultInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    """
Inference Server for Hull Tactical Competition

Features:
- Automatic model and preprocessor loading
- Robust error handling
- Detailed logs for debugging
- Prediction validation

- Optimized for performance
    """
    
    def __init__(self):
        """
        Initialization: Load the model and preprocessor

        IMPORTANT: Must be FAST (< 15 minutes) to avoid timeout on Kaggle
        """
        init_start = time.time()
        
        print("=" * 80)
        print(" HULL TACTICAL - INFERENCE SERVER")
        print("=" * 80)
        print(f" Initialization: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # File paths
        self.model_path = Path('xgb_model.pkl')
        self.preprocessor_path = Path('preprocessor.pkl')
        
        # Verify that the files exist
        self._validate_files()
        
        #Load the modele
        print("\n Load the model")
        model_start = time.time()
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            model_time = time.time() - model_start
            print(f"    Model loaded in {model_time:.2f}s")
            print(f"    Type: {type(self.model).__name__}")
            
            # Verify that the model has a predict method
            if not hasattr(self.model, 'predict'):
                raise AttributeError("The model does not have a `predict` method.")
                
        except Exception as e:
            print(f"   ERROR while loading the model: {e}")
            raise
        
        # Loading  preprocessor
        print("\n Loading preprocessor...")
        prep_start = time.time()
        
        try:
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            prep_time = time.time() - prep_start
            print(f"    Preprocessor loaded at {prep_time:.2f}s")
            print(f"    Type: {type(self.preprocessor).__name__}")
            
            # Verify that the preprocessor has a transform method
            if not hasattr(self.preprocessor, 'transform'):
                raise AttributeError("The model does not have a 'transform' method")
            
            # Display the number of expected features
            if hasattr(self.preprocessor, 'feature_names'):
                print(f"    Features attendues: {len(self.preprocessor.feature_names)}")
            else:
                print("     Warning: preprocessor.feature_names non disponible")
                
        except Exception as e:
            print(f"    ERROR while loading the preprocessor: {e}")
            raise
        
        # Statistics
        self.batch_count = 0
        self.total_predictions = 0
        self.total_time = 0
        self.errors = []
        
        # Define the prediction function
        def predict(test_batch):
            """
            Prediction function called by the gateway

        Args:
            test_batch: tuple containing (DataFrame Polars,)

        Returns:
            predictions: pandas Series with the predictions
            """
            return self._predict_batch(test_batch)
        
        # Initialize the server with the predict function
        super().__init__(predict)
        
        init_time = time.time() - init_start
        print("\n" + "=" * 80)
        print(f"  InferenceServer successfully initialized at {init_time:.2f}s")
        print("=" * 80 + "\n")
        
        # Warning si le temps d'initialisation est trop long
        if init_time > 60:
            print(f"  WARNING: Long initialization time ({init_time:.1f}s)")
            print("  This could cause problems during submission.")
        
    def _validate_files(self):
        """Verify that all necessary files exist"""
        print("\n File verification...")
        
        files_to_check = [
            (self.model_path, "Model"),
            (self.preprocessor_path, "Preprocessor")
        ]
        
        missing_files = []
        for file_path, name in files_to_check:
            if not file_path.exists():
                missing_files.append(f"{name}: {file_path}")
                print(f"    {name} manquant: {file_path}")
            else:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"    {name}: {file_path} ({size_mb:.2f} MB)")
        
        if missing_files:
            error_msg = "Missing files:\n" + "\n".join(missing_files)
            raise FileNotFoundError(error_msg)
    
    def _predict_batch(self, test_batch):
        """
        Faire les prédictions sur un batch
        
        Args:
            test_batch: tuple contenant un DataFrame Polars
            
        Returns:
            predictions: pandas Series
        """
        batch_start = time.time()
        self.batch_count += 1
        
        try:
            # 1. Extraire et convertir le DataFrame
            extract_start = time.time()
            
            if not isinstance(test_batch, tuple):
                raise TypeError(f"test_batch doit être un tuple, reçu: {type(test_batch)}")
            
            if len(test_batch) == 0:
                raise ValueError("test_batch est vide")
            
            test_df_polars = test_batch[0]
            
            if not isinstance(test_df_polars, (pl.DataFrame, pd.DataFrame)):
                raise TypeError(f"test_batch[0] doit être un DataFrame, reçu: {type(test_df_polars)}")
            
            # Convertir en Pandas si nécessaire
            if isinstance(test_df_polars, pl.DataFrame):
                test_df = test_df_polars.to_pandas()
            else:
                test_df = test_df_polars.copy()
            
            extract_time = time.time() - extract_start
            
            # 2. Logs du batch
            print(f"\n{'='*80}")
            print(f" BATCH #{self.batch_count}")
            print(f"{'='*80}")
            print(f"   Shape: {test_df.shape}")
            
            if 'date_id' in test_df.columns:
                date_ids = test_df['date_id'].tolist()
                print(f"   Date IDs: {date_ids}")
            
            print(f"   Colonnes: {len(test_df.columns)}")
            print(f"     Extraction: {extract_time:.3f}s")
            
            # 3. Vérifications de sécurité
            self._validate_input(test_df)
            
            # 4. Preprocessing
            prep_start = time.time()
            
            try:
                X_test = self.preprocessor.transform(test_df)
                prep_time = time.time() - prep_start
                
                print(f"     Preprocessing: {prep_time:.3f}s")
                print(f"   Features shape: {X_test.shape}")
                
                # Vérifier la shape
                if hasattr(self.preprocessor, 'feature_names'):
                    expected_features = len(self.preprocessor.feature_names)
                    if X_test.shape[1] != expected_features:
                        print(f"     Warning: Shape mismatch!")
                        print(f"      Expected: {expected_features} features")
                        print(f"      Got: {X_test.shape[1]} features")
                
            except Exception as e:
                print(f"    ERREUR lors du preprocessing: {e}")
                raise
            
            # 5. Prédiction
            pred_start = time.time()
            
            try:
                predictions = self.model.predict(X_test)
                pred_time = time.time() - pred_start
                
                print(f"     Prédiction: {pred_time:.3f}s")
                print(f"   Prédictions: {len(predictions)}")
                
            except Exception as e:
                print(f"    ERREUR lors de la prédiction: {e}")
                raise
            
            # 6. Validation des prédictions
            predictions = self._validate_predictions(predictions, test_df)
            
            # 7. Convertir en pandas Series
            if not isinstance(predictions, pd.Series):
                predictions_series = pd.Series(predictions, name='prediction')
            else:
                predictions_series = predictions
            
            # 8. Statistiques
            batch_time = time.time() - batch_start
            self.total_predictions += len(predictions)
            self.total_time += batch_time
            
            print(f"\n    Statistiques des prédictions:")
            print(f"      Min     : {predictions.min():.6f}")
            print(f"      Max     : {predictions.max():.6f}")
            print(f"      Mean    : {predictions.mean():.6f}")
            print(f"      Std     : {predictions.std():.6f}")
            print(f"      Median  : {np.median(predictions):.6f}")
            
            # Vérifier les valeurs extrêmes
            extreme_count = np.sum(np.abs(predictions) > 0.05)
            if extreme_count > 0:
                print(f"        {extreme_count} prédictions > |0.05|")
            
            print(f"\n     Temps total batch: {batch_time:.3f}s")
            print(f"    Vitesse: {len(predictions)/batch_time:.1f} predictions/s")
            
            # Warning si le batch est trop lent
            if batch_time > 30:
                print(f"\n     WARNING: Batch très lent ({batch_time:.1f}s)")
                print("      Risque de timeout si beaucoup de batches")
            
            print(f"{'='*80}\n")
            
            return predictions_series
            
        except Exception as e:
            # Enregistrer l'erreur
            error_info = {
                'batch': self.batch_count,
                'error': str(e),
                'type': type(e).__name__
            }
            self.errors.append(error_info)
            
            print(f"\n{'='*80}")
            print(f" ERREUR DANS LE BATCH #{self.batch_count}")
            print(f"{'='*80}")
            print(f"Type: {type(e).__name__}")
            print(f"Message: {e}")
            print(f"{'='*80}\n")
            
            # Re-raise l'erreur pour que Kaggle la détecte
            raise
    
    def _validate_input(self, test_df):
        """Valider les données d'entrée"""
        # Vérifier que le DataFrame n'est pas vide
        if len(test_df) == 0:
            raise ValueError("DataFrame de test vide")
        
        # Vérifier qu'il y a des colonnes
        if len(test_df.columns) == 0:
            raise ValueError("DataFrame de test sans colonnes")
        
        # Vérifier les colonnes attendues
        expected_cols = ['date_id', 'D1', 'M1', 'V1', 'P1', 'S1', 'I1', 'E1']
        missing_cols = [col for col in expected_cols if col not in test_df.columns]
        
        if missing_cols:
            print(f"     Colonnes manquantes: {missing_cols}")
            # Ne pas raise, le preprocessor gérera
    
    def _validate_predictions(self, predictions, test_df):
        """Valider et nettoyer les prédictions"""
        
        # Vérifier la taille
        if len(predictions) != len(test_df):
            raise ValueError(
                f"Nombre de prédictions ({len(predictions)}) != "
                f"nombre de lignes ({len(test_df)})"
            )
        
        # Vérifier les NaN
        nan_count = np.isnan(predictions).sum()
        if nan_count > 0:
            print(f"     {nan_count} NaN détectés dans les prédictions")
            print(f"      Remplacement par 0...")
            predictions = np.nan_to_num(predictions, nan=0.0)
        
        # Vérifier les Inf
        inf_count = np.isinf(predictions).sum()
        if inf_count > 0:
            print(f"     {inf_count} Inf détectés dans les prédictions")
            print(f"      Remplacement par 0...")
            predictions = np.nan_to_num(predictions, posinf=0.0, neginf=0.0)
        
        # Vérifier les valeurs extrêmes (clipper si nécessaire)
        extreme_threshold = 0.1  # 10% de rendement en 1 jour est extrême
        extreme_mask = np.abs(predictions) > extreme_threshold
        if extreme_mask.sum() > 0:
            print(f"     {extreme_mask.sum()} valeurs extrêmes (>|{extreme_threshold}|)")
            print(f"      Clipping à ±{extreme_threshold}...")
            predictions = np.clip(predictions, -extreme_threshold, extreme_threshold)
        
        return predictions
    
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        """
        Retourner le gateway pour les tests locaux
        """
        return default_gateway.DefaultGateway(data_paths)
    
    def print_summary(self):
        """Afficher un résumé des performances"""
        print("\n" + "=" * 80)
        print(" RÉSUMÉ DES PERFORMANCES")
        print("=" * 80)
        print(f"   Batches traités: {self.batch_count}")
        print(f"   Prédictions totales: {self.total_predictions}")
        
        if self.batch_count > 0:
            avg_time = self.total_time / self.batch_count
            avg_speed = self.total_predictions / self.total_time
            print(f"   Temps moyen/batch: {avg_time:.3f}s")
            print(f"   Vitesse moyenne: {avg_speed:.1f} predictions/s")
        
        if self.errors:
            print(f"\n     Erreurs: {len(self.errors)}")
            for err in self.errors[:5]:  # Afficher max 5 erreurs
                print(f"      - Batch {err['batch']}: {err['type']} - {err['error']}")
        else:
            print(f"\n    Aucune erreur")
        
        print("=" * 80 + "\n")


# ==============================================================================
# TEST LOCAL (optionnel)
# ==============================================================================

def test_inference_server_standalone():
    """
    Test standalone de l'InferenceServer
    Usage: python default_inference_server.py
    """
    
    print("\n" + "=" * 80)
    print("TEST STANDALONE DE L'INFERENCE SERVER")
    print("=" * 80 + "\n")
    
    # 1. Créer l'inference server
    try:
        inference_server = DefaultInferenceServer()
    except Exception as e:
        print(f"\n Échec de l'initialisation: {e}")
        return False
    
    # 2. Charger un échantillon de test
    if not Path('test.csv').exists():
        print("\n test.csv non trouvé. Test standalone annulé.")
        return False
    
    print(" Chargement d'un échantillon de test...")
    test = pd.read_csv('test.csv')
    print(f"   Test shape: {test.shape}\n")
    
    # 3. Identifier les batch_ids
    if 'batch_id' in test.columns:
        batch_col = 'batch_id'
    else:
        batch_col = 'date_id'
    
    batch_ids = test[batch_col].unique()
    print(f"Colonne de batch: {batch_col}")
    print(f"Nombre de batches: {len(batch_ids)}")
    print(f"Batch IDs: {batch_ids.tolist()}\n")
    
    # 4. Tester sur quelques batches
    num_batches_to_test = min(10, len(batch_ids))
    print(f" Test sur {num_batches_to_test} batches...\n")
    
    all_predictions = []
    
    for i, batch_id in enumerate(batch_ids[:num_batches_to_test]):
        try:
            # Préparer le batch
            test_batch_df = test[test[batch_col] == batch_id]
            
            # Convertir en Polars (comme le fait le gateway)
            test_batch_polars = pl.from_pandas(test_batch_df)
            test_batch = (test_batch_polars,)
            
            # Faire la prédiction
            predictions = inference_server._predict_batch(test_batch)
            
            # Stocker les résultats
            results = pd.DataFrame({
                batch_col: test_batch_df[batch_col].values,
                'prediction': predictions.values
            })
            all_predictions.append(results)
            
        except Exception as e:
            print(f"\n Erreur lors du test du batch {batch_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 5. Afficher les résultats
    if all_predictions:
        all_results = pd.concat(all_predictions, ignore_index=True)
        
        print("\n" + "=" * 80)
        print("RÉSULTATS DES TESTS")
        print("=" * 80)
        print(all_results.to_string(index=False))
        
        print(f"\nStatistiques globales:")
        print(f"  Min: {all_results['prediction'].min():.6f}")
        print(f"  Max: {all_results['prediction'].max():.6f}")
        print(f"  Mean: {all_results['prediction'].mean():.6f}")
        print(f"  Std: {all_results['prediction'].std():.6f}")
    
    # 6. Afficher le résumé
    inference_server.print_summary()
    
    print("=" * 80)
    print(" TEST STANDALONE RÉUSSI !")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    """
    Test local de l'InferenceServer
    Usage: python default_inference_server.py
    """
    
    success = test_inference_server_standalone()
    
    if success:
        print("\n Prochaines étapes:")
        print("  1. Exécuter: python test_local.py --mode full")
        print("  2. Vérifier que tous les batches passent")
        print("  3. Soumettre à Kaggle !")
        sys.exit(0)
    else:
        print("\n Des problèmes ont été détectés.")
        print("  Corrigez-les avant de soumettre à Kaggle.")
        sys.exit(1)