"""
Script d'analyse complète des performances pour Hull Tactical
Génère des visualisations et des métriques détaillées
VERSION FINALE OPTIMISÉE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


# ==============================================================================
# MÉTRIQUES FINANCIÈRES
# ==============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """Calculer le Sharpe Ratio annualisé"""
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0 or np.isnan(std_return):
        return 0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    sharpe_annualized = sharpe * np.sqrt(252)
    
    return sharpe_annualized


def calculate_sortino_ratio(returns, risk_free_rate=0):
    """Calculer le Sortino Ratio"""
    mean_return = np.mean(returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return 0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0
    
    sortino = (mean_return - risk_free_rate) / downside_std
    sortino_annualized = sortino * np.sqrt(252)
    
    return sortino_annualized


def calculate_max_drawdown(returns):
    """Calculer le Maximum Drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(returns):
    """Calculer le Calmar Ratio"""
    annual_return = np.mean(returns) * 252
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return 0
    
    return annual_return / max_dd


def calculate_hit_rate(predictions, actuals):
    """Calculer le Hit Rate"""
    return np.mean(np.sign(predictions) == np.sign(actuals))


def calculate_information_ratio(predictions, actuals):
    """Calculer l'Information Ratio"""
    excess_returns = predictions - actuals
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0
    
    return np.mean(excess_returns) / tracking_error


# ==============================================================================
# ANALYSE PRINCIPALE
# ==============================================================================

def analyze_validation_performance():
    """Analyser les performances sur le validation set"""
    
    print("=" * 80)
    print("🔍 ANALYSE DES PERFORMANCES - VALIDATION SET")
    print("=" * 80 + "\n")
    
    # 1. Charger les données
    print("1️⃣  Chargement des données...")
    print("-" * 80)
    
    if not Path('train.csv').exists():
        print("❌ train.csv non trouvé")
        return None
    
    train = pd.read_csv('train.csv')
    print(f"✓ Train shape: {train.shape}")
    
    # Split train/validation (même split que baseline_model.py)
    split_idx = int(len(train) * 0.8)
    val_data = train.iloc[split_idx:].copy()
    
    print(f"✓ Validation set: {len(val_data)} observations")
    print(f"✓ Période: date_id {val_data['date_id'].min()} à {val_data['date_id'].max()}")
    
    # 2. Charger le modèle et faire des prédictions
    print("\n2️⃣  Chargement du modèle et génération des prédictions...")
    print("-" * 80)
    
    if not Path('xgb_model.pkl').exists() or not Path('preprocessor.pkl').exists():
        print("❌ Modèle ou preprocessor non trouvé")
        print("   Exécutez d'abord: python baseline_model.py")
        return None
    
    import pickle
    
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    print("✓ Modèle chargé")
    print("✓ Preprocessor chargé")
    
    # Preprocessing
    print("\n⏳ Preprocessing des données de validation...")
    X_val = preprocessor.transform(val_data)
    print(f"✓ Features shape: {X_val.shape}")
    
    # Prédictions
    print("⏳ Génération des prédictions...")
    predictions = model.predict(X_val)
    actuals = val_data['market_forward_excess_returns'].values
    
    print(f"✓ {len(predictions)} prédictions générées")
    
    # 3. Calculer les métriques
    print("\n3️⃣  Calcul des métriques...")
    print("-" * 80)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'MSE': mean_squared_error(actuals, predictions),
        'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
        'MAE': mean_absolute_error(actuals, predictions),
        'R²': r2_score(actuals, predictions),
        'Sharpe': calculate_sharpe_ratio(predictions),
        'Sortino': calculate_sortino_ratio(predictions),
        'Hit_Rate': calculate_hit_rate(predictions, actuals),
        'Max_Drawdown': calculate_max_drawdown(pd.Series(predictions)),
        'Calmar': calculate_calmar_ratio(predictions),
        'Information_Ratio': calculate_information_ratio(predictions, actuals),
        'Correlation': np.corrcoef(predictions, actuals)[0, 1],
        'Volatility': np.std(predictions)
    }
    
    print("\n" + "=" * 80)
    print("📊 MÉTRIQUES DE PERFORMANCE")
    print("=" * 80 + "\n")
    
    print("🎯 Erreurs de Prédiction:")
    print(f"   MSE             : {metrics['MSE']:.8f}")
    print(f"   RMSE            : {metrics['RMSE']:.6f}")
    print(f"   MAE             : {metrics['MAE']:.6f}")
    print(f"   R²              : {metrics['R²']:.6f}")
    
    print("\n📈 Ratios de Performance:")
    print(f"   Sharpe Ratio    : {metrics['Sharpe']:.4f}")
    print(f"   Sortino Ratio   : {metrics['Sortino']:.4f}")
    print(f"   Calmar Ratio    : {metrics['Calmar']:.4f}")
    print(f"   Info Ratio      : {metrics['Information_Ratio']:.4f}")
    
    print("\n🎲 Précision Directionnelle:")
    print(f"   Hit Rate        : {metrics['Hit_Rate']:.2%}")
    print(f"   Correlation     : {metrics['Correlation']:.4f}")
    
    print("\n📉 Risque:")
    print(f"   Max Drawdown    : {metrics['Max_Drawdown']:.2%}")
    print(f"   Volatility      : {metrics['Volatility']:.6f}")
    
    # Évaluation du Sharpe Ratio
    print("\n" + "=" * 80)
    print("🎯 ÉVALUATION DU SHARPE RATIO")
    print("=" * 80)
    
    sharpe = metrics['Sharpe']
    
    if sharpe > 1.5:
        level = "🏆 EXCEPTIONNEL"
        comment = "Top 1% ! Vous avez un modèle de classe mondiale !"
    elif sharpe > 1.0:
        level = "🔵 EXCELLENT"
        comment = "Top 10% ! Performances remarquables !"
    elif sharpe > 0.7:
        level = "🟢 TRÈS BON"
        comment = "Top 25% ! Vous êtes très compétitif !"
    elif sharpe > 0.5:
        level = "🟡 BON"
        comment = "Baseline solide, continuez à optimiser !"
    elif sharpe > 0.3:
        level = "🟠 ACCEPTABLE"
        comment = "Baseline fonctionnel, mais optimisation nécessaire."
    else:
        level = "🔴 FAIBLE"
        comment = "Vérifiez les features et la stratégie."
    
    print(f"\n{level}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"→ {comment}")
    
    # 4. Créer les visualisations
    print("\n4️⃣  Génération des visualisations...")
    print("-" * 80)
    
    create_visualizations(predictions, actuals, val_data, metrics)
    
    # 5. Analyse par régime
    print("\n5️⃣  Analyse par régime de marché...")
    print("-" * 80)
    
    analyze_by_regime(predictions, actuals)
    
    # 6. Analyse des erreurs
    print("\n6️⃣  Analyse des erreurs...")
    print("-" * 80)
    
    analyze_errors(predictions, actuals)
    
    # 7. Analyse temporelle
    print("\n7️⃣  Analyse temporelle...")
    print("-" * 80)
    
    analyze_temporal_patterns(predictions, actuals, val_data)
    
    # 8. Recommandations
    print("\n8️⃣  Recommandations...")
    print("-" * 80)
    
    print_recommendations(metrics)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 80)
    
    print("\n📁 Fichiers générés:")
    print("   - performance_analysis.png")
    print("   - error_analysis.png")
    print("   - feature_importance.png (si disponible)")
    print("   - regime_analysis.png")
    
    return metrics


# ==============================================================================
# VISUALISATIONS
# ==============================================================================

def create_visualizations(predictions, actuals, val_data, metrics):
    """Créer les visualisations principales"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Scatter plot: Prédictions vs Actuals
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(actuals, predictions, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # Ligne de prédiction parfaite
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Returns', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Returns', fontsize=11, fontweight='bold')
    ax1.set_title(f'Predictions vs Actuals\nCorr: {metrics["Correlation"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Returns
    ax2 = plt.subplot(3, 3, 2)
    cumulative_pred = (1 + pd.Series(predictions)).cumprod()
    cumulative_actual = (1 + pd.Series(actuals)).cumprod()
    
    ax2.plot(cumulative_pred.values, label='Strategy (Predictions)', linewidth=2)
    ax2.plot(cumulative_actual.values, label='Market (Actuals)', linewidth=2, alpha=0.7)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Trading Days', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution des erreurs
    ax3 = plt.subplot(3, 3, 3)
    errors = predictions - actuals
    ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.6f}')
    
    ax3.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title(f'Error Distribution\nMAE: {metrics["MAE"]:.6f}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Rolling Sharpe Ratio
    ax4 = plt.subplot(3, 3, 4)
    window = 50
    
    rolling_sharpe_pred = pd.Series(predictions).rolling(window).apply(
        lambda x: calculate_sharpe_ratio(x) if len(x) == window else np.nan
    )
    rolling_sharpe_actual = pd.Series(actuals).rolling(window).apply(
        lambda x: calculate_sharpe_ratio(x) if len(x) == window else np.nan
    )
    
    ax4.plot(rolling_sharpe_pred.values, label='Strategy', linewidth=2)
    ax4.plot(rolling_sharpe_actual.values, label='Market', linewidth=2, alpha=0.7)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(1, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1.0')
    
    ax4.set_xlabel(f'Time (rolling window = {window} days)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax4.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Drawdown
    ax5 = plt.subplot(3, 3, 5)
    cumulative = (1 + pd.Series(predictions)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    ax5.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, color='red')
    ax5.plot(drawdown.values, color='darkred', linewidth=1.5)
    
    ax5.set_xlabel('Trading Days', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Drawdown', fontsize=11, fontweight='bold')
    ax5.set_title(f'Drawdown\nMax DD: {metrics["Max_Drawdown"]:.2%}', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution Prédictions vs Actuals
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(actuals, bins=50, alpha=0.5, label='Actuals', color='blue', edgecolor='black')
    ax6.hist(predictions, bins=50, alpha=0.5, label='Predictions', color='red', edgecolor='black')
    ax6.axvline(0, color='black', linestyle='--', linewidth=1)
    
    ax6.set_xlabel('Returns', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. QQ Plot
    ax7 = plt.subplot(3, 3, 7)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax7)
    ax7.set_title('Q-Q Plot (Error Normality)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Residuals vs Predicted
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(predictions, errors, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax8.axhline(0, color='red', linestyle='--', linewidth=2)
    
    # Ligne de tendance
    z = np.polyfit(predictions, errors, 1)
    p = np.poly1d(z)
    ax8.plot(predictions, p(predictions), "g--", alpha=0.8, linewidth=2, label='Trend')
    
    ax8.set_xlabel('Predicted Returns', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax8.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Métriques Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    📊 MÉTRIQUES CLÉS
    
    Sharpe Ratio:     {metrics['Sharpe']:.4f}
    Sortino Ratio:    {metrics['Sortino']:.4f}
    Hit Rate:         {metrics['Hit_Rate']:.2%}
    
    MAE:              {metrics['MAE']:.6f}
    RMSE:             {metrics['RMSE']:.6f}
    Correlation:      {metrics['Correlation']:.4f}
    
    Max Drawdown:     {metrics['Max_Drawdown']:.2%}
    Volatility:       {metrics['Volatility']:.6f}
    
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ performance_analysis.png sauvegardé")
    plt.close()


def analyze_by_regime(predictions, actuals):
    """Analyser les performances par régime de marché"""
    
    # Définir les régimes
    regimes = pd.cut(actuals, 
                     bins=[-np.inf, -0.005, 0.005, np.inf],
                     labels=['Bear', 'Neutral', 'Bull'])
    
    print("\n📊 Performance par Régime:\n")
    
    regime_metrics = []
    
    for regime in ['Bear', 'Neutral', 'Bull']:
        mask = regimes == regime
        if mask.sum() == 0:
            continue
        
        pred_regime = predictions[mask]
        actual_regime = actuals[mask]
        
        sharpe = calculate_sharpe_ratio(pred_regime)
        hit_rate = calculate_hit_rate(pred_regime, actual_regime)
        mae = np.mean(np.abs(pred_regime - actual_regime))
        correlation = np.corrcoef(pred_regime, actual_regime)[0, 1]
        
        print(f"  {regime:8s} ({mask.sum():4d} jours):")
        print(f"    Sharpe      : {sharpe:7.4f}")
        print(f"    Hit Rate    : {hit_rate:7.2%}")
        print(f"    MAE         : {mae:.6f}")
        print(f"    Correlation : {correlation:.4f}")
        print()
        
        regime_metrics.append({
            'Regime': regime,
            'Days': mask.sum(),
            'Sharpe': sharpe,
            'Hit_Rate': hit_rate
        })
    
    # Visualisation par régime
    if regime_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        df_regimes = pd.DataFrame(regime_metrics)
        
        # Sharpe par régime
        axes[0].bar(df_regimes['Regime'], df_regimes['Sharpe'], 
                   color=['red', 'gray', 'green'], alpha=0.7, edgecolor='black')
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[0].set_ylabel('Sharpe Ratio', fontweight='bold')
        axes[0].set_title('Sharpe Ratio par Régime', fontweight='bold', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Hit Rate par régime
        axes[1].bar(df_regimes['Regime'], df_regimes['Hit_Rate'] * 100,
                   color=['red', 'gray', 'green'], alpha=0.7, edgecolor='black')
        axes[1].axhline(50, color='black', linestyle='--', linewidth=1, label='Random (50%)')
        axes[1].set_ylabel('Hit Rate (%)', fontweight='bold')
        axes[1].set_title('Hit Rate par Régime', fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✓ regime_analysis.png sauvegardé")
        plt.close()


def analyze_errors(predictions, actuals):
    """Analyser les erreurs en détail"""
    
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    
    print("\n📊 Analyse des Erreurs:\n")
    
    # Statistiques des erreurs
    print("  Distribution:")
    print(f"    Mean Error      : {errors.mean():.6f}")
    print(f"    Std Error       : {errors.std():.6f}")
    print(f"    Skewness        : {pd.Series(errors).skew():.4f}")
    print(f"    Kurtosis        : {pd.Series(errors).kurtosis():.4f}")
    
    # Erreurs extrêmes
    threshold_95 = np.percentile(abs_errors, 95)
    extreme_errors = abs_errors > threshold_95
    
    print(f"\n  Erreurs Extrêmes (top 5%):")
    print(f"    Nombre          : {extreme_errors.sum()}")
    print(f"    Seuil           : {threshold_95:.6f}")
    print(f"    Max Error       : {abs_errors.max():.6f}")
    
    # Visualisation des erreurs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram des erreurs absolues
    axes[0, 0].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 0].axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {abs_errors.mean():.6f}')
    axes[0, 0].axvline(abs_errors.median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {abs_errors.median():.6f}')
    axes[0, 0].set_xlabel('Absolute Error', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Absolute Error Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Erreurs dans le temps
    axes[0, 1].plot(errors, alpha=0.7, linewidth=1)
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].fill_between(range(len(errors)), errors, 0, alpha=0.3)
    axes[0, 1].set_xlabel('Trading Days', fontweight='bold')
    axes[0, 1].set_ylabel('Error', fontweight='bold')
    axes[0, 1].set_title('Errors Over Time', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Autocorrélation des erreurs
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(pd.Series(errors), ax=axes[1, 0])
    axes[1, 0].set_title('Error Autocorrelation', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Erreurs par magnitude de prédiction
    axes[1, 1].scatter(np.abs(predictions), abs_errors, alpha=0.5, s=20, 
                      edgecolors='k', linewidth=0.5)
    axes[1, 1].set_xlabel('|Predicted Returns|', fontweight='bold')
    axes[1, 1].set_ylabel('Absolute Error', fontweight='bold')
    axes[1, 1].set_title('Error vs Prediction Magnitude', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ error_analysis.png sauvegardé")
    plt.close()


def analyze_temporal_patterns(predictions, actuals, val_data):
    """Analyser les patterns temporels"""
    
    print("\n📊 Analyse Temporelle:\n")
    
    # Performance par période
    n_periods = 4
    period_size = len(predictions) // n_periods
    
    print(f"  Performance par période ({n_periods} périodes):\n")
    
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else len(predictions)
        
        pred_period = predictions[start_idx:end_idx]
        actual_period = actuals[start_idx:end_idx]
        
        sharpe = calculate_sharpe_ratio(pred_period)
        hit_rate = calculate_hit_rate(pred_period, actual_period)
        
        date_start = val_data.iloc[start_idx]['date_id']
        date_end = val_data.iloc[end_idx-1]['date_id']
        
        print(f"    Période {i+1} (date_id {date_start}-{date_end}):")
        print(f"      Sharpe    : {sharpe:.4f}")
        print(f"      Hit Rate  : {hit_rate:.2%}")


def print_recommendations(metrics):
    """Afficher des recommandations basées sur les métriques"""
    
    print("\n💡 RECOMMANDATIONS\n")
    
    sharpe = metrics['Sharpe']
    hit_rate = metrics['Hit_Rate']
    correlation = metrics['Correlation']
    
    recommendations = []
    
    # Sharpe Ratio
    if sharpe < 0.3:
        recommendations.append({
            'priority': '🔴 HAUTE',
            'issue': f'Sharpe Ratio faible ({sharpe:.4f})',
            'actions': [
                'Vérifier l\'absence de look-ahead bias',
                'Augmenter le poids des lagged features',
                'Essayer des modèles plus complexes (ensemble)',
                'Optimiser les hyperparamètres avec Optuna'
            ]
        })
    elif sharpe < 0.5:
        recommendations.append({
            'priority': '🟡 MOYENNE',
            'issue': f'Sharpe Ratio modéré ({sharpe:.4f})',
            'actions': [
                'Feature engineering plus agressif',
                'Tuning des hyperparamètres',
                'Tester différentes fenêtres de rolling features'
            ]
        })
    
    # Hit Rate
    if hit_rate < 0.52:
        recommendations.append({
            'priority': '🟡 MOYENNE',
            'issue': f'Hit Rate proche du hasard ({hit_rate:.2%})',
            'actions': [
                'Focus sur features de direction (momentum, sentiment)',
                'Créer des features de régime de marché',
                'Essayer un modèle de classification pour la direction'
            ]
        })
    
    # Correlation
    if correlation < 0.15:
        recommendations.append({
            'priority': '🟡 MOYENNE',
            'issue': f'Corrélation faible ({correlation:.4f})',
            'actions': [
                'Vérifier la qualité des features',
                'Augmenter la complexité du modèle',
                'Feature selection plus agressive'
            ]
        })
    
    # Si tout va bien
    if sharpe > 0.7 and hit_rate > 0.53 and correlation > 0.25:
        recommendations.append({
            'priority': '✅ SUCCÈS',
            'issue': 'Excellentes performances !',
            'actions': [
                'Fine-tuning des hyperparamètres',
                'Ensemble methods (stacking/blending)',
                'Features d\'interaction avancées',
                'SOUMETTRE À KAGGLE ! 🚀'
            ]
        })
    
    # Afficher les recommandations
    if recommendations:
        for rec in recommendations:
            print(f"{rec['priority']} - {rec['issue']}")
            print("  Actions suggérées:")
            for action in rec['actions']:
                print(f"    • {action}")
            print()
    
    # Prochaines étapes générales
    print("📋 Prochaines étapes générales:")
    
    if sharpe > 0.5:
        print("  1. ✅ Tester localement: python test_local.py --mode full")
        print("  2. 🚀 SOUMETTRE À KAGGLE !")
        print("  3. 📊 (Optionnel) Continuer à optimiser")
    else:
        print("  1. 🔧 Implémenter les recommandations ci-dessus")
        print("  2. 🔄 Re-entraîner: python baseline_model.py")
        print("  3. 📊 Re-analyser: python analyze_results.py")
        print("  4. 🧪 Tester: python test_local.py --mode full")


# ==============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

def analyze_feature_importance():
    """Analyser l'importance des features si disponible"""
    
    if not Path('feature_importance.csv').exists():
        print("\n⚠️  feature_importance.csv non trouvé")
        return
    
    print("\n9️⃣  Analyse de l'importance des features...")
    print("-" * 80)
    
    fi = pd.read_csv('feature_importance.csv')
    
    print("\n🔝 Top 20 Features:")
    print(fi.head(20).to_string(index=False))
    
    # Analyser les types de features importantes
    top_20 = fi.head(20)
    
    feature_types = {
        'lagged': 0,
        'volatility': 0,
        'momentum': 0,
        'sentiment': 0,
        'price': 0,
        'interest': 0,
        'economic': 0,
        'other': 0
    }
    
    for feature in top_20['feature']:
        if 'lagged' in feature or 'target' in feature:
            feature_types['lagged'] += 1
        elif feature.startswith('V') or 'vol' in feature:
            feature_types['volatility'] += 1
        elif feature.startswith('M') or 'momentum' in feature:
            feature_types['momentum'] += 1
        elif feature.startswith('S') or 'sentiment' in feature:
            feature_types['sentiment'] += 1
        elif feature.startswith('P') or 'price' in feature:
            feature_types['price'] += 1
        elif feature.startswith('I') or 'interest' in feature:
            feature_types['interest'] += 1
        elif feature.startswith('E') or 'economic' in feature:
            feature_types['economic'] += 1
        else:
            feature_types['other'] += 1
    
    print("\n📊 Distribution des types (Top 20):")
    for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / 20 * 100
            print(f"   {ftype.capitalize():12s}: {count:2d} ({pct:5.1f}%)")
    
    # Vérifier que les lagged features sont bien utilisées
    if feature_types['lagged'] >= 5:
        print("\n✅ Excellent ! Les lagged features dominent le top 20.")
    elif feature_types['lagged'] >= 3:
        print("\n✓ Bon. Les lagged features sont bien présentes.")
    else:
        print("\n⚠️  Peu de lagged features dans le top 20.")
        print("   → Vérifiez que le preprocessing crée bien ces features.")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 20 features
    top_20_sorted = fi.head(20).sort_values('importance')
    axes[0].barh(range(len(top_20_sorted)), top_20_sorted['importance'])
    axes[0].set_yticks(range(len(top_20_sorted)))
    axes[0].set_yticklabels(top_20_sorted['feature'])
    axes[0].set_xlabel('Importance', fontweight='bold')
    axes[0].set_title('Top 20 Feature Importance', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Distribution par type
    types_data = [(k, v) for k, v in feature_types.items() if v > 0]
    types_data.sort(key=lambda x: x[1], reverse=True)
    
    axes[1].bar([t[0] for t in types_data], [t[1] for t in types_data],
               color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('Feature Types in Top 20', fontweight='bold', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n   ✓ feature_importance.png sauvegardé")
    plt.close()


# ==============================================================================
# COMPARAISON AVEC DES BENCHMARKS
# ==============================================================================

def compare_with_benchmarks(metrics):
    """Comparer avec des benchmarks de référence"""
    
    print("\n🎯 COMPARAISON AVEC BENCHMARKS")
    print("-" * 80 + "\n")
    
    benchmarks = {
        'Random': {'Sharpe': 0.0, 'Hit_Rate': 0.50, 'description': 'Prédictions aléatoires'},
        'Naive (Mean)': {'Sharpe': 0.1, 'Hit_Rate': 0.50, 'description': 'Toujours prédire la moyenne'},
        'Simple Lag': {'Sharpe': 0.3, 'Hit_Rate': 0.51, 'description': 'Utiliser seulement lag-1'},
        'Good Model': {'Sharpe': 0.7, 'Hit_Rate': 0.54, 'description': 'Modèle compétitif'},
        'Excellent Model': {'Sharpe': 1.0, 'Hit_Rate': 0.56, 'description': 'Top 10%'},
    }
    
    your_sharpe = metrics['Sharpe']
    your_hit_rate = metrics['Hit_Rate']
    
    print("  Votre modèle:")
    print(f"    Sharpe    : {your_sharpe:.4f}")
    print(f"    Hit Rate  : {your_hit_rate:.2%}")
    print()
    
    # Trouver où vous vous situez
    position = "Entre Random et Simple Lag"
    for name, bench in sorted(benchmarks.items(), key=lambda x: x[1]['Sharpe']):
        if your_sharpe >= bench['Sharpe']:
            position = f"Au-dessus de '{name}'"
    
    print(f"  Position: {position}")
    print()
    
    print("  Benchmarks de référence:")
    for name, bench in sorted(benchmarks.items(), key=lambda x: x[1]['Sharpe']):
        indicator = "✓" if your_sharpe >= bench['Sharpe'] else " "
        print(f"    {indicator} {name:20s}: Sharpe {bench['Sharpe']:.2f}, Hit {bench['Hit_Rate']:.0%} - {bench['description']}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    
    print("\n" + "🔍" * 40)
    print("ANALYSE COMPLÈTE DES PERFORMANCES")
    print("🔍" * 40 + "\n")
    
    # Vérifier les prérequis
    if not Path('train.csv').exists():
        print("❌ train.csv non trouvé")
        print("   Téléchargez les données Kaggle d'abord.")
        exit(1)
    
    if not Path('xgb_model.pkl').exists() or not Path('preprocessor.pkl').exists():
        print("❌ Modèle ou preprocessor non trouvé")
        print("   Exécutez d'abord: python baseline_model.py")
        exit(1)
    
    try:
        # Analyse principale
        metrics = analyze_validation_performance()
        
        if metrics is None:
            print("\n❌ Échec de l'analyse")
            exit(1)
        
        # Analyse de feature importance
        analyze_feature_importance()
        
        # Comparaison avec benchmarks
        compare_with_benchmarks(metrics)
        
        print("\n" + "=" * 80)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        print("=" * 80)
        
        print("\n📊 Résumé des Métriques:")
        print(f"   Sharpe Ratio     : {metrics['Sharpe']:.4f}")
        print(f"   Hit Rate         : {metrics['Hit_Rate']:.2%}")
        print(f"   Correlation      : {metrics['Correlation']:.4f}")
        print(f"   Max Drawdown     : {metrics['Max_Drawdown']:.2%}")
        
        print("\n📁 Fichiers générés:")
        generated_files = []
        for file in ['performance_analysis.png', 'error_analysis.png', 
                     'regime_analysis.png', 'feature_importance.png']:
            if Path(file).exists():
                generated_files.append(file)
                print(f"   ✓ {file}")
        
        if metrics['Sharpe'] > 0.7:
            print("\n🚀 PRÊT POUR LA SOUMISSION !")
            print("   Exécutez: python test_local.py --mode full")
            print("   Puis soumettez à Kaggle !")
        elif metrics['Sharpe'] > 0.5:
            print("\n✓ Performances correctes.")
            print("   Vous pouvez soumettre ou continuer à optimiser.")
        else:
            print("\n📈 Optimisation recommandée.")
            print("   Consultez les recommandations ci-dessus.")
        
        exit(0)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERREUR LORS DE L'ANALYSE")
        print("=" * 80)
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Suggestions:")
        print("  1. Vérifiez que train.csv existe et est valide")
        print("  2. Vérifiez que le modèle a été entraîné correctement")
        print("  3. Vérifiez que tous les packages sont installés (matplotlib, seaborn)")
        
        exit(1)