"""Page Interactive de Prétraitement & Modélisation — Hull Tactical."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── sys.path ──────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent.parent
_ROOT    = _APP_DIR.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="Prétraitement Interactif — Hull Tactical",
    page_icon="⚙️",
    layout="wide",
)

# ── Dark CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"]*{color:#cbd5e1;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);
    border:1px solid #1e3a5f;border-radius:12px;padding:1rem;
    box-shadow:0 4px 16px rgba(0,212,255,.08);}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700;}
.section-title{
    font-size:1rem;font-weight:700;color:#00d4ff;
    letter-spacing:.05em;text-transform:uppercase;
    margin-bottom:.8rem;padding-bottom:.3rem;border-bottom:1px solid #1e3a5f;
}
.step-card{
    background:linear-gradient(135deg,#0f1f3d 0%,#1a2f4e 100%);
    border:1px solid #1e3a5f;border-left:3px solid #00d4ff;
    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.2rem;
}
.step-num{
    display:inline-flex;align-items:center;justify-content:center;
    background:#00d4ff;color:#0a0e1a;border-radius:50%;
    width:26px;height:26px;font-weight:800;font-size:.85rem;
    margin-right:.6rem;flex-shrink:0;
}
.info-pill{
    display:inline-block;background:rgba(0,212,255,.09);
    border:1px solid rgba(0,212,255,.25);color:#00d4ff;
    border-radius:20px;padding:.2rem .8rem;font-size:.78rem;font-weight:600;
}
.warn-pill{
    display:inline-block;background:rgba(245,158,11,.09);
    border:1px solid rgba(245,158,11,.25);color:#f59e0b;
    border-radius:20px;padding:.2rem .8rem;font-size:.78rem;font-weight:600;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0", family="Inter,sans-serif"),
)

from utils.artifacts import load_train_data, feature_category  # noqa: E402

# ── Constantes ────────────────────────────────────────────────────────────────
TARGET_COL   = "market_forward_excess_returns"
EXCLUDE_COLS = {"date_id", TARGET_COL}

CAT_COLORS = {
    "Forward Returns": "#00d4ff", "Price": "#f59e0b",   "Economic": "#10b981",
    "Momentum":        "#7b5ea7", "Volatility": "#ef4444", "Sentiment": "#06b6d4",
    "Engineered":      "#64748b", "Lagged Target": "#a3e635", "Interest": "#fb923c",
    "Target Stats":    "#e879f9", "Raw": "#94a3b8",
}

# ── Chargement des données ────────────────────────────────────────────────────
df_full = load_train_data()

st.title("⚙️ Prétraitement Interactif")
st.markdown(
    "Pipeline pas-à-pas : **sélection de l'intervalle** → "
    "**gestion des NaN** → **imputation** → **importance des variables**."
)
st.markdown("---")

if df_full is None:
    st.error("❌ Données d'entraînement non trouvées. Lancez d'abord le téléchargement Kaggle.")
    st.stop()

N_TOTAL = len(df_full)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Sélection de l'intervalle d'entraînement
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>1</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Sélection de l'intervalle d'entraînement
        </span>
    </div>
    <div style='color:#64748b;font-size:.85rem;margin-top:.4rem;margin-left:2rem;'>
        Définissez un sous-ensemble continu du dataset en pourcentage (0 % → 100 %).
        L'intervalle doit être <strong style='color:#f59e0b;'>strictement positif</strong>
        (cutoff supérieur &gt; cutoff inférieur).
    </div>
</div>
""", unsafe_allow_html=True)

col_c1, col_c2, col_c3 = st.columns([2, 2, 3])

with col_c1:
    lower_pct = st.number_input(
        "Cutoff inférieur (%)",
        min_value=0.0,
        max_value=99.9,
        value=0.0,
        step=0.5,
        format="%.1f",
        help=f"Position de départ dans le dataset (0 % = ligne 0, dataset total = {N_TOTAL:,} lignes)",
    )

with col_c2:
    upper_pct = st.number_input(
        "Cutoff supérieur (%)",
        min_value=0.1,
        max_value=100.0,
        value=100.0,
        step=0.5,
        format="%.1f",
        help="Position de fin dans le dataset (100 % = dernière ligne)",
    )

# ── Validation : intervalle strictement positif ───────────────────────────────
interval_size_pct = upper_pct - lower_pct
if interval_size_pct <= 0:
    with col_c3:
        st.error(
            f"❌ Intervalle invalide : le cutoff supérieur ({upper_pct:.1f} %) "
            f"doit être strictement supérieur au cutoff inférieur ({lower_pct:.1f} %)."
        )
    st.stop()

lower_idx = int(N_TOTAL * lower_pct / 100)
upper_idx = int(N_TOTAL * upper_pct / 100)
upper_idx = max(upper_idx, lower_idx + 1)   # garantit au moins 1 ligne

df_slice = df_full.iloc[lower_idx:upper_idx].copy().reset_index(drop=True)
n_slice  = len(df_slice)

with col_c3:
    st.markdown(f"""
    <div style='background:rgba(0,212,255,.07);border:1px solid #1e3a5f;
                border-radius:8px;padding:.8rem 1rem;margin-top:1.6rem;'>
        <div style='color:#94a3b8;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;'>
            Données sélectionnées
        </div>
        <div style='color:#00d4ff;font-weight:800;font-size:1.25rem;margin:.2rem 0;'>
            {n_slice:,} lignes
        </div>
        <div style='color:#64748b;font-size:.78rem;'>
            Lignes&nbsp;{lower_idx:,}&nbsp;→&nbsp;{upper_idx:,}
            &nbsp;|&nbsp;{lower_pct:.1f}%&nbsp;→&nbsp;{upper_pct:.1f}%
            &nbsp;|&nbsp;Taille&nbsp;intervalle&nbsp;:&nbsp;{interval_size_pct:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# Barre de progression visuelle de l'intervalle
st.markdown("<br>", unsafe_allow_html=True)
bar_left  = f"{lower_pct:.1f}%"
bar_right = f"{upper_pct:.1f}%"
bar_width = f"{interval_size_pct:.1f}%"
bar_start = f"{lower_pct:.1f}%"
st.markdown(f"""
<div style='background:#1e293b;border-radius:8px;height:28px;position:relative;
            border:1px solid #1e3a5f;overflow:hidden;margin-bottom:0.3rem;'>
    <div style='position:absolute;left:{bar_start};width:{bar_width};height:100%;
                background:linear-gradient(90deg,#00d4ff55,#00d4ff99);border-radius:4px;'></div>
    <div style='position:absolute;left:2px;top:50%;transform:translateY(-50%);
                color:#475569;font-size:.75rem;'>0%</div>
    <div style='position:absolute;right:2px;top:50%;transform:translateY(-50%);
                color:#475569;font-size:.75rem;'>100%</div>
</div>
<div style='display:flex;justify-content:space-between;color:#94a3b8;font-size:.75rem;'>
    <span>↑ {bar_left}</span><span>{bar_right} ↑</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Valeurs manquantes & Suppression des colonnes
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>2</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Gestion des valeurs manquantes
        </span>
    </div>
    <div style='color:#64748b;font-size:.85rem;margin-top:.4rem;margin-left:2rem;'>
        Choisissez un seuil de suppression. Toute colonne dépassant ce seuil est éliminée automatiquement.
        Le tableau ci-dessous liste toutes les colonnes avec des NaN, triées par ordre décroissant.
    </div>
</div>
""", unsafe_allow_html=True)

col_t1, col_t2 = st.columns([1, 3])
with col_t1:
    nan_threshold_pct = st.number_input(
        "Seuil de suppression (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="Colonnes dont le % NaN > seuil → supprimées du DataFrame",
    )

with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.3);
                border-radius:8px;padding:.65rem 1rem;'>
        ⚠️ Les colonnes avec plus de <strong style='color:#f59e0b;'>{nan_threshold_pct}%</strong>
        de valeurs manquantes seront <strong style='color:#ef4444;'>supprimées</strong> du DataFrame.
        Seuil par défaut : <strong>30%</strong> (paramètre utilisé en production).
    </div>
    """, unsafe_allow_html=True)

# ── Construction du tableau des NaN ──────────────────────────────────────────
nan_counts   = df_slice.isnull().sum()
has_nan_mask = nan_counts > 0
nan_counts_nz = nan_counts[has_nan_mask]
nan_pcts_nz   = (nan_counts_nz / n_slice * 100).round(2)

if len(nan_counts_nz) == 0:
    st.success("✅ Aucune valeur manquante dans l'intervalle sélectionné. Aucune colonne ne sera supprimée.")
    df_clean = df_slice.copy()
    cols_dropped = []
else:
    nan_table = pd.DataFrame({
        "Colonne":              nan_counts_nz.index.tolist(),
        "Valeurs manquantes":   nan_counts_nz.values,
        "% Manquant":           nan_pcts_nz.values,
        "Supprimée":            ["✅ Oui" if p > nan_threshold_pct else "❌ Non"
                                 for p in nan_pcts_nz.values],
    }).sort_values("% Manquant", ascending=False).reset_index(drop=True)

    # ── KPI ──────────────────────────────────────────────────────────────────
    n_dropped_cols   = (nan_pcts_nz > nan_threshold_pct).sum()
    n_kept_with_nan  = (nan_pcts_nz <= nan_threshold_pct).sum()

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.metric("Colonnes totales", df_slice.shape[1])
    with kc2:
        st.metric("Colonnes avec NaN", len(nan_counts_nz))
    with kc3:
        st.metric("Colonnes supprimées", int(n_dropped_cols),
                  delta=f"> {nan_threshold_pct}% NaN", delta_color="inverse")
    with kc4:
        st.metric("Conservées avec NaN", int(n_kept_with_nan),
                  delta="→ seront imputées")

    st.markdown("")

    # ── Tableau récapitulatif coloré ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Tableau des valeurs manquantes — tri décroissant</div>",
                unsafe_allow_html=True)

    def _color_nan_row(row):
        if row["Supprimée"] == "✅ Oui":
            # Rouge : sera supprimée
            return ["background-color:rgba(239,68,68,.18);color:#ef4444"] * len(row)
        # Orange doux : sera imputée
        return ["background-color:rgba(245,158,11,.08);color:#e2e8f0"] * len(row)

    styled_nan = (
        nan_table.style
        .apply(_color_nan_row, axis=1)
        .format({"% Manquant": "{:.2f}%", "Valeurs manquantes": "{:,}"})
    )
    st.dataframe(styled_nan, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='font-size:.78rem;color:#475569;margin-top:.3rem;'>
        🔴 <strong>Rouge</strong> = colonne supprimée (dépasse le seuil) &nbsp;|&nbsp;
        🟡 <strong>Orange</strong> = colonne conservée mais à imputer
    </div>
    """, unsafe_allow_html=True)

    # ── Graphique en barres des % NaN ────────────────────────────────────────
    bar_colors = [
        "#ef4444" if p > nan_threshold_pct else "#f59e0b"
        for p in nan_table["% Manquant"]
    ]
    fig_nan = go.Figure()
    fig_nan.add_trace(go.Bar(
        x=nan_table["Colonne"],
        y=nan_table["% Manquant"],
        marker=dict(color=bar_colors, line=dict(width=0.5, color="#0a0e1a")),
        text=[f"{p:.1f}%" for p in nan_table["% Manquant"]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=9),
        hovertemplate="<b>%{x}</b><br>% Manquant : %{y:.2f}%<extra></extra>",
    ))
    fig_nan.add_hline(
        y=nan_threshold_pct,
        line_dash="dash", line_color="#00d4ff", line_width=2,
        annotation_text=f"  Seuil : {nan_threshold_pct}%",
        annotation_font_color="#00d4ff",
        annotation_position="top right",
    )
    fig_nan.update_layout(
        **PLOTLY_DARK,
        height=320,
        title=f"% Valeurs manquantes par colonne — seuil = {nan_threshold_pct}%",
        xaxis_title="Colonne", yaxis_title="% Manquant",
        xaxis_tickangle=-45, showlegend=False,
        margin=dict(b=110, t=50),
    )
    st.plotly_chart(fig_nan, use_container_width=True)

    # ── Suppression effective ─────────────────────────────────────────────────
    cols_dropped = nan_table[nan_table["Supprimée"] == "✅ Oui"]["Colonne"].tolist()
    df_clean = df_slice.drop(columns=cols_dropped, errors="ignore")
    st.info(
        f"✅ DataFrame après suppression : **{df_clean.shape[0]:,} lignes × "
        f"{df_clean.shape[1]} colonnes**  —  {len(cols_dropped)} colonne(s) supprimée(s)."
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Méthode d'imputation
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>3</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Méthode d'imputation des valeurs manquantes
        </span>
    </div>
    <div style='color:#64748b;font-size:.85rem;margin-top:.4rem;margin-left:2rem;'>
        Sélectionnez la méthode appliquée sur les colonnes <em>conservées</em>
        après l'étape 2. Les méthodes sont issues du pipeline de production du projet.
    </div>
</div>
""", unsafe_allow_html=True)

IMPUTATION_METHODS = {
    "Médiane":               "Remplace les NaN par la médiane de chaque colonne. Robuste aux outliers.",
    "Moyenne":               "Remplace les NaN par la moyenne de chaque colonne. Simple et rapide.",
    "Forward Fill":          "Propage la dernière valeur valide vers l'avant — adapté aux séries temporelles.",
    "Backward Fill":         "Propage la prochaine valeur valide vers l'arrière — adapté aux séries temporelles.",
    "Interpolation linéaire":"Interpole linéairement entre les valeurs connues (axis=0).",
    "KNN (5 voisins)":       "Imputation par les 5 voisins les plus proches (scikit-learn KNNImputer).",
    "CatBoost":              "Imputation par modèle CatBoost — méthode utilisée en production (peut être lente).",
}

col_imp1, col_imp2 = st.columns([1, 2])
with col_imp1:
    imputation_method = st.selectbox(
        "Méthode d'imputation",
        list(IMPUTATION_METHODS.keys()),
        index=0,
        help="Appliquée uniquement aux colonnes conservées après l'étape 2",
    )
with col_imp2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(0,212,255,.05);border:1px solid #1e3a5f;
                border-radius:8px;padding:.65rem 1rem;'>
        ℹ️ <strong style='color:#00d4ff;'>{imputation_method}</strong> —
        {IMPUTATION_METHODS[imputation_method]}
    </div>
    """, unsafe_allow_html=True)

# ── Colonnes de features à imputer ───────────────────────────────────────────
feat_cols = [c for c in df_clean.columns if c not in EXCLUDE_COLS]
nan_before = int(df_clean[feat_cols].isnull().sum().sum())

if nan_before == 0:
    st.success("✅ Aucune valeur manquante dans les features — imputation non nécessaire.")
    df_imputed = df_clean.copy()
else:
    if imputation_method == "CatBoost":
        st.warning(
            "⚠️ CatBoost Imputer : le calcul peut prendre plusieurs minutes "
            f"({len(feat_cols)} colonnes × {n_slice:,} lignes). "
            "En cas d'échec, repli automatique sur la médiane."
        )

    with st.spinner(f"Imputation en cours : {imputation_method}…"):
        df_imputed = df_clean.copy()

        if imputation_method == "Médiane":
            medians = df_imputed[feat_cols].median()
            df_imputed[feat_cols] = df_imputed[feat_cols].fillna(medians)

        elif imputation_method == "Moyenne":
            means = df_imputed[feat_cols].mean()
            df_imputed[feat_cols] = df_imputed[feat_cols].fillna(means)

        elif imputation_method == "Forward Fill":
            df_imputed[feat_cols] = (
                df_imputed[feat_cols]
                .ffill()
                .fillna(df_imputed[feat_cols].median())
            )

        elif imputation_method == "Backward Fill":
            df_imputed[feat_cols] = (
                df_imputed[feat_cols]
                .bfill()
                .fillna(df_imputed[feat_cols].median())
            )

        elif imputation_method == "Interpolation linéaire":
            df_imputed[feat_cols] = (
                df_imputed[feat_cols]
                .interpolate(method="linear", axis=0, limit_direction="both")
                .fillna(df_imputed[feat_cols].median())
            )

        elif imputation_method == "KNN (5 voisins)":
            from sklearn.impute import KNNImputer
            imputer_knn = KNNImputer(n_neighbors=5)
            n_fit = min(2000, len(df_imputed))
            if len(df_imputed) > 2000:
                st.info(
                    f"ℹ️ KNN : ajustement sur {n_fit:,} lignes (sous-échantillon) "
                    f"pour la performance, puis application sur {len(df_imputed):,} lignes."
                )
                rng_idx = np.random.default_rng(42).choice(
                    len(df_imputed), n_fit, replace=False
                )
                imputer_knn.fit(df_imputed.iloc[rng_idx][feat_cols])
                df_imputed[feat_cols] = imputer_knn.transform(df_imputed[feat_cols])
            else:
                df_imputed[feat_cols] = imputer_knn.fit_transform(df_imputed[feat_cols])

        elif imputation_method == "CatBoost":
            try:
                from src.data.preprocessor import CatBoostImputer
                cb_imputer = CatBoostImputer(iterations=50, verbose=False)
                df_imputed[feat_cols] = cb_imputer.fit_transform(df_imputed[feat_cols])
            except Exception as exc:
                st.error(f"CatBoost échoué ({exc}). Repli sur la médiane.")
                medians = df_imputed[feat_cols].median()
                df_imputed[feat_cols] = df_imputed[feat_cols].fillna(medians)

        # Filet de sécurité : NaN résiduels → médiane
        still_nan = df_imputed[feat_cols].isnull().sum().sum()
        if still_nan > 0:
            fallback_medians = df_imputed[feat_cols].median().fillna(0)
            df_imputed[feat_cols] = df_imputed[feat_cols].fillna(fallback_medians)

    nan_after = int(df_imputed[feat_cols].isnull().sum().sum())

    ri1, ri2, ri3 = st.columns(3)
    with ri1:
        st.metric("NaN avant imputation", f"{nan_before:,}")
    with ri2:
        st.metric("NaN après imputation", f"{nan_after:,}",
                  delta=f"−{nan_before - nan_after:,}", delta_color="inverse")
    with ri3:
        st.metric("Méthode appliquée", imputation_method)

st.info(
    f"✅ DataFrame final : **{df_imputed.shape[0]:,} lignes × {df_imputed.shape[1]} colonnes** "
    f"({len(feat_cols)} features disponibles pour la modélisation)."
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Sélection & Importance des variables
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>4</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Sélection & Importance des variables pour la modélisation
        </span>
    </div>
    <div style='color:#64748b;font-size:.85rem;margin-top:.4rem;margin-left:2rem;'>
        Choisissez le nombre de variables et la méthode de calcul de l'importance.
        Le graphique se met à jour dynamiquement.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Features disponibles ──────────────────────────────────────────────────────
avail_features = [c for c in feat_cols if c in df_imputed.columns]
n_avail        = len(avail_features)

if n_avail < 2:
    st.warning("⚠️ Moins de 2 features disponibles après prétraitement. Élargissez l'intervalle ou réduisez le seuil NaN.")
    st.stop()

# ── Contrôles ──────────────────────────────────────────────────────────────────
col_s1, col_s2 = st.columns([3, 2])

with col_s1:
    n_features_selected = st.slider(
        f"Nombre de variables à afficher (1 → {n_avail})",
        min_value=1,
        max_value=n_avail,
        value=min(20, n_avail),
        step=1,
        help=f"{n_avail} features disponibles après imputation",
    )

with col_s2:
    importance_method = st.selectbox(
        "Méthode de calcul de l'importance",
        ["Random Forest", "Permutation Importance", "Régression Ridge (|coef|)"],
        help=(
            "Random Forest : feature_importances_ (gain moyen) | "
            "Permutation : baisse de performance quand la feature est mélangée | "
            "Ridge : valeur absolue des coefficients standardisés"
        ),
    )

# ── Données de modélisation ───────────────────────────────────────────────────
valid_mask = df_imputed[TARGET_COL].notna()
X_all = df_imputed.loc[valid_mask, avail_features].fillna(0)
y_all = df_imputed.loc[valid_mask, TARGET_COL].values

if len(X_all) < 30:
    st.warning(f"⚠️ Seulement {len(X_all)} lignes avec une target valide — données insuffisantes pour la modélisation.")
    st.stop()

# ── Clé de cache session (recompute seulement si paramètres changent) ─────────
cache_key = (
    f"{importance_method}|{lower_pct:.1f}|{upper_pct:.1f}|"
    f"{nan_threshold_pct}|{imputation_method}|{n_avail}|{len(X_all)}"
)
if "imp_cache" not in st.session_state:
    st.session_state.imp_cache = {}

if cache_key not in st.session_state.imp_cache:
    with st.spinner(f"Calcul de l'importance des variables ({importance_method})…"):
        try:
            # Sous-échantillon pour la performance (≤ 3000 lignes)
            n_sample = min(3000, len(X_all))
            rng      = np.random.default_rng(42)
            idx_s    = rng.choice(len(X_all), n_sample, replace=False)
            X_s      = X_all.iloc[idx_s]
            y_s      = y_all[idx_s]

            if importance_method == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                mdl = RandomForestRegressor(
                    n_estimators=80, max_depth=6,
                    random_state=42, n_jobs=-1,
                )
                mdl.fit(X_s, y_s)
                imp_vals = mdl.feature_importances_

            elif importance_method == "Permutation Importance":
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.inspection import permutation_importance as _perm_imp
                mdl = RandomForestRegressor(
                    n_estimators=40, max_depth=5,
                    random_state=42, n_jobs=-1,
                )
                mdl.fit(X_s, y_s)
                perm = _perm_imp(mdl, X_s, y_s, n_repeats=5, random_state=42, n_jobs=-1)
                imp_vals = np.maximum(perm.importances_mean, 0.0)

            elif importance_method == "Régression Ridge (|coef|)":
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(X_all)
                mdl      = Ridge(alpha=1.0)
                mdl.fit(X_scaled, y_all)
                imp_vals = np.abs(mdl.coef_)

            imp_df_full = pd.DataFrame({
                "feature":    avail_features,
                "importance": imp_vals,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            st.session_state.imp_cache[cache_key] = imp_df_full

        except Exception as exc:
            st.error(f"Erreur lors du calcul de l'importance : {exc}")
            st.stop()

imp_df_full = st.session_state.imp_cache[cache_key]

# ── Top N features ────────────────────────────────────────────────────────────
top_n = imp_df_full.head(n_features_selected).copy()
top_n["Category"] = top_n["feature"].apply(feature_category)
top_n["color"]    = top_n["Category"].map(lambda c: CAT_COLORS.get(c, "#64748b"))

# Normalisation pour l'affichage (max = 100)
max_imp = top_n["importance"].max() or 1.0
top_n["imp_norm"] = top_n["importance"] / max_imp * 100

# ── Graphique ─────────────────────────────────────────────────────────────────
fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    x=top_n["imp_norm"].values[::-1],
    y=top_n["feature"].values[::-1],
    orientation="h",
    marker=dict(
        color=top_n["color"].values[::-1],
        line=dict(width=0.5, color="#0a0e1a"),
    ),
    text=[f"{v:.1f}" for v in top_n["imp_norm"].values[::-1]],
    textposition="outside",
    textfont=dict(color="#94a3b8", size=9),
    customdata=np.stack([
        top_n["Category"].values[::-1],
        top_n["importance"].values[::-1],
    ], axis=-1),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Catégorie : %{customdata[0]}<br>"
        "Score brut : %{customdata[1]:.8f}<br>"
        "Score normalisé : %{x:.2f}<extra></extra>"
    ),
))
chart_height = max(350, n_features_selected * 24 + 80)
fig_imp.update_layout(
    **PLOTLY_DARK,
    height=chart_height,
    title=dict(
        text=f"Top {n_features_selected} variables — <b>{importance_method}</b>",
        font=dict(size=14),
    ),
    xaxis_title="Score d'importance (normalisé, max = 100)",
    yaxis=dict(tickfont=dict(size=10 if n_features_selected <= 30 else 8)),
    margin=dict(l=260, r=80, t=60, b=40),
    showlegend=False,
)
st.plotly_chart(fig_imp, use_container_width=True)

# ── Métriques résumé ──────────────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Features sélectionnées", n_features_selected, f"/{n_avail} disponibles")
with col_m2:
    top_feat = top_n.iloc[0]["feature"] if len(top_n) else "—"
    label    = (top_feat[:16] + "…") if len(top_feat) > 18 else top_feat
    st.metric("Feature #1", label)
with col_m3:
    top_raw  = top_n.iloc[0]["importance"] if len(top_n) else 0.0
    st.metric("Score brut #1", f"{top_raw:.6f}")
with col_m4:
    n_cats = int(top_n["Category"].nunique())
    st.metric("Catégories représentées", n_cats, f"/{len(CAT_COLORS)}")

# ── Répartition par catégorie ─────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_pie2, col_bar2 = st.columns(2)

cat_dist = (
    top_n.groupby("Category")["importance"]
    .agg(count="count", total="sum")
    .reset_index()
    .sort_values("count", ascending=False)
)
cat_dist["color"] = cat_dist["Category"].map(lambda c: CAT_COLORS.get(c, "#64748b"))

with col_pie2:
    st.markdown("<div class='section-title'>Répartition par catégorie (nombre)</div>",
                unsafe_allow_html=True)
    fig_pie_cat = go.Figure(go.Pie(
        labels=cat_dist["Category"],
        values=cat_dist["count"],
        marker=dict(colors=cat_dist["color"].tolist(),
                    line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
        hole=0.42,
        hovertemplate="<b>%{label}</b><br>Nb features : %{value}<br>%{percent}<extra></extra>",
    ))
    fig_pie_cat.update_layout(
        **PLOTLY_DARK, height=300,
        showlegend=False,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_pie_cat, use_container_width=True)

with col_bar2:
    st.markdown("<div class='section-title'>Score total d'importance par catégorie</div>",
                unsafe_allow_html=True)
    cat_dist_sorted = cat_dist.sort_values("total", ascending=False)
    fig_cat_bar = go.Figure(go.Bar(
        x=cat_dist_sorted["Category"],
        y=cat_dist_sorted["total"],
        marker=dict(color=cat_dist_sorted["color"].tolist(),
                    line=dict(width=0.5, color="#0a0e1a")),
        text=[f"{v:.4f}" for v in cat_dist_sorted["total"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=9),
        hovertemplate="<b>%{x}</b><br>Score total : %{y:.6f}<extra></extra>",
    ))
    fig_cat_bar.update_layout(
        **PLOTLY_DARK, height=300,
        xaxis_title="Catégorie", yaxis_title="Σ Importance",
        xaxis_tickangle=-30, showlegend=False,
        margin=dict(t=10, b=60),
    )
    st.plotly_chart(fig_cat_bar, use_container_width=True)

# ── Tableau détaillé ──────────────────────────────────────────────────────────
with st.expander("📋 Tableau détaillé des features sélectionnées"):
    display_df = top_n[["feature", "Category", "importance", "imp_norm"]].copy()
    display_df.columns   = ["Feature", "Catégorie", "Score brut", "Score normalisé"]
    display_df.index     = range(1, len(display_df) + 1)

    styled_imp = (
        display_df.style
        .format({"Score brut": "{:.8f}", "Score normalisé": "{:.2f}"})
        .background_gradient(subset=["Score normalisé"], cmap="Blues")
    )
    st.dataframe(styled_imp, use_container_width=True)

# ── Export CSV ────────────────────────────────────────────────────────────────
export_df  = top_n[["feature", "Category", "importance", "imp_norm"]].copy()
export_df.columns = ["Feature", "Catégorie", "Importance_brute", "Importance_normalisée"]
csv_export = export_df.to_csv(index=False)
st.download_button(
    "⬇️ Télécharger la sélection de features (CSV)",
    csv_export,
    file_name="features_selectionnees.csv",
    mime="text/csv",
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#334155;font-size:.78rem;padding:.4rem 0;'>
    Pipeline Prétraitement Interactif — Hull Tactical Market Prediction &nbsp;|&nbsp;
    Étapes : Cutoff → NaN → Imputation → Importance
</div>
""", unsafe_allow_html=True)
