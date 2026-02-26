"""
Interactive Preprocessing Page — Hull Tactical.

Central control panel that synchronises preprocessing parameters
with the full training pipeline:
  • Data interval selection (cutoff)
  • Missing-value drop threshold
  • Imputation method
  • Number of features & feature importance
  • Config save → artifacts/preprocessing_config.json
  • Retrain trigger → scripts/retrain.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── sys.path ──────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent.parent
_ROOT    = _APP_DIR.parent
for _p in [str(_APP_DIR), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(
    page_title="Preprocessing — Hull Tactical",
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
.section-title{font-size:1rem;font-weight:700;color:#00d4ff;letter-spacing:.05em;
    text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.3rem;border-bottom:1px solid #1e3a5f;}
.step-card{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);
    border:1px solid #1e3a5f;border-left:3px solid #00d4ff;
    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.2rem;}
.step-num{display:inline-flex;align-items:center;justify-content:center;
    background:#00d4ff;color:#0a0e1a;border-radius:50%;
    width:26px;height:26px;font-weight:800;font-size:.85rem;margin-right:.6rem;}
.prod-card{background:linear-gradient(135deg,#1a0f3d,#2e1a4e);
    border:1px solid #3a1e6f;border-left:3px solid #7b5ea7;
    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.2rem;}
.sync-badge{display:inline-block;background:rgba(16,185,129,.12);
    border:1px solid rgba(16,185,129,.35);color:#10b981;
    border-radius:20px;padding:.2rem .75rem;font-size:.76rem;font-weight:600;margin-right:.4rem;}
.warn-badge{display:inline-block;background:rgba(245,158,11,.12);
    border:1px solid rgba(245,158,11,.35);color:#f59e0b;
    border-radius:20px;padding:.2rem .75rem;font-size:.76rem;font-weight:600;}
.param-row{display:flex;justify-content:space-between;padding:.35rem 0;
    border-bottom:1px solid #1e3a5f;font-size:.86rem;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0", family="Inter,sans-serif"),
)

try:
    from utils.artifacts import (  # noqa: E402, I001
        load_train_data,
        load_metrics,
        feature_category,
        load_preprocessing_config,
        save_preprocessing_config,
    )
except Exception as _import_err:
    st.error(
        f"**Error loading `utils.artifacts`:**\n\n"
        f"```\n{type(_import_err).__name__}: {_import_err}\n```\n\n"
        "Check that `app/utils/artifacts.py` is present and correct."
    )
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL   = "market_forward_excess_returns"
EXCLUDE_COLS = {"date_id", TARGET_COL}
CAT_COLORS   = {
    "Forward Returns": "#00d4ff", "Price": "#f59e0b",   "Economic": "#10b981",
    "Momentum":        "#7b5ea7", "Volatility": "#ef4444", "Sentiment": "#06b6d4",
    "Engineered":      "#64748b", "Lagged Target": "#a3e635", "Interest": "#fb923c",
    "Target Stats":    "#e879f9", "Raw": "#94a3b8",
}

IMPUTATION_METHODS = {
    "Median":               "Column median — robust to outliers.",
    "Mean":                 "Column mean — fast, sensitive to outliers.",
    "Forward Fill":         "Last valid value forward — suited for time series.",
    "Backward Fill":        "Next valid value backward — time series.",
    "Linear Interpolation": "Linear interpolation between known values.",
    "KNN (5 neighbors)":    "KNN Imputer scikit-learn (5 neighbors) — accurate, slower.",
    "CatBoost":             "CatBoost Imputer — production method (slow).",
}
# Methods natively mapped to CatBoost in trainer; others → Median
_TRAINER_CATBOOST = {"CatBoost"}

# ── Load data & config ────────────────────────────────────────────────────────
df_full = load_train_data()
saved_cfg = load_preprocessing_config()   # last saved config (or defaults)

st.title("⚙️ Interactive Preprocessing")
st.markdown(
    "Data pipeline control panel — "
    "explore interactively, then **save** and **retrain** the models."
)

# ── Saved-config status banner ────────────────────────────────────────────────
if saved_cfg.get("saved_at"):
    st.markdown(f"""
    <div style='background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.25);
                border-radius:8px;padding:.55rem 1rem;margin-bottom:.5rem;font-size:.84rem;'>
        ✅ Last configuration saved on
        <strong style='color:#10b981;'>{saved_cfg["saved_at"]}</strong>
        &nbsp;—&nbsp;
        cutoff {saved_cfg["lower_pct"]:.1f}% → {saved_cfg["upper_pct"]:.1f}%,
        NaN threshold {saved_cfg["nan_threshold_pct"]}%,
        imputation &laquo; {saved_cfg["imputation_method"]} &raquo;,
        {saved_cfg["n_features"]} features.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if df_full is None:
    st.error("❌ Data unavailable — check your Kaggle credentials.")
    st.stop()

N_TOTAL = len(df_full)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data interval
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>1</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Training interval (cutoff)
        </span>
    </div>
    <div style='color:#64748b;font-size:.83rem;margin-top:.35rem;margin-left:2rem;'>
        Select any continuous portion of the dataset (0 → 100 %).
        The interval must be <strong style='color:#f59e0b;'>strictly positive</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

col_c1, col_c2, col_c3 = st.columns([2, 2, 3])
with col_c1:
    lower_pct = st.number_input(
        "Lower cutoff (%)", min_value=0.0, max_value=99.9,
        value=float(saved_cfg["lower_pct"]),
        step=0.5, format="%.1f",
        help=f"0% = row 0   |   total dataset = {N_TOTAL:,} rows",
    )
with col_c2:
    upper_pct = st.number_input(
        "Upper cutoff (%)", min_value=0.1, max_value=100.0,
        value=float(saved_cfg["upper_pct"]),
        step=0.5, format="%.1f",
        help="100% = last row",
    )

interval_size = upper_pct - lower_pct
if interval_size <= 0:
    with col_c3:
        st.error(f"❌ Invalid interval: upper cutoff ({upper_pct:.1f}%) "
                 f"≤ lower cutoff ({lower_pct:.1f}%).")
    st.stop()

lower_idx = int(N_TOTAL * lower_pct / 100)
upper_idx = max(int(N_TOTAL * upper_pct / 100), lower_idx + 1)
df_slice  = df_full.iloc[lower_idx:upper_idx].copy().reset_index(drop=True)
n_slice   = len(df_slice)

with col_c3:
    st.markdown(f"""
    <div style='background:rgba(0,212,255,.07);border:1px solid #1e3a5f;
                border-radius:8px;padding:.75rem 1rem;margin-top:1.6rem;'>
        <div style='color:#94a3b8;font-size:.76rem;text-transform:uppercase;'>Selection</div>
        <div style='color:#00d4ff;font-weight:800;font-size:1.2rem;'>{n_slice:,} rows</div>
        <div style='color:#64748b;font-size:.77rem;'>
            {lower_idx:,} → {upper_idx:,} &nbsp;|&nbsp;
            {lower_pct:.1f}% → {upper_pct:.1f}% &nbsp;|&nbsp;
            Δ {interval_size:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# Range visualisation bar
st.markdown(f"""
<div style='background:#1e293b;border-radius:6px;height:22px;position:relative;
            border:1px solid #1e3a5f;overflow:hidden;margin:.4rem 0 .2rem;'>
    <div style='position:absolute;left:{lower_pct:.1f}%;width:{interval_size:.1f}%;height:100%;
                background:linear-gradient(90deg,#00d4ff55,#00d4ff99);'></div>
    <div style='position:absolute;left:2px;top:50%;transform:translateY(-50%);
                color:#475569;font-size:.72rem;'>0%</div>
    <div style='position:absolute;right:2px;top:50%;transform:translateY(-50%);
                color:#475569;font-size:.72rem;'>100%</div>
</div>
<div style='display:flex;justify-content:space-between;color:#64748b;font-size:.73rem;'>
    <span>↑ {lower_pct:.1f}%</span><span>{upper_pct:.1f}% ↑</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Missing-value threshold
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>2</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Missing-value drop threshold
        </span>
    </div>
    <div style='color:#64748b;font-size:.83rem;margin-top:.35rem;margin-left:2rem;'>
        Columns whose NaN % exceeds the threshold → dropped.
        Table sorted descending with visual highlighting.
    </div>
</div>
""", unsafe_allow_html=True)

col_t1, col_t2 = st.columns([1, 3])
with col_t1:
    nan_threshold_pct = st.number_input(
        "Threshold (%)", min_value=0, max_value=100,
        value=int(saved_cfg["nan_threshold_pct"]),
        step=1,
        help="Default production value: 30%",
    )
with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.3);
                border-radius:8px;padding:.6rem 1rem;'>
        ⚠️ Any column with <strong style='color:#f59e0b;'>{'>'} {nan_threshold_pct}%</strong>
        NaN will be <strong style='color:#ef4444;'>dropped</strong>.
        Current production threshold: <strong>30%</strong>.
    </div>
    """, unsafe_allow_html=True)

nan_counts    = df_slice.isnull().sum()
nan_counts_nz = nan_counts[nan_counts > 0]
nan_pcts_nz   = (nan_counts_nz / n_slice * 100).round(2)

if len(nan_counts_nz) == 0:
    st.success("✅ No missing values in this interval.")
    df_clean = df_slice.copy()
    cols_dropped: list[str] = []
else:
    nan_table = pd.DataFrame({
        "Column":          nan_counts_nz.index.tolist(),
        "Missing values":  nan_counts_nz.values,
        "% Missing":       nan_pcts_nz.values,
        "Dropped":         ["✅ Yes" if p > nan_threshold_pct else "❌ No"
                            for p in nan_pcts_nz.values],
    }).sort_values("% Missing", ascending=False).reset_index(drop=True)

    n_drop = int((nan_pcts_nz > nan_threshold_pct).sum())
    n_keep = int((nan_pcts_nz <= nan_threshold_pct).sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total columns",      df_slice.shape[1])
    with k2: st.metric("Columns with NaN",   len(nan_counts_nz))
    with k3: st.metric("Dropped",            n_drop, delta=f"> {nan_threshold_pct}% NaN", delta_color="inverse")
    with k4: st.metric("Kept with NaN",      n_keep, delta="→ imputed")

    st.markdown("")
    st.markdown("<div class='section-title'>Table — descending sort</div>", unsafe_allow_html=True)

    def _color_nan_row(row):
        if row["Dropped"] == "✅ Yes":
            return ["background-color:rgba(239,68,68,.18);color:#ef4444"] * len(row)
        return ["background-color:rgba(245,158,11,.08);color:#e2e8f0"] * len(row)

    st.dataframe(
        nan_table.style.apply(_color_nan_row, axis=1)
                       .format({"% Missing": "{:.2f}%", "Missing values": "{:,}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown("""
    <div style='font-size:.76rem;color:#475569;'>
        🔴 Red = dropped (exceeds threshold) &nbsp;|&nbsp; 🟡 Orange = kept, will be imputed
    </div>""", unsafe_allow_html=True)

    # Bar chart
    fig_nan = go.Figure(go.Bar(
        x=nan_table["Column"],
        y=nan_table["% Missing"],
        marker=dict(
            color=["#ef4444" if p > nan_threshold_pct else "#f59e0b"
                   for p in nan_table["% Missing"]],
            line=dict(width=0.5, color="#0a0e1a"),
        ),
        text=[f"{p:.1f}%" for p in nan_table["% Missing"]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=9),
        hovertemplate="<b>%{x}</b><br>% Missing: %{y:.2f}%<extra></extra>",
    ))
    fig_nan.add_hline(y=nan_threshold_pct, line_dash="dash", line_color="#00d4ff", line_width=2,
                      annotation_text=f"  Threshold {nan_threshold_pct}%",
                      annotation_font_color="#00d4ff", annotation_position="top right")
    fig_nan.update_layout(**PLOTLY_DARK, height=300,
                          title=f"% Missing values per column (threshold = {nan_threshold_pct}%)",
                          xaxis_tickangle=-45, showlegend=False, margin=dict(b=110, t=50))
    st.plotly_chart(fig_nan, use_container_width=True)

    cols_dropped = nan_table[nan_table["Dropped"] == "✅ Yes"]["Column"].tolist()
    df_clean = df_slice.drop(columns=cols_dropped, errors="ignore")
    st.info(f"✅ Cleaned DataFrame: **{df_clean.shape[0]:,} × {df_clean.shape[1]}**"
            f" — {len(cols_dropped)} column(s) dropped.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Imputation method
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>3</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Imputation method
        </span>
    </div>
    <div style='color:#64748b;font-size:.83rem;margin-top:.35rem;margin-left:2rem;'>
        Applied dynamically to columns retained after step 2.
        <strong style='color:#7b5ea7;'>CatBoost</strong> and
        <strong style='color:#00d4ff;'>Median</strong> are natively supported
        by the training pipeline — other methods are mapped to
        <em>Median</em> at retraining.
    </div>
</div>
""", unsafe_allow_html=True)

imp_methods_list = list(IMPUTATION_METHODS.keys())
saved_imp        = saved_cfg["imputation_method"]
imp_idx          = imp_methods_list.index(saved_imp) if saved_imp in imp_methods_list else 0

col_imp1, col_imp2 = st.columns([1, 2])
with col_imp1:
    imputation_method = st.selectbox(
        "Imputation method", imp_methods_list, index=imp_idx,
        help="CatBoost = production method | Others = interactive exploration",
    )
with col_imp2:
    st.markdown("<br>", unsafe_allow_html=True)
    is_production = imputation_method in _TRAINER_CATBOOST or imputation_method == "Median"
    badge = (
        "<span class='sync-badge'>✅ Production-native</span>"
        if is_production
        else "<span class='warn-badge'>⚠️ → Median at retraining</span>"
    )
    st.markdown(f"""
    <div style='background:rgba(0,212,255,.05);border:1px solid #1e3a5f;
                border-radius:8px;padding:.6rem 1rem;'>
        {badge}
        <span style='color:#94a3b8;font-size:.84rem;margin-left:.4rem;'>
            {IMPUTATION_METHODS[imputation_method]}
        </span>
    </div>
    """, unsafe_allow_html=True)

feat_cols    = [c for c in df_clean.columns if c not in EXCLUDE_COLS]
nan_before   = int(df_clean[feat_cols].isnull().sum().sum())

if nan_before == 0:
    st.success("✅ No missing values — imputation not needed.")
    df_imputed = df_clean.copy()
else:
    if imputation_method == "CatBoost":
        st.warning("⚠️ CatBoost Imputer: may take several minutes.")
    with st.spinner(f"Imputing: {imputation_method}…"):
        df_imputed = df_clean.copy()
        if imputation_method == "Median":
            df_imputed[feat_cols] = df_imputed[feat_cols].fillna(df_imputed[feat_cols].median())
        elif imputation_method == "Mean":
            df_imputed[feat_cols] = df_imputed[feat_cols].fillna(df_imputed[feat_cols].mean())
        elif imputation_method == "Forward Fill":
            df_imputed[feat_cols] = df_imputed[feat_cols].ffill().fillna(df_imputed[feat_cols].median())
        elif imputation_method == "Backward Fill":
            df_imputed[feat_cols] = df_imputed[feat_cols].bfill().fillna(df_imputed[feat_cols].median())
        elif imputation_method == "Linear Interpolation":
            df_imputed[feat_cols] = (
                df_imputed[feat_cols]
                .interpolate(method="linear", axis=0, limit_direction="both")
                .fillna(df_imputed[feat_cols].median())
            )
        elif imputation_method == "KNN (5 neighbors)":
            from sklearn.impute import KNNImputer
            knn_imp  = KNNImputer(n_neighbors=5)
            n_fit    = min(2000, len(df_imputed))
            rng_idx  = np.random.default_rng(42).choice(len(df_imputed), n_fit, replace=False)
            knn_imp.fit(df_imputed.iloc[rng_idx][feat_cols])
            df_imputed[feat_cols] = knn_imp.transform(df_imputed[feat_cols])
        elif imputation_method == "CatBoost":
            try:
                from src.data.preprocessor import CatBoostImputer
                cb = CatBoostImputer(iterations=50, verbose=False)
                df_imputed[feat_cols] = cb.fit_transform(df_imputed[feat_cols])
            except Exception as exc:
                st.error(f"CatBoost failed ({exc}) → falling back to median.")
                df_imputed[feat_cols] = df_imputed[feat_cols].fillna(df_imputed[feat_cols].median())
        # Safety net
        still_nan = df_imputed[feat_cols].isnull().sum().sum()
        if still_nan > 0:
            fallback = df_imputed[feat_cols].median().fillna(0)
            df_imputed[feat_cols] = df_imputed[feat_cols].fillna(fallback)

    nan_after = int(df_imputed[feat_cols].isnull().sum().sum())
    ri1, ri2, ri3 = st.columns(3)
    with ri1: st.metric("NaN before", f"{nan_before:,}")
    with ri2: st.metric("NaN after", f"{nan_after:,}", delta=f"−{nan_before - nan_after:,}", delta_color="inverse")
    with ri3: st.metric("Method", imputation_method)

st.info(f"✅ DataFrame ready: **{df_imputed.shape[0]:,} × {df_imputed.shape[1]}** "
        f"({len(feat_cols)} features available).")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='step-card'>
    <div style='display:flex;align-items:center;'>
        <span class='step-num'>4</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.05rem;'>
            Feature Selection & Importance
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

avail_features = [c for c in feat_cols if c in df_imputed.columns]
n_avail        = len(avail_features)

if n_avail < 2:
    st.warning("Not enough features — widen the interval or lower the NaN threshold.")
    st.stop()

col_s1, col_s2 = st.columns([3, 2])
with col_s1:
    n_features_selected = st.slider(
        f"Number of features (1 → {n_avail})",
        min_value=1, max_value=n_avail,
        value=min(int(saved_cfg["n_features"]), n_avail),
        step=1,
    )
with col_s2:
    importance_method = st.selectbox(
        "Importance method",
        ["Random Forest", "Permutation Importance", "Ridge Regression (|coef|)"],
    )

# Modelling data
valid_mask = df_imputed[TARGET_COL].notna()
X_all      = df_imputed.loc[valid_mask, avail_features].fillna(0)
y_all      = df_imputed.loc[valid_mask, TARGET_COL].values

if len(X_all) < 30:
    st.warning(f"Only {len(X_all)} rows with valid target.")
    st.stop()

# Session cache
cache_key = (f"{importance_method}|{lower_pct:.1f}|{upper_pct:.1f}|"
             f"{nan_threshold_pct}|{imputation_method}|{n_avail}|{len(X_all)}")
if "imp_cache" not in st.session_state:
    st.session_state.imp_cache = {}

if cache_key not in st.session_state.imp_cache:
    with st.spinner(f"Computing importance ({importance_method})…"):
        try:
            n_s   = min(3000, len(X_all))
            rng   = np.random.default_rng(42)
            idx_s = rng.choice(len(X_all), n_s, replace=False)
            X_s, y_s = X_all.iloc[idx_s], y_all[idx_s]

            if importance_method == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                mdl = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)
                mdl.fit(X_s, y_s)
                imp_vals = mdl.feature_importances_

            elif importance_method == "Permutation Importance":
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.inspection import permutation_importance as _perm
                mdl = RandomForestRegressor(n_estimators=40, max_depth=5, random_state=42, n_jobs=-1)
                mdl.fit(X_s, y_s)
                perm = _perm(mdl, X_s, y_s, n_repeats=5, random_state=42, n_jobs=-1)
                imp_vals = np.maximum(perm.importances_mean, 0.0)

            else:  # Ridge
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                sc      = StandardScaler()
                Xs_all  = sc.fit_transform(X_all)
                mdl     = Ridge(alpha=1.0)
                mdl.fit(Xs_all, y_all)
                imp_vals = np.abs(mdl.coef_)

            imp_df_full = pd.DataFrame(
                {"feature": avail_features, "importance": imp_vals}
            ).sort_values("importance", ascending=False).reset_index(drop=True)
            st.session_state.imp_cache[cache_key] = imp_df_full

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

imp_df_full = st.session_state.imp_cache[cache_key]
top_n = imp_df_full.head(n_features_selected).copy()
top_n["Category"] = top_n["feature"].apply(feature_category)
top_n["color"]    = top_n["Category"].map(lambda c: CAT_COLORS.get(c, "#64748b"))
max_imp = top_n["importance"].max() or 1.0
top_n["imp_norm"] = top_n["importance"] / max_imp * 100

fig_imp = go.Figure(go.Bar(
    x=top_n["imp_norm"].values[::-1],
    y=top_n["feature"].values[::-1],
    orientation="h",
    marker=dict(color=top_n["color"].values[::-1], line=dict(width=0.5, color="#0a0e1a")),
    text=[f"{v:.1f}" for v in top_n["imp_norm"].values[::-1]],
    textposition="outside",
    textfont=dict(color="#94a3b8", size=9),
    customdata=np.stack([top_n["Category"].values[::-1], top_n["importance"].values[::-1]], axis=-1),
    hovertemplate="<b>%{y}</b><br>Category: %{customdata[0]}<br>Raw score: %{customdata[1]:.8f}<extra></extra>",
))
fig_imp.update_layout(
    **PLOTLY_DARK,
    height=max(350, n_features_selected * 24 + 80),
    title=f"Top {n_features_selected} features — {importance_method}",
    xaxis_title="Normalized score (max = 100)",
    yaxis=dict(tickfont=dict(size=10 if n_features_selected <= 30 else 8)),
    margin=dict(l=260, r=80, t=55, b=40), showlegend=False,
)
st.plotly_chart(fig_imp, use_container_width=True)

# KPI summary
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Selected features", n_features_selected, f"/{n_avail}")
with m2:
    tf = top_n.iloc[0]["feature"] if len(top_n) else "—"
    st.metric("Feature #1", (tf[:16] + "…") if len(tf) > 18 else tf)
with m3: st.metric("Raw score #1", f"{top_n.iloc[0]['importance']:.6f}" if len(top_n) else "—")
with m4: st.metric("Categories", int(top_n["Category"].nunique()))

# Export
with st.expander("📋 Detailed table + CSV export"):
    disp = top_n[["feature", "Category", "importance", "imp_norm"]].copy()
    disp.columns = ["Feature", "Category", "Raw score", "Normalized score"]
    disp.index = range(1, len(disp) + 1)
    st.dataframe(
        disp.style.format({"Raw score": "{:.8f}", "Normalized score": "{:.2f}"})
                  .background_gradient(subset=["Normalized score"], cmap="Blues"),
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download (CSV)", top_n[["feature", "Category", "importance", "imp_norm"]]
        .rename(columns={"feature": "Feature", "Category": "Category",
                         "importance": "Raw_score", "imp_norm": "Normalized_score"})
        .to_csv(index=False),
        "selected_features.csv", "text/csv",
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION PANEL — Save & Retrain
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='prod-card'>
    <div style='display:flex;align-items:center;'>
        <span style='font-size:1.4rem;margin-right:.6rem;'>🚀</span>
        <span style='font-weight:700;color:#e2e8f0;font-size:1.1rem;'>
            Production configuration & Retraining
        </span>
    </div>
    <div style='color:#64748b;font-size:.83rem;margin-top:.4rem;'>
        Save the current parameters to
        <code style='color:#7b5ea7;'>artifacts/preprocessing_config.json</code>
        then launch the full retraining.
        All dashboard pages will update automatically.
    </div>
</div>
""", unsafe_allow_html=True)

# Current config summary
_trainer_imp = "CatBoost" if imputation_method in _TRAINER_CATBOOST else "Median"
st.markdown("<div class='section-title'>Parameters that will be applied to the training pipeline</div>",
            unsafe_allow_html=True)

param_rows = [
    ("Data interval",          f"{lower_pct:.1f}% → {upper_pct:.1f}%  ({n_slice:,} rows)", "✅"),
    ("NaN threshold",          f"{nan_threshold_pct}%  ({len(cols_dropped) if isinstance(cols_dropped, list) else '?'} columns dropped)", "✅"),
    ("Imputation (production)", f"{_trainer_imp}" + ("" if imputation_method == _trainer_imp else f"  ← mapped from '{imputation_method}'"), "✅" if imputation_method in _TRAINER_CATBOOST or imputation_method == "Median" else "⚠️"),
    ("N features",             str(n_features_selected), "✅"),
    ("Feature selection",      "Combined (CatBoost 50% + Correlation 25% + MI 25%)", "✅"),
]

for label, val, icon in param_rows:
    color = "#10b981" if icon == "✅" else "#f59e0b"
    st.markdown(f"""
    <div class='param-row'>
        <span style='color:#94a3b8;'>{label}</span>
        <span style='color:{color};font-weight:600;'>{icon} {val}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model selection
st.markdown("<div class='section-title'>Models to train</div>", unsafe_allow_html=True)
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1: train_lgbm = st.checkbox("LightGBM",    value=True)
with col_m2: train_xgb  = st.checkbox("XGBoost",     value=True)
with col_m3: train_cat  = st.checkbox("CatBoost",    value=True)
with col_m4: train_rf   = st.checkbox("RandomForest",value=True)
with col_m5: train_ens  = st.checkbox("Ensemble",    value=True)

models_to_train = []
if train_lgbm: models_to_train.append("lightgbm")
if train_xgb:  models_to_train.append("xgboost")
if train_cat:  models_to_train.append("catboost")
if train_rf:   models_to_train.append("random_forest")

st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])

# ── Save button ───────────────────────────────────────────────────────────────
with btn_col1:
    if st.button("💾 Save configuration", type="secondary", use_container_width=True):
        cfg_to_save = {
            "lower_pct":         lower_pct,
            "upper_pct":         upper_pct,
            "nan_threshold_pct": nan_threshold_pct,
            "imputation_method": imputation_method,
            "n_features":        n_features_selected,
        }
        save_preprocessing_config(cfg_to_save)
        st.success(
            "✅ Configuration saved to `artifacts/preprocessing_config.json`. "
            "The next retraining run will use these parameters."
        )
        st.rerun()

# ── Retrain button ────────────────────────────────────────────────────────────
with btn_col2:
    retrain_clicked = st.button(
        "🚀 Launch retraining",
        type="primary",
        use_container_width=True,
        disabled=(len(models_to_train) == 0),
    )

with btn_col3:
    if len(models_to_train) == 0:
        st.warning("⚠️ Select at least one model.")
    else:
        st.markdown(f"""
        <div style='color:#64748b;font-size:.82rem;padding-top:.35rem;'>
            Models: {', '.join(m.title() for m in models_to_train)}
            {'+ Ensemble' if train_ens else ''}
            &nbsp;|&nbsp; Script: <code>scripts/retrain.py</code>
        </div>
        """, unsafe_allow_html=True)

# ── Retraining execution ──────────────────────────────────────────────────────
if retrain_clicked and models_to_train:
    # 1. Auto-save config before launching
    cfg_to_save = {
        "lower_pct":         lower_pct,
        "upper_pct":         upper_pct,
        "nan_threshold_pct": nan_threshold_pct,
        "imputation_method": imputation_method,
        "n_features":        n_features_selected,
    }
    save_preprocessing_config(cfg_to_save)
    st.info("💾 Configuration automatically saved before launch.")

    # 2. Build command
    cmd = [sys.executable, str(_ROOT / "scripts" / "retrain.py"),
           "--models"] + models_to_train
    if not train_ens:
        cmd.append("--no-ensemble")

    st.markdown("<div class='section-title'>Training log</div>", unsafe_allow_html=True)
    log_placeholder = st.empty()
    status_placeholder = st.empty()

    log_lines: list[str] = []

    # 3. Launch subprocess
    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_ROOT),
            env=env,
            bufsize=1,
        )
        status_placeholder.info("⏳ Training in progress…")

        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                log_lines.append(line)
            if len(log_lines) > 80:
                log_lines = log_lines[-80:]
            log_placeholder.code("\n".join(log_lines), language="")

        process.wait()

        if process.returncode == 0:
            status_placeholder.success("✅ Retraining completed successfully!")
            # Clear caches so all pages pick up the new results
            st.cache_data.clear()
        else:
            status_placeholder.error(
                f"❌ Retraining failed (return code: {process.returncode}). "
                "Check the log above."
            )

    except FileNotFoundError:
        st.error("❌ `scripts/retrain.py` not found. Check the project structure.")
    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")

    # 4. Show updated metrics if available
    if retrain_clicked:
        st.markdown("<div class='section-title'>Results after retraining</div>",
                    unsafe_allow_html=True)
        new_metrics = load_metrics()
        rows = []
        for model_name, vals in new_metrics.items():
            if vals.get("rmse") is not None:
                rows.append({
                    "Model":      model_name,
                    "RMSE":       f"{vals['rmse']:.6f}",
                    "MAE":        f"{vals['mae']:.6f}" if vals.get("mae") else "—",
                    "R²":         f"{vals['r2']:+.4f}" if vals.get("r2") is not None else "—",
                    "Dir. Acc.":  f"{vals['dir_acc']:.2f}%" if vals.get("dir_acc") else "—",
                    "Trained on": (vals.get("timestamp") or "")[:19] or "—",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("➡️ Navigate to the other pages to see the updated charts.")
        else:
            st.info("No results available — verify that the training completed successfully.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#334155;font-size:.78rem;padding:.4rem 0;'>
    ⚙️ Preprocessing Panel — Hull Tactical Market Prediction &nbsp;|&nbsp;
    Config → <code>artifacts/preprocessing_config.json</code> &nbsp;|&nbsp;
    Script → <code>scripts/retrain.py</code>
</div>
""", unsafe_allow_html=True)
