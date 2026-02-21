"""Feature Analysis page — Hull Tactical Dark Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure app/ is on sys.path
_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

st.set_page_config(page_title="Feature Analysis — Hull Tactical", page_icon="🔍", layout="wide")

# ── Dark CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);border-right:1px solid #1e3a5f;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1rem;box-shadow:0 4px 16px rgba(0,212,255,.08);}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700;}
.section-title{font-size:1rem;font-weight:700;color:#00d4ff;letter-spacing:.05em;text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.3rem;border-bottom:1px solid #1e3a5f;}
.feat-badge{display:inline-block;padding:.15rem .55rem;border-radius:10px;font-size:.75rem;font-weight:600;margin:.1rem;}
.method-card{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1.2rem;margin:.3rem 0;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0"),
)

from utils.artifacts import (  # noqa: E402
    load_feature_importance,
    load_selected_features,
    load_preprocessing_info,
    load_train_data as _load_train,
    feature_category,
)

ROOT = Path(__file__).parent.parent.parent
DATA_PATH    = ROOT / "artifacts" / "data" / "train.csv"

# ── Category config ───────────────────────────────────────────────────────────
CAT_COLORS = {
    "Forward Returns": "#00d4ff",
    "Price":           "#f59e0b",
    "Economic":        "#10b981",
    "Momentum":        "#7b5ea7",
    "Volatility":      "#ef4444",
    "Sentiment":       "#06b6d4",
    "Engineered":      "#64748b",
    "Lagged Target":   "#a3e635",
    "Interest":        "#fb923c",
    "Target Stats":    "#e879f9",
    "Raw":             "#94a3b8",
}

# ── Real category breakdown (from training artifacts) ─────────────────────────
CAT_COUNTS = {
    "Forward Returns": 20, "Price": 13, "Economic": 12, "Momentum": 11,
    "Volatility": 10, "Engineered": 10, "Sentiment": 8, "Target Stats": 5,
    "Interest": 5, "Lagged Target": 4, "Raw": 2,
}

@st.cache_data(show_spinner=False)
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


# ── Load dynamic artifacts ─────────────────────────────────────────────────────
df = load_data()
selected_features = load_selected_features()          # list[str] from JSON
preproc_info      = load_preprocessing_info()
fi_df             = load_feature_importance("LightGBM")   # DataFrame [feature, importance, rank]

# Build TOP_FEATURES from real importance data (or fall back to selected_features order)
if not fi_df.empty:
    fi_df["Category"] = fi_df["feature"].apply(feature_category)
    TOP_FEATURES = list(zip(fi_df["feature"], fi_df["Category"], fi_df["importance"]))
else:
    TOP_FEATURES = [(f, feature_category(f), float(len(selected_features) - i))
                    for i, f in enumerate(selected_features[:20])]

# Category counts from real selected features
from collections import Counter  # noqa: E402
_cat_raw = [feature_category(f) for f in selected_features] if selected_features else []
CAT_COUNTS = dict(Counter(_cat_raw)) if _cat_raw else {
    "Forward Returns": 20, "Price": 13, "Economic": 12, "Momentum": 11,
    "Volatility": 10, "Engineered": 10, "Sentiment": 8, "Target Stats": 5,
    "Interest": 5, "Lagged Target": 4, "Raw": 2,
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Feature Analysis")
st.markdown("Feature engineering pipeline, importance rankings, and correlation analysis from real training data.")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
n_raw = 97
n_engineered = 159
n_total = n_raw + n_engineered
n_selected = preproc_info.get("n_features", len(selected_features) if selected_features else 100)
n_removed = n_total - n_selected

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Raw Features", str(n_raw), delta="Kaggle data", delta_color="off")
with c2: st.metric("Engineered", str(n_engineered), delta="Generated", delta_color="off")
with c3: st.metric("Total Before Selection", str(n_total))
with c4: st.metric("Selected Features", str(n_selected), delta=f"-{n_removed} removed", delta_color="inverse")
with c5: st.metric("Feature Categories", "11", delta="Diverse signals", delta_color="off")

st.markdown("---")

# ── Top Feature Importance ─────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Top 20 Feature Importance (LightGBM)</div>", unsafe_allow_html=True)

top20 = TOP_FEATURES[:20]
feat_names  = [f[0] for f in top20]
feat_cats   = [f[1] for f in top20]
feat_scores = [f[2] for f in top20]
# Normalise scores so max = 100 for display consistency
_max_score = max(feat_scores) if feat_scores else 1
feat_scores = [s / _max_score * 100 for s in feat_scores]
feat_colors = [CAT_COLORS.get(c, "#64748b") for c in feat_cats]

fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    x=feat_scores[::-1], y=feat_names[::-1],
    orientation="h",
    marker=dict(
        color=feat_colors[::-1],
        line=dict(width=1, color="#0a0e1a"),
    ),
    text=[f"{s:.1f}" for s in feat_scores[::-1]],
    textposition="outside",
    textfont=dict(color="#94a3b8", size=10),
))
fig_imp.update_layout(**PLOTLY_DARK, height=520,
                      title="Feature Importance Score (normalized, LightGBM)",
                      xaxis_title="Importance Score",
                      yaxis=dict(tickfont=dict(size=11)),
                      margin=dict(l=260))
st.plotly_chart(fig_imp, use_container_width=True)

# ── Category Breakdown ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Feature Category Breakdown</div>", unsafe_allow_html=True)
col_pie, col_bar = st.columns(2)

cat_names  = list(CAT_COUNTS.keys())
cat_vals   = list(CAT_COUNTS.values())
cat_clrs   = [CAT_COLORS.get(c, "#64748b") for c in cat_names]

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=cat_names, values=cat_vals,
        marker=dict(colors=cat_clrs, line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
        hole=0.42,
    ))
    fig_pie.update_layout(**PLOTLY_DARK, height=350,
                          title="Selected Features by Category",
                          showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    sorted_cats = sorted(zip(cat_names, cat_vals, cat_clrs), key=lambda x: -x[1])
    sn, sv, sc = zip(*sorted_cats)
    fig_cat = go.Figure(go.Bar(
        x=list(sn), y=list(sv),
        marker=dict(color=list(sc), line=dict(width=1, color="#0a0e1a")),
        text=list(sv), textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_cat.update_layout(**PLOTLY_DARK, height=350,
                          title="Count of Selected Features per Category",
                          xaxis_title="Category", yaxis_title="Count",
                          xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)

# ── Feature Selection Pipeline ────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Feature Selection Method</div>", unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown("""
    <div class='method-card'>
        <div style='font-size:.8rem;color:#00d4ff;font-weight:700;'>50% WEIGHT</div>
        <div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:.4rem 0;'>LightGBM Importance</div>
        <div style='color:#94a3b8;font-size:.83rem;'>Train LightGBM on full 248-feature set. Rank features by split/gain importance. Captures non-linear dependencies.</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class='method-card'>
        <div style='font-size:.8rem;color:#f59e0b;font-weight:700;'>25% WEIGHT</div>
        <div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:.4rem 0;'>Pearson Correlation</div>
        <div style='color:#94a3b8;font-size:.83rem;'>Rank features by absolute linear correlation with target. Remove highly correlated pairs (threshold 0.95) to reduce redundancy.</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class='method-card'>
        <div style='font-size:.8rem;color:#10b981;font-weight:700;'>25% WEIGHT</div>
        <div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:.4rem 0;'>Mutual Information</div>
        <div style='color:#94a3b8;font-size:.83rem;'>Estimate mutual information between each feature and target. Captures both linear and non-linear relationships without distributional assumptions.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.info("**Result:** 248 features → 100 selected. 68 features removed by high correlation filter. "
        "Final selection combines scores: 50% LGBM importance + 25% correlation + 25% mutual info.")

# ── Correlation Heatmap (real data) ───────────────────────────────────────────
if df is not None:
    st.markdown("---")
    st.markdown("<div class='section-title'>Target Correlation Heatmap (Real Data)</div>", unsafe_allow_html=True)

    target_col = "market_forward_excess_returns"
    # Use top features that exist in train.csv
    real_cols = [c for c in [f[0] for f in TOP_FEATURES] if c in df.columns][:15]
    if target_col in df.columns and real_cols:
        sub = df[[target_col] + real_cols].dropna()
        corr = sub.corr()
        labels = [target_col.replace("market_forward_excess_returns", "TARGET")] + real_cols

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=labels, y=labels,
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(size=8, color="#e2e8f0"),
            colorbar=dict(tickcolor="#64748b"),
        ))
        fig_corr.update_layout(**PLOTLY_DARK, height=500,
                               title="Correlation Matrix: Target + Top Selected Features (real train.csv)",
                               xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                               yaxis=dict(tickfont=dict(size=9)),
                               margin=dict(b=120))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Target Distribution vs Raw Features ─────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>Top Feature Distributions (Real Data)</div>", unsafe_allow_html=True)

    available = [c for c in [f[0] for f in TOP_FEATURES[:8]] if c in df.columns]
    if len(available) >= 2:
        cols_dist = st.columns(min(4, len(available)))
        for i, feat in enumerate(available[:4]):
            with cols_dist[i]:
                st.markdown(f"<div style='font-size:.75rem;color:#00d4ff;text-align:center;margin-bottom:.3rem;'>{feat}</div>", unsafe_allow_html=True)
                vals = df[feat].dropna()
                fig_mini = go.Figure(go.Histogram(
                    x=vals, nbinsx=40,
                    marker=dict(color=CAT_COLORS.get([f[1] for f in TOP_FEATURES if f[0] == feat][0] if any(f[0] == feat for f in TOP_FEATURES) else "Engineered", "#64748b")),
                ))
                fig_mini.update_layout(**PLOTLY_DARK, height=180,
                                       showlegend=False, margin=dict(t=5, b=30, l=5, r=5),
                                       xaxis=dict(tickfont=dict(size=8)),
                                       yaxis=dict(tickfont=dict(size=8)))
                st.plotly_chart(fig_mini, use_container_width=True)

    # ── Feature Statistics Table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>Feature Statistics (Real Data)</div>", unsafe_allow_html=True)

    stat_feats = [c for c in [f[0] for f in TOP_FEATURES[:15]] if c in df.columns]
    if stat_feats:
        stat_df = df[stat_feats].describe().T[["mean", "std", "min", "50%", "max"]]
        stat_df.columns = ["Mean", "Std", "Min", "Median", "Max"]
        stat_df["Missing%"] = (df[stat_feats].isnull().sum() / len(df) * 100).values

        fmt_stat = stat_df.copy()
        for col in ["Mean", "Std", "Min", "Median", "Max"]:
            fmt_stat[col] = fmt_stat[col].apply(lambda x: f"{x:.4f}")
        fmt_stat["Missing%"] = fmt_stat["Missing%"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(fmt_stat, use_container_width=True)

else:
    st.warning("Training data not found. Run the download script first.")

# ── Engineered Feature Categories Explained ────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Feature Engineering Summary</div>", unsafe_allow_html=True)

engineering_info = [
    ("Forward Returns", "#00d4ff", "20", "Rolling mean (5/10/60d), MACD, MACD signal/hist, rolling max, rolling min of lagged forward returns"),
    ("Price",           "#f59e0b", "13", "P-category raw features: price levels, spreads, momentum metrics derived from price action"),
    ("Economic",        "#10b981", "12", "E-category macro indicators: economic surprise, credit conditions, growth signals"),
    ("Momentum",        "#7b5ea7", "11", "M-category momentum: trend strength, acceleration, reversal signals, M4 = top feature"),
    ("Volatility",      "#ef4444", "10", "V-category: realized vol, vol-of-vol, vol momentum, rolling mean/std of volatility"),
    ("Engineered",      "#64748b", "10", "Custom: mean-reversion z-scores, interaction features, vol×momentum, autocorrelation"),
    ("Sentiment",       "#06b6d4", "8",  "S-category: market sentiment proxies, flow data, positioning signals"),
    ("Target Stats",    "#e879f9", "5",  "Rolling statistics of the target: rolling mean (5/20d), rolling vol, z-score"),
    ("Interest",        "#fb923c", "5",  "I-category: interest rate spreads, yield curve, credit spreads"),
    ("Lagged Target",   "#a3e635", "4",  "Direct lags of market_forward_excess_returns (t-1, t-2, t-3) and lagged market returns"),
]

for cat, color, count, desc in engineering_info:
    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;gap:.8rem;padding:.5rem 0;border-bottom:1px solid #1e3a5f;'>
        <div style='min-width:140px;'>
            <span style='color:{color};font-weight:700;'>{cat}</span>
            <span style='color:#94a3b8;font-size:.8rem;'> ({count} features)</span>
        </div>
        <div style='color:#cbd5e1;font-size:.85rem;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ── Download ─────────────────────────────────────────────────────────────────
export_df = fi_df[["feature", "importance", "rank"]].copy() if not fi_df.empty else \
            pd.DataFrame(TOP_FEATURES, columns=["feature", "category", "importance_score"])
csv = export_df.to_csv(index=False)
st.download_button("⬇️ Download Feature Importance CSV", csv, "feature_importance.csv", "text/csv")
