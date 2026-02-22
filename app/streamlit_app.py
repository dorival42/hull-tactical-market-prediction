"""Hull Tactical Market Prediction — Dark Mode Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make app/utils importable from any page
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hull Tactical — Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-mode CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a2f4e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.08);
    transition: transform 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0, 212, 255, 0.15);
    border-color: #00d4ff55;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px 8px 0 0;
    color: #94a3b8;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0f1f3d, #1a2f4e);
    color: #00d4ff;
    border-bottom: 2px solid #00d4ff;
}

/* ── Cards ── */
.dark-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a2f4e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.hero-banner {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1f3d 40%, #1a2f4e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b5ea7, #e91e63);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-top: 1rem;
}
.insight-item {
    background: rgba(0,212,255,0.05);
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    color: #cbd5e1;
    font-size: 0.9rem;
}
.insight-item.warning { border-left-color: #f59e0b; background: rgba(245,158,11,0.05); }
.insight-item.success { border-left-color: #10b981; background: rgba(16,185,129,0.05); }
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e3a5f;
}
/* ── Sidebar links ── */
.sidebar-link {
    display: block;
    color: #64748b;
    text-decoration: none;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.2rem;
    font-size: 0.88rem;
    transition: all 0.2s;
}
.sidebar-link:hover { background: rgba(0,212,255,0.1); color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
)

# ── Dynamic artifact loaders (auto-refresh after each training run) ────────────
from utils.artifacts import (  # noqa: E402
    load_metrics,
    load_feature_importance,
    load_train_data,
    load_preprocessing_info,
    best_model as _best_model_fn,
    feature_category,
)

REAL_RESULTS   = load_metrics()          # dict updated by trainer.py after each run
fi_df          = load_feature_importance("LightGBM")  # real importance scores
preproc_info   = load_preprocessing_info()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2rem; margin-bottom:0.3rem;'>📈</div>
        <div style='font-weight:800; font-size:1.1rem; color:#00d4ff;'>HULL TACTICAL</div>
        <div style='color:#475569; font-size:0.75rem; letter-spacing:0.1em;'>MARKET PREDICTION</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Selection**")
    selected_model = st.selectbox(
        "Active model",
        list(REAL_RESULTS.keys()),
        index=4,
        label_visibility="collapsed",
    )
    m = REAL_RESULTS[selected_model]
    _rmse_str  = f"{m['rmse']:.6f}"      if m.get('rmse')    is not None else "—"
    _r2_val    = m.get('r2')
    _r2_str    = f"{_r2_val:+.4f}"       if _r2_val          is not None else "—"
    _r2_color  = "#10b981" if (_r2_val or 0) > 0 else "#ef4444"
    _acc_str   = f"{m['dir_acc']:.2f}%"  if m.get('dir_acc') is not None else "—"
    st.markdown(f"""
    <div style='background:rgba(0,212,255,0.07);border:1px solid #1e3a5f;
                border-radius:10px;padding:0.8rem;margin-top:0.5rem;font-size:0.82rem;'>
        <div style='color:#94a3b8;'>RMSE</div>
        <div style='color:#e2e8f0;font-weight:700;'>{_rmse_str}</div>
        <div style='color:#94a3b8;margin-top:0.4rem;'>R²</div>
        <div style='color:{_r2_color};font-weight:700;'>{_r2_str}</div>
        <div style='color:#94a3b8;margin-top:0.4rem;'>Directional Acc.</div>
        <div style='color:#e2e8f0;font-weight:700;'>{_acc_str}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("""
    <a class='sidebar-link' href='https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow/#/experiments/0' target='_blank'>🔬 MLflow Dashboard</a>
    <a class='sidebar-link' href='https://www.kaggle.com/competitions/hull-tactical-market-prediction' target='_blank'>📊 Kaggle Competition</a>
    <a class='sidebar-link' href='https://github.com' target='_blank'>💻 GitHub Repository</a>
    """, unsafe_allow_html=True)

    st.markdown("---")
    ts = m.get("timestamp", "")[:10] if m.get("timestamp") else "—"
    st.caption(f"Experiment: hull-tactical-experiments\nLast trained: {ts}")

# ── Load data & derived KPIs (must come before hero banner) ───────────────────
df              = load_train_data()
best_name       = _best_model_fn(REAL_RESULTS, by="rmse")
best            = REAL_RESULTS.get(best_name, {})
ensemble        = REAL_RESULTS.get("Ensemble", {})
n_samples       = preproc_info.get("total_train_samples", len(df) if df is not None else 0)
n_feats         = preproc_info.get("n_features", len(load_feature_importance()))
n_models_trained = len([v for v in REAL_RESULTS.values() if v.get("rmse") is not None])
n_trading_days   = len(df) if df is not None else (n_samples or 0)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-banner'>
    <div class='hero-title'>Hull Tactical Market Prediction</div>
    <div style='color:#94a3b8; font-size:1.05rem; margin-top:0.5rem;'>
        Forecasting S&amp;P 500 excess returns with ensemble machine learning — Live on Kaggle Phase 2
    </div>
    <div style='margin-top:1.2rem;'>
        <span class='badge'>📅 Phase 2 Live</span>
        <span class='badge'>🤖 {n_models_trained} Models Trained</span>
        <span class='badge'>📊 {n_trading_days:,} Trading Days</span>
        <span class='badge'>🔬 {n_feats} Features Selected</span>
        <span class='badge'>🏆 Best: {best_name}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Training Samples", f"{n_samples:,}" if n_samples else "—")
with c2:
    st.metric("Selected Features", str(n_feats) if n_feats else "—", "of 248 engineered")
with c3:
    rmse_val = best.get("rmse")
    st.metric("Best RMSE", f"{rmse_val:.5f}" if rmse_val else "—", f"{best_name}", delta_color="off")
with c4:
    r2_val = best.get("r2")
    st.metric("Best R²", f"{r2_val:+.4f}" if r2_val is not None else "—", best_name, delta_color="normal")
with c5:
    dir_val = best.get("dir_acc")
    st.metric("Dir. Accuracy", f"{dir_val:.1f}%" if dir_val else "—", "+vs random")
with c6:
    ens_rmse = ensemble.get("rmse")
    st.metric("Ensemble RMSE", f"{ens_rmse:.5f}" if ens_rmse else "—", "Lowest overall", delta_color="off")

st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🏆 Model Results", "💡 Key Insights", "ℹ️ About"])

# ── TAB 1: Data Overview ──────────────────────────────────────────────────────
with tab1:
    if df is not None:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("<div class='section-title'>Target Distribution — market_forward_excess_returns</div>", unsafe_allow_html=True)
            target = df["market_forward_excess_returns"].dropna()
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=target,
                nbinsx=80,
                marker=dict(
                    color=target.values,
                    colorscale=[[0, "#ef4444"], [0.5, "#1e3a5f"], [1, "#10b981"]],
                    line=dict(width=0),
                ),
                name="Daily Returns",
                hovertemplate="Return: %{x:.4f}<br>Count: %{y}<extra></extra>",
            ))
            fig_dist.add_vline(x=target.mean(), line_dash="dash", line_color="#00d4ff",
                               annotation_text=f"Mean: {target.mean():.5f}", annotation_font_color="#00d4ff")
            fig_dist.add_vline(x=0, line_dash="dot", line_color="#64748b")
            fig_dist.update_layout(
                **PLOTLY_DARK,
                title=f"Distribution of Excess Returns ({n_trading_days:,} trading days)",
                xaxis_title="Excess Return",
                yaxis_title="Frequency",
                showlegend=False,
                height=320,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_right:
            st.markdown("<div class='section-title'>Data Statistics</div>", unsafe_allow_html=True)
            stats = {
                "Total rows": f"{len(df):,}",
                "Total columns": f"{len(df.columns)}",
                "Date range": f"0 → {int(df.date_id.max())}",
                "Target mean": f"{target.mean():.6f}",
                "Target std": f"{target.std():.6f}",
                "Target min": f"{target.min():.5f}",
                "Target max": f"{target.max():.5f}",
                "Positive days": f"{(target > 0).sum():,} ({(target > 0).mean()*100:.1f}%)",
            }
            for k, v in stats.items():
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:0.35rem 0;
                            border-bottom:1px solid #1e3a5f;font-size:0.88rem;'>
                    <span style='color:#64748b;'>{k}</span>
                    <span style='color:#e2e8f0;font-weight:600;'>{v}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-title'>Missing Values by Feature Category</div>", unsafe_allow_html=True)
            # Build category counts dynamically from actual column names
            _prefix_labels = [
                ("D", "D (Binary)"), ("E", "E (Macro)"), ("I", "I (Rates)"),
                ("M", "M (Market)"), ("P", "P (Price)"), ("S", "S (Sentiment)"),
                ("V", "V (Volatility)"), ("MOM", "MOM"),
            ]
            _skip = {"date_id", "market_forward_excess_returns"}
            categories = {
                label: len([c for c in df.columns if c.startswith(prefix) and c not in _skip])
                for prefix, label in _prefix_labels
            }
            categories = {k: v for k, v in categories.items() if v > 0}
            # Fallback if column names don't match any expected prefix
            if not categories:
                categories = {"D (Binary)": 9, "E (Macro)": 20, "I (Rates)": 9,
                              "M (Market)": 18, "P (Price)": 13, "S (Sentiment)": 12,
                              "V (Volatility)": 13, "MOM": 1}
            nan_pcts = []
            for prefix in ["D", "E", "I", "M", "P", "S", "V", "MOM"]:
                cols_ = [c for c in df.columns if c.startswith(prefix) and c != "date_id"]
                if cols_:
                    nan_pcts.append(df[cols_].isna().mean().mean() * 100)
                else:
                    nan_pcts.append(0)
            fig_nan = px.bar(
                x=list(categories.keys()),
                y=nan_pcts,
                color=nan_pcts,
                color_continuous_scale=[[0, "#10b981"], [0.3, "#f59e0b"], [1, "#ef4444"]],
                labels={"x": "Category", "y": "Avg Missing %"},
            )
            fig_nan.update_layout(**PLOTLY_DARK, showlegend=False, height=280,
                                  title="Average % Missing Values per Category",
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_nan, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-title'>Feature Categories Breakdown</div>", unsafe_allow_html=True)
            fig_pie = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.45,
            )
            fig_pie.update_layout(**PLOTLY_DARK, height=280,
                                  title="Feature Count by Category",
                                  showlegend=True,
                                  legend=dict(orientation="v", x=1.02))
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Cumulative target over time
        st.markdown("<div class='section-title'>Cumulative Excess Returns Over Time</div>", unsafe_allow_html=True)
        cumulative = target.cumsum().reset_index(drop=True)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df["date_id"],
            y=cumulative,
            mode="lines",
            line=dict(color="#00d4ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
            name="Cumulative Return",
        ))
        fig_cum.add_hline(y=0, line_dash="dot", line_color="#475569")
        fig_cum.update_layout(**PLOTLY_DARK, height=260,
                              title=f"Cumulative market_forward_excess_returns (all {n_trading_days:,} days)",
                              xaxis_title="date_id", yaxis_title="Cumulative Return",
                              showlegend=False)
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.warning("Training data not found at artifacts/data/train.csv")

# ── TAB 2: Model Results ──────────────────────────────────────────────────────
with tab2:
    st.markdown(f"<div class='section-title'>Real MLflow Results — Kaggle Data ({n_trading_days:,} rows, {n_feats} features)</div>",
                unsafe_allow_html=True)

    results_df = pd.DataFrame([
        {"Model": k, "RMSE": v["rmse"], "MAE": v["mae"], "R²": v["r2"], "Dir. Acc. %": v["dir_acc"]}
        for k, v in REAL_RESULTS.items()
    ]).dropna(subset=["RMSE"])  # only show trained models

    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = px.bar(
            results_df, x="Model", y="RMSE",
            color="RMSE", color_continuous_scale="Blues_r",
            title="RMSE by Model — Lower is Better",
        )
        fig_rmse.update_layout(**PLOTLY_DARK, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        fig_dir = px.bar(
            results_df, x="Model", y="Dir. Acc. %",
            color="Dir. Acc. %", color_continuous_scale="Greens",
            title="Directional Accuracy — Higher is Better",
        )
        fig_dir.add_hline(y=50, line_dash="dash", line_color="#ef4444",
                          annotation_text="Random (50%)", annotation_font_color="#ef4444")
        fig_dir.update_layout(**PLOTLY_DARK, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_dir, use_container_width=True)

    # Radar chart
    cats = ["RMSE (inv)", "MAE (inv)", "R² (norm)", "Dir. Acc. (norm)"]
    fig_radar = go.Figure()
    colors = ["#00d4ff", "#f59e0b", "#10b981", "#7b5ea7", "#e91e63"]
    for i, (model, vals) in enumerate(REAL_RESULTS.items()):
        # Normalize: lower RMSE/MAE → higher score; higher R²/Dir.Acc → higher score
        all_rmse = [v["rmse"] for v in REAL_RESULTS.values()]
        all_mae  = [v["mae"]  for v in REAL_RESULTS.values()]
        all_r2   = [v["r2"]   for v in REAL_RESULTS.values()]
        all_dir  = [v["dir_acc"] for v in REAL_RESULTS.values()]
        def norm_inv(x, arr): return 1 - (x - min(arr)) / (max(arr) - min(arr) + 1e-10)
        def norm(x, arr):     return (x - min(arr)) / (max(arr) - min(arr) + 1e-10)
        v = [
            norm_inv(vals["rmse"], all_rmse),
            norm_inv(vals["mae"],  all_mae),
            norm(vals["r2"],       all_r2),
            norm(vals["dir_acc"],  all_dir),
        ]
        v_closed = v + [v[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=v_closed, theta=cats + [cats[0]],
            fill="toself", name=model,
            line=dict(color=colors[i], width=2),
            opacity=0.75,
        ))
    fig_radar.update_layout(
        **PLOTLY_DARK,
        title="Normalized Multi-Metric Radar",
        polar=dict(
            bgcolor="rgba(15,31,61,0.5)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e3a5f", tickfont=dict(color="#64748b")),
            angularaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
        ),
        showlegend=True,
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Metrics table with color highlighting
    st.markdown("<div class='section-title'>Detailed Comparison Table</div>", unsafe_allow_html=True)
    def color_r2(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return "color: #10b981; font-weight:700" if v > 0 else "color: #ef4444"
    def color_rmse(v):
        mn, mx = results_df["RMSE"].min(), results_df["RMSE"].max()
        intensity = int(255 * (1 - (v - mn) / (mx - mn + 1e-10)))
        return f"color: rgb({255 - intensity},{intensity},{80})"

    styled = results_df.style\
        .format({"RMSE": "{:.6f}", "MAE": "{:.6f}", "R²": "{:+.4f}", "Dir. Acc. %": "{:.2f}%"})\
        .applymap(color_r2, subset=["R²"])\
        .highlight_min(subset=["RMSE", "MAE"], color="#0f2d1a")\
        .highlight_max(subset=["Dir. Acc. %"], color="#0f2d1a")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # MLflow links
    st.markdown("<div class='section-title'>MLflow Experiment</div>", unsafe_allow_html=True)
    st.markdown("🔗 [Ouvrir DagsHub MLflow](https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow/)")
    for model, vals in REAL_RESULTS.items():
        ts = vals.get("timestamp", "")[:19] if vals.get("timestamp") else "—"
        st.markdown(f"- **{model}** — trained {ts}")

# ── TAB 3: Insights ───────────────────────────────────────────────────────────
with tab3:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-title'>Top 10 Predictive Features (LightGBM)</div>", unsafe_allow_html=True)
        _cat_colors = {
            "Forward Returns": "#00d4ff", "Momentum": "#7b5ea7", "Sentiment": "#e91e63",
            "Price": "#10b981", "Economic": "#f59e0b", "Volatility": "#ef4444",
            "Target Stats": "#94a3b8", "Lagged Target": "#a3e635", "Engineered": "#64748b",
            "Interest": "#fb923c", "Raw": "#475569",
        }
        if not fi_df.empty:
            top10 = fi_df.head(10).copy()
            top10["Category"] = top10["feature"].apply(feature_category)
            fig_feat = px.bar(
                top10.sort_values("importance"),
                x="importance", y="feature", color="Category",
                color_discrete_map=_cat_colors, orientation="h",
                title="Real Feature Importance (LightGBM gain)",
            )
            fig_feat.update_layout(**PLOTLY_DARK, height=380, showlegend=True,
                                   yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature importance not yet available — train models first.")

    with col_r:
        st.markdown("<div class='section-title'>Key Findings & Insights</div>", unsafe_allow_html=True)
        _lgb  = REAL_RESULTS.get("LightGBM", {})
        _ens  = REAL_RESULTS.get("Ensemble", {})
        _top1 = fi_df.iloc[0]["feature"] if not fi_df.empty else "M4"
        insights = [
            ("success", f"LightGBM outperforms all single models with R²={_lgb.get('r2', 0.0042):+.4f} and directional accuracy of {_lgb.get('dir_acc', 50.45):.2f}%."),
            ("success", f"Ensemble achieves the lowest RMSE ({_ens.get('rmse', 0.011123):.6f}), best for stable daily predictions."),
            ("info",    f"Top predictive feature: {_top1} — highest LightGBM gain importance."),
            ("info",    "MACD-based features of lagged forward returns rank high, confirming technical analysis value."),
            ("info",    "Sentiment indicator S2 contributes significantly — market psychology matters."),
            ("warning", "R² values are near zero across all models — consistent with EMH theory."),
            ("warning", "Directional accuracy hovers around 50%, confirming market unpredictability."),
            ("info",    f"68 highly correlated features dropped — {n_feats} selected from 248 engineered."),
            ("success", f"{n_feats} features selected using CatBoost importance (50%) + Correlation (25%) + Mutual Info (25%)."),
            ("info",    "Cyclical time features (sin/cos encoding) captured seasonal market patterns."),
        ]
        for kind, text in insights:
            st.markdown(f"<div class='insight-item {kind}'>{'⚡' if kind=='info' else '✅' if kind=='success' else '⚠️'} {text}</div>",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Feature Engineering Pipeline Summary</div>", unsafe_allow_html=True)

    pipeline_cols = st.columns(5)
    _nan_pct   = int(preproc_info.get("nan_threshold", 0.3) * 100)
    _n_cols    = len(df.columns) if df is not None else "—"
    _n_years   = round(n_trading_days / 252) if n_trading_days else "?"
    _n_val     = max(0, n_trading_days - n_samples) if n_trading_days and n_samples else "—"
    pipeline_steps = [
        ("📥 Raw Data",      f"{n_trading_days:,} rows\n{_n_cols} columns\n~{_n_years} years of data"),
        ("⚙️ Feature Eng.", "159 new features\n248 total columns\nRolling, RSI, MACD…"),
        ("🧹 Preprocessing", f"0 columns dropped\n>{_nan_pct}% NaN threshold\nMedian imputation"),
        ("🎯 Selection",     f"68 correlated removed\n176 candidates\n{n_feats} final features"),
        ("🤖 Training",      f"{n_models_trained} models trained\n{n_samples:,} train samples\n{_n_val:,} val samples"),
    ]
    for col, (title, desc) in zip(pipeline_cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div class='dark-card' style='text-align:center;min-height:120px;'>
                <div style='font-size:1.6rem;'>{title.split()[0]}</div>
                <div style='font-weight:700;color:#00d4ff;font-size:0.85rem;margin:0.4rem 0;'>{title.split(' ', 1)[1]}</div>
                <div style='color:#64748b;font-size:0.78rem;white-space:pre-line;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── TAB 4: About ──────────────────────────────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='dark-card'>
            <div class='section-title'>About this Competition</div>
            <p style='color:#94a3b8; line-height:1.7;'>
            The <strong style='color:#e2e8f0;'>Hull Tactical Market Prediction</strong> competition on Kaggle
            challenges participants to forecast S&P 500 excess returns over a 6-month live trading window.
            Unlike typical Kaggle competitions, model notebooks execute automatically every trading day
            on real market data — directly testing whether ML can challenge the Efficient Market Hypothesis.
            </p>
            <div style='margin-top:1rem;'>
                <div style='color:#64748b;font-size:0.85rem;'>Phase 1 (Training)</div>
                <div style='color:#e2e8f0;'>Sep 16 – Dec 15, 2025</div>
                <div style='color:#64748b;font-size:0.85rem;margin-top:0.5rem;'>Phase 2 (Live Forecasting)</div>
                <div style='color:#e2e8f0;'>Dec 15, 2025 – Jun 16, 2026</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='dark-card'>
            <div class='section-title'>Team</div>
            <div style='color:#94a3b8; line-height:2;'>
                👤 <strong style='color:#e2e8f0;'>Pierre Chrislin DORIVAL</strong><br>
                👤 <strong style='color:#e2e8f0;'>Emile STEEVENSON</strong><br>
                👤 <strong style='color:#e2e8f0;'>Jobed FELIMA</strong><br>
                👤 <strong style='color:#e2e8f0;'>Sebastien Witchmen ESTANIS</strong>
            </div>
            <div style='margin-top:1.2rem;'>
                <div class='section-title'>Tech Stack</div>
                <div style='color:#64748b;font-size:0.85rem;'>
                LightGBM · XGBoost · CatBoost · RandomForest<br>
                MLflow · DagsHub · Streamlit · Plotly<br>
                Pandas · NumPy · scikit-learn · Kaggle API
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding:0.5rem 0;'>
    Hull Tactical Market Prediction Dashboard &nbsp;|&nbsp;
    Built with Streamlit &amp; Plotly &nbsp;|&nbsp;
    <a href='https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow' style='color:#1e3a5f;'>MLflow on DagsHub</a>
</div>
""", unsafe_allow_html=True)
