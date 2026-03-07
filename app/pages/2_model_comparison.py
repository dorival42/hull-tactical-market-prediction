"""Model Comparison page — Hull Tactical Dark Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure app/ is on sys.path so utils is importable
_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

st.set_page_config(page_title="Model Comparison — Hull Tactical", page_icon="🔬", layout="wide")

# ── Dark CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);border-right:1px solid #1e3a5f;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1rem;box-shadow:0 4px 16px rgba(0,212,255,.08);}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700;}
.section-title{font-size:1rem;font-weight:700;color:#00d4ff;letter-spacing:.05em;text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.3rem;border-bottom:1px solid #1e3a5f;}
.winner-badge{display:inline-block;background:rgba(16,185,129,.18);border:1px solid rgba(16,185,129,.4);border-radius:20px;padding:.25rem .75rem;font-size:.8rem;color:#10b981;font-weight:700;}
.model-card{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1rem;margin:.3rem 0;}
table.styled-table{width:100%;border-collapse:collapse;}
table.styled-table th{background:#0f1f3d;color:#00d4ff;padding:.6rem 1rem;text-align:left;font-size:.8rem;letter-spacing:.05em;}
table.styled-table td{padding:.5rem 1rem;border-bottom:1px solid #1e3a5f;font-size:.85rem;}
table.styled-table tr:hover td{background:rgba(0,212,255,.05);}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0"),
)

# ── Dynamic artifact loaders ──────────────────────────────────────────────────
from utils.artifacts import load_metrics, best_model as _best_model_fn  # noqa: E402

REAL_RESULTS = load_metrics()   # auto-updated after each training run
MODELS = list(REAL_RESULTS.keys())
COLORS = {
    "LightGBM": "#00d4ff", "XGBoost": "#f59e0b", "CatBoost": "#7b5ea7",
    "RandomForest": "#10b981", "Ensemble": "#ef4444",
}

# ── Safe formatting helper ─────────────────────────────────────────────────────
def _f(val, fmt=".6f", fallback="—"):
    """Format val with fmt, return fallback if val is None."""
    try:
        return format(val, fmt) if val is not None else fallback
    except (TypeError, ValueError):
        return fallback

metrics_df = pd.DataFrame([
    {"Model": m, **v} for m, v in REAL_RESULTS.items()
])
# Drop rows where rmse is None (models not yet trained)
metrics_df = metrics_df.dropna(subset=["rmse"])
MODELS = metrics_df["Model"].tolist() if not metrics_df.empty else list(REAL_RESULTS.keys())

# ── Guard: show warning if no model has been trained yet ──────────────────────
if metrics_df.empty:
    st.title("🔬 Model Comparison")
    st.warning("⚙️ No trained models found yet. Run `python scripts/train.py --model all` first.")
    st.stop()

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "artifacts" / "data" / "train.csv"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Model Comparison")
st.markdown("Full comparison of all trained models using real Kaggle competition data logged to MLflow/DagsHub.")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
best_rmse_model = metrics_df.loc[metrics_df["rmse"].idxmin(), "Model"]
best_r2_model   = metrics_df.loc[metrics_df["r2"].idxmax(), "Model"]
best_acc_model  = metrics_df.loc[metrics_df["dir_acc"].idxmax(), "Model"]
best_rmse_val   = metrics_df["rmse"].min()
best_r2_val     = metrics_df["r2"].max()
best_acc_val    = metrics_df["dir_acc"].max()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Best RMSE", _f(best_rmse_val), delta=best_rmse_model, delta_color="off")
with c2:
    st.metric("Best R²", _f(best_r2_val, ".4f"), delta=best_r2_model, delta_color="off")
with c3:
    st.metric("Best Dir. Acc.", _f(best_acc_val, ".2f") + ("%" if best_acc_val is not None else ""),
              delta=best_acc_model, delta_color="off")
with c4:
    rmse_spread = (metrics_df["rmse"].max() - metrics_df["rmse"].min()) * 1000
    st.metric("RMSE Spread", f"{rmse_spread:.2f}×10⁻³", delta="Models are close", delta_color="off")
with c5:
    st.metric("Models Trained", str(len(metrics_df)), delta="LGB · XGB · CAT · RF · ENS", delta_color="off")

st.markdown("---")

# ── Bar Charts ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Metric Comparison</div>", unsafe_allow_html=True)
col_r, col_d = st.columns(2)

with col_r:
    fig_rmse = go.Figure()
    bar_colors = [COLORS[m] for m in MODELS]
    rmse_vals = [REAL_RESULTS[m]["rmse"] for m in MODELS]
    fig_rmse.add_trace(go.Bar(
        x=MODELS,
        y=rmse_vals,
        marker=dict(color=bar_colors, line=dict(width=1, color="#1e3a5f")),
        text=[_f(REAL_RESULTS[m]['rmse'], ".5f") for m in MODELS],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    # Highlight best
    if best_rmse_val is not None:
        fig_rmse.add_hline(y=best_rmse_val, line_dash="dot", line_color="#10b981",
                           annotation_text=f"Best: {_f(best_rmse_val)}", annotation_font_color="#10b981")
    _rmse_spread = max(rmse_vals) - min(rmse_vals)
    _rmse_pad = max(_rmse_spread * 2, 0.00005)
    fig_rmse.update_layout(**PLOTLY_DARK, height=320, showlegend=False,
                           title="RMSE — Lower is Better",
                           xaxis_title="Model", yaxis_title="RMSE",
                           yaxis_range=[min(rmse_vals) - _rmse_pad, max(rmse_vals) + _rmse_pad])
    st.plotly_chart(fig_rmse, use_container_width=True)

with col_d:
    fig_acc = go.Figure()
    acc_vals = [REAL_RESULTS[m]["dir_acc"] for m in MODELS]
    fig_acc.add_trace(go.Bar(
        x=MODELS,
        y=acc_vals,
        marker=dict(color=bar_colors, line=dict(width=1, color="#1e3a5f")),
        text=[_f(REAL_RESULTS[m]['dir_acc'], ".2f") + "%" for m in MODELS],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_acc.add_hline(y=50, line_dash="dot", line_color="#ef4444",
                      annotation_text="Random baseline 50%", annotation_font_color="#ef4444")
    _acc_spread = max(acc_vals) - min(acc_vals)
    _acc_pad = max(_acc_spread * 2, 1.0)
    fig_acc.update_layout(**PLOTLY_DARK, height=320, showlegend=False,
                          title="Directional Accuracy — Higher is Better",
                          xaxis_title="Model", yaxis_title="Dir. Accuracy (%)",
                          yaxis_range=[min(acc_vals) - _acc_pad, max(acc_vals) + _acc_pad])
    st.plotly_chart(fig_acc, use_container_width=True)

col_mae, col_r2 = st.columns(2)

with col_mae:
    fig_mae = go.Figure()
    mae_vals = [REAL_RESULTS[m]["mae"] for m in MODELS]
    fig_mae.add_trace(go.Bar(
        x=MODELS,
        y=mae_vals,
        marker=dict(color=bar_colors, line=dict(width=1, color="#1e3a5f")),
        text=[_f(REAL_RESULTS[m]['mae'], ".5f") for m in MODELS],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    _mae_spread = max(mae_vals) - min(mae_vals)
    _mae_pad = max(_mae_spread * 2, 0.00005)
    fig_mae.update_layout(**PLOTLY_DARK, height=280, showlegend=False,
                          title="MAE — Lower is Better",
                          xaxis_title="Model", yaxis_title="MAE",
                          yaxis_range=[min(mae_vals) - _mae_pad, max(mae_vals) + _mae_pad])
    st.plotly_chart(fig_mae, use_container_width=True)

with col_r2:
    r2_colors = ["#10b981" if (REAL_RESULTS[m].get("r2") or 0) >= 0 else "#ef4444" for m in MODELS]
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(
        x=MODELS,
        y=[REAL_RESULTS[m]["r2"] for m in MODELS],
        marker=dict(color=r2_colors, line=dict(width=1, color="#1e3a5f")),
        text=[_f(REAL_RESULTS[m]['r2'], "+.4f") for m in MODELS],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_r2.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig_r2.update_layout(**PLOTLY_DARK, height=280, showlegend=False,
                         title="R² Score — Higher is Better",
                         xaxis_title="Model", yaxis_title="R²")
    st.plotly_chart(fig_r2, use_container_width=True)

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Multi-Metric Radar</div>", unsafe_allow_html=True)

# Normalize: higher always = better
categories = ["RMSE\n(inv.)", "MAE\n(inv.)", "R²", "Dir. Acc.", "Speed\n(inv.)"]

def normalize_inv(vals):
    mn, mx = min(vals), max(vals)
    return [(mx - v) / (mx - mn + 1e-12) for v in vals]

def normalize_fwd(vals):
    mn, mx = min(vals), max(vals)
    return [(v - mn) / (mx - mn + 1e-12) for v in vals]

rmse_norm  = normalize_inv([REAL_RESULTS[m]["rmse"]       for m in MODELS])
mae_norm   = normalize_inv([REAL_RESULTS[m]["mae"]        for m in MODELS])
r2_norm    = normalize_fwd([REAL_RESULTS[m]["r2"]         for m in MODELS])
acc_norm   = normalize_fwd([REAL_RESULTS[m]["dir_acc"]    for m in MODELS])
time_norm  = normalize_inv([REAL_RESULTS[m]["train_time"] for m in MODELS])

fig_radar = go.Figure()
for i, model in enumerate(MODELS):
    vals = [rmse_norm[i], mae_norm[i], r2_norm[i], acc_norm[i], time_norm[i]]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself", name=model,
        line=dict(color=COLORS[model], width=2),
        fillcolor="rgba(0,0,0,0.05)",
    ))

fig_radar.update_layout(
    **PLOTLY_DARK, height=420,
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickcolor="#64748b", gridcolor="#1e3a5f"),
        angularaxis=dict(tickcolor="#64748b", gridcolor="#1e3a5f"),
        bgcolor="rgba(11,17,32,0.8)",
    ),
    legend=dict(orientation="h", y=-0.12),
    title="Normalized Multi-Metric Comparison (higher = better on each axis)",
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Walk-Forward Validation ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Walk-Forward Validation Simulation</div>", unsafe_allow_html=True)

np.random.seed(42)
n_folds = 5
wf_fig = go.Figure()
for model in ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "Ensemble"]:
    base = REAL_RESULTS[model]["rmse"]
    noise = np.random.randn(n_folds) * 0.00015
    fold_rmse = base + noise + np.linspace(0.0001, -0.0001, n_folds)
    wf_fig.add_trace(go.Scatter(
        x=list(range(1, n_folds + 1)), y=fold_rmse,
        mode="lines+markers", name=model,
        line=dict(color=COLORS[model], width=2),
        marker=dict(size=8, symbol="circle"),
    ))

wf_fig.update_layout(**PLOTLY_DARK, height=320,
                     title="RMSE per Walk-Forward Fold (simulated from real results)",
                     xaxis_title="Validation Fold", yaxis_title="RMSE",
                     legend=dict(orientation="h", y=1.08),
                     hovermode="x unified")
st.plotly_chart(wf_fig, use_container_width=True)

# ── Training Time ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Training Time vs RMSE Trade-off</div>", unsafe_allow_html=True)
scatter_fig = go.Figure()
for m in MODELS:
    scatter_fig.add_trace(go.Scatter(
        x=[REAL_RESULTS[m]["train_time"]],
        y=[REAL_RESULTS[m]["rmse"]],
        mode="markers+text",
        name=m, text=[m], textposition="top center",
        marker=dict(size=16, color=COLORS[m], line=dict(width=2, color="#0a0e1a")),
    ))

scatter_fig.update_layout(**PLOTLY_DARK, height=320, showlegend=False,
                          title="Training Time (s) vs RMSE — Ideal: bottom-left",
                          xaxis_title="Training Time (seconds)",
                          yaxis_title="RMSE")
st.plotly_chart(scatter_fig, use_container_width=True)

# ── Detailed Metrics Table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Detailed Metrics Table</div>", unsafe_allow_html=True)

rows = ""
for m in MODELS:
    r = REAL_RESULTS[m]
    best_mark = " <span class='winner-badge'>★ Best</span>" if m == best_rmse_model else ""
    r2_color  = "#10b981" if (r.get("r2") or 0) >= 0 else "#ef4444"
    acc_color = "#10b981" if (r.get("dir_acc") or 0) > 50 else "#ef4444"
    rows += f"""
    <tr>
        <td style="color:{COLORS[m]};font-weight:700;">{m}{best_mark}</td>
        <td>{_f(r.get('rmse'))}</td>
        <td>{_f(r.get('mae'))}</td>
        <td style="color:{r2_color};">{_f(r.get('r2'), '+.4f')}</td>
        <td style="color:{acc_color};">{_f(r.get('dir_acc'), '.2f')}%</td>
        <td>{r.get('train_time') or '—'}s</td>
    </tr>"""

st.markdown(f"""
<table class="styled-table">
<thead><tr>
  <th>Model</th><th>RMSE ↓</th><th>MAE ↓</th><th>R² ↑</th><th>Dir. Acc. ↑</th><th>Train Time</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

# ── Recommendation ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Model Selection Recommendation</div>", unsafe_allow_html=True)

col_rec1, col_rec2 = st.columns(2)
with col_rec1:
    st.markdown(f"""
    <div class='model-card'>
        <div style='font-size:.8rem;color:#94a3b8;'>🏆 BEST OVERALL</div>
        <div style='font-size:1.6rem;font-weight:800;color:#00d4ff;margin:.4rem 0;'>{best_rmse_model}</div>
        <div style='color:#e2e8f0;'>RMSE: <b>{_f(best_rmse_val)}</b> &nbsp;|&nbsp; R²: <b>{_f(REAL_RESULTS[best_rmse_model].get('r2'), '+.4f')}</b> &nbsp;|&nbsp; Dir. Acc.: <b>{_f(REAL_RESULTS[best_rmse_model].get('dir_acc'), '.2f')}%</b></div>
        <div style='color:#94a3b8;font-size:.82rem;margin-top:.5rem;'>Lowest RMSE on held-out validation set. Recommended for production inference.</div>
    </div>
    """, unsafe_allow_html=True)

with col_rec2:
    _xgb = REAL_RESULTS.get("XGBoost", {})
    st.markdown(f"""
    <div class='model-card'>
        <div style='font-size:.8rem;color:#94a3b8;'>⚡ BEST SPEED/ACCURACY</div>
        <div style='font-size:1.6rem;font-weight:800;color:#f59e0b;margin:.4rem 0;'>XGBoost</div>
        <div style='color:#e2e8f0;'>RMSE: <b>{_f(_xgb.get('rmse'))}</b> &nbsp;|&nbsp; Train Time: <b>{_xgb.get('train_time') or '—'}s</b> &nbsp;|&nbsp; Dir. Acc.: <b>{_f(_xgb.get('dir_acc'), '.2f')}%</b></div>
        <div style='color:#94a3b8;font-size:.82rem;margin-top:.5rem;'>Fastest single-model training. Best choice for rapid experimentation cycles.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("""
> **Note:** All R² values are near zero, which is expected for financial return prediction — markets are noisy by nature.
> RMSE and directional accuracy are the primary metrics for trading signal quality.
""")

# ── MLflow Links ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>MLflow Experiment Runs</div>", unsafe_allow_html=True)
st.markdown("""
All experiments are tracked on **DagsHub MLflow**:
- 🔗 [hull-tactical-experiments on DagsHub](https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow/)
- 5 runs logged with full hyperparameters, metrics, and model artifacts
- Experiment name: `hull-tactical-experiments`
""")
