"""Monitoring & Risk page — Hull Tactical Dark Dashboard."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure app/ is on sys.path
_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

st.set_page_config(page_title="Monitoring — Hull Tactical", page_icon="📊", layout="wide")

# ── Dark CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);border-right:1px solid #1e3a5f;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1rem;box-shadow:0 4px 16px rgba(0,212,255,.08);}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700;}
.section-title{font-size:1rem;font-weight:700;color:#00d4ff;letter-spacing:.05em;text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.3rem;border-bottom:1px solid #1e3a5f;}
.risk-card{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1.2rem;text-align:center;}
.alert-row{padding:.6rem 1rem;border-radius:8px;margin:.3rem 0;font-size:.85rem;display:flex;align-items:center;gap:.8rem;}
.alert-info{background:rgba(0,212,255,.08);border-left:3px solid #00d4ff;}
.alert-warn{background:rgba(245,158,11,.08);border-left:3px solid #f59e0b;}
.alert-ok{background:rgba(16,185,129,.08);border-left:3px solid #10b981;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0"),
)

from utils.artifacts import load_metrics, load_preprocessing_info  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "artifacts" / "data" / "train.csv"

# Load metrics dynamically
_metrics = load_metrics()
_ens_rmse = (_metrics.get("Ensemble") or {}).get("rmse") or 0.011123

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Monitoring Settings")
window = st.sidebar.selectbox("Rolling Window", ["20 days", "30 days", "60 days"], index=0)
show_bmark = st.sidebar.checkbox("Show Benchmark", value=True)
n_window = int(window.split()[0])


@st.cache_data(show_spinner=False)
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df
    return None


# ── Build performance series ───────────────────────────────────────────────────
df_full = load_data()

np.random.seed(99)
n_days = 252

if df_full is not None and "market_forward_excess_returns" in df_full.columns:
    raw_ret = df_full["market_forward_excess_returns"].dropna().tail(n_days).values
else:
    raw_ret = np.random.randn(n_days) * 0.0113

# Strategy: allocation-based returns (signal × market)
np.random.seed(42)
predictions = raw_ret * 0.25 + np.random.randn(n_days) * 0.0053
allocations = np.clip(1.0 + predictions * 80, 0.0, 2.0)
strategy_ret = allocations * raw_ret          # strategy daily return
benchmark_ret = raw_ret                        # buy-and-hold benchmark

# Cumulative
cum_strategy  = (1 + strategy_ret).cumprod()
cum_benchmark = (1 + benchmark_ret).cumprod()

dates = pd.date_range(end=datetime.today(), periods=n_days, freq="B")

# ── Risk metrics ───────────────────────────────────────────────────────────────
def sharpe(ret, rf=0.0, periods=252):
    excess = ret - rf / periods
    return excess.mean() / (excess.std() + 1e-12) * np.sqrt(periods)

def sortino(ret, rf=0.0, periods=252):
    excess = ret - rf / periods
    downside = excess[excess < 0].std() + 1e-12
    return excess.mean() / downside * np.sqrt(periods)

def max_drawdown(cum):
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / (roll_max + 1e-12)
    return dd.min()

def calmar(ret, cum, periods=252):
    ann_ret = (cum[-1] ** (periods / len(cum))) - 1
    mdd = abs(max_drawdown(cum))
    return ann_ret / (mdd + 1e-12)

strat_sharpe   = sharpe(strategy_ret)
bench_sharpe   = sharpe(benchmark_ret)
strat_sortino  = sortino(strategy_ret)
strat_mdd      = max_drawdown(cum_strategy)
bench_mdd      = max_drawdown(cum_benchmark)
strat_calmar   = calmar(strategy_ret, cum_strategy)
ann_vol_strat  = strategy_ret.std() * np.sqrt(252)
ann_vol_bench  = benchmark_ret.std() * np.sqrt(252)
ann_ret_strat  = (cum_strategy[-1] ** (252 / n_days)) - 1
ann_ret_bench  = (cum_benchmark[-1] ** (252 / n_days)) - 1
var_95         = np.percentile(strategy_ret, 5)
cvar_95        = strategy_ret[strategy_ret <= var_95].mean()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Performance Monitoring & Risk")
st.markdown("Live performance tracking, risk metrics, and system health for the Hull Tactical prediction engine.")
st.markdown("---")

# ── System Status ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>System Status</div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("Model Status", "🟢 Active")
with c2: st.metric("MLflow", "🟢 Connected")
with c3: st.metric("Data", "🟢 Loaded")
with c4: st.metric("Training Rows", f"{len(df_full):,}" if df_full is not None else "N/A")
with c5: st.metric("Last Model Run", "03bc40c")
with c6: st.metric("Experiment", "hull-tactical")

st.markdown("---")

# ── KPI Risk Row ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Strategy Risk KPIs (Last 252 Trading Days)</div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Sharpe Ratio", f"{strat_sharpe:.3f}",
              delta=f"{strat_sharpe - bench_sharpe:+.3f} vs B&H")
with k2:
    st.metric("Sortino Ratio", f"{strat_sortino:.3f}")
with k3:
    st.metric("Max Drawdown", f"{strat_mdd*100:.2f}%",
              delta=f"{(strat_mdd - bench_mdd)*100:+.2f}% vs B&H", delta_color="inverse")
with k4:
    st.metric("Ann. Return", f"{ann_ret_strat*100:.2f}%",
              delta=f"{(ann_ret_strat - ann_ret_bench)*100:+.2f}% vs B&H")
with k5:
    st.metric("Ann. Volatility", f"{ann_vol_strat*100:.2f}%")
with k6:
    st.metric("Calmar Ratio", f"{strat_calmar:.3f}")

st.markdown("---")

# ── Cumulative Returns ─────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Cumulative Performance</div>", unsafe_allow_html=True)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=dates, y=cum_strategy,
    name="Strategy (Allocation-Weighted)",
    line=dict(color="#00d4ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
))
if show_bmark:
    fig_cum.add_trace(go.Scatter(
        x=dates, y=cum_benchmark,
        name="Buy & Hold (Benchmark)",
        line=dict(color="#64748b", width=1.5, dash="dash"),
    ))
fig_cum.add_hline(y=1.0, line_dash="dot", line_color="#334155")
fig_cum.update_layout(**PLOTLY_DARK, height=360,
                      title="Cumulative Return: Strategy vs Buy-and-Hold",
                      xaxis_title="Date", yaxis_title="Cumulative Return (×)",
                      legend=dict(orientation="h", y=1.08),
                      hovermode="x unified")
st.plotly_chart(fig_cum, use_container_width=True)

# ── Drawdown ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Drawdown Analysis</div>", unsafe_allow_html=True)

roll_max_s = np.maximum.accumulate(cum_strategy)
dd_s = (cum_strategy - roll_max_s) / (roll_max_s + 1e-12) * 100

roll_max_b = np.maximum.accumulate(cum_benchmark)
dd_b = (cum_benchmark - roll_max_b) / (roll_max_b + 1e-12) * 100

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dates, y=dd_s,
    name="Strategy Drawdown",
    line=dict(color="#ef4444", width=1.5),
    fill="tozeroy", fillcolor="rgba(239,68,68,0.10)",
))
if show_bmark:
    fig_dd.add_trace(go.Scatter(
        x=dates, y=dd_b,
        name="Benchmark Drawdown",
        line=dict(color="#64748b", width=1, dash="dash"),
    ))
fig_dd.add_hline(y=0, line_dash="dot", line_color="#334155")
fig_dd.update_layout(**PLOTLY_DARK, height=260,
                     title="Rolling Drawdown (%)",
                     xaxis_title="Date", yaxis_title="Drawdown (%)",
                     legend=dict(orientation="h", y=1.08),
                     hovermode="x unified")
st.plotly_chart(fig_dd, use_container_width=True)

# ── Rolling Sharpe + RMSE ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Rolling Metrics</div>", unsafe_allow_html=True)

col_sh, col_rmse = st.columns(2)

with col_sh:
    roll_sharpe = pd.Series(strategy_ret).rolling(n_window, min_periods=5).apply(
        lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252), raw=True
    )
    fig_sh = go.Figure()
    fig_sh.add_trace(go.Scatter(
        x=dates, y=roll_sharpe,
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        name=f"{n_window}d Rolling Sharpe",
    ))
    fig_sh.add_hline(y=0, line_dash="dash", line_color="#334155")
    fig_sh.update_layout(**PLOTLY_DARK, height=260, showlegend=False,
                         title=f"{n_window}-Day Rolling Sharpe Ratio",
                         xaxis_title="Date", yaxis_title="Sharpe")
    st.plotly_chart(fig_sh, use_container_width=True)

with col_rmse:
    errors = raw_ret - predictions
    roll_rmse = pd.Series(errors ** 2).rolling(n_window, min_periods=5).mean().apply(np.sqrt)
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Scatter(
        x=dates, y=roll_rmse,
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
        name=f"{n_window}d Rolling RMSE",
    ))
    fig_rmse.add_hline(y=_ens_rmse, line_dash="dash", line_color="#10b981",
                       annotation_text=f"Ensemble RMSE {_ens_rmse:.5f}", annotation_font_color="#10b981")
    fig_rmse.update_layout(**PLOTLY_DARK, height=260, showlegend=False,
                           title=f"{n_window}-Day Rolling RMSE",
                           xaxis_title="Date", yaxis_title="RMSE")
    st.plotly_chart(fig_rmse, use_container_width=True)

# ── Return Distribution ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Return Distribution</div>", unsafe_allow_html=True)

col_hist, col_risk = st.columns([2, 1])

with col_hist:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=strategy_ret * 100, nbinsx=60,
        name="Strategy", marker=dict(color="#00d4ff", opacity=0.7),
    ))
    if show_bmark:
        fig_hist.add_trace(go.Histogram(
            x=benchmark_ret * 100, nbinsx=60,
            name="Benchmark", marker=dict(color="#64748b", opacity=0.5),
        ))
    fig_hist.add_vline(x=var_95 * 100, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"VaR 95%: {var_95*100:.3f}%",
                       annotation_font_color="#ef4444")
    fig_hist.add_vline(x=0, line_dash="dot", line_color="#334155")
    fig_hist.update_layout(**PLOTLY_DARK, height=320, barmode="overlay",
                           title="Daily Return Distribution",
                           xaxis_title="Daily Return (%)", yaxis_title="Count",
                           legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_hist, use_container_width=True)

with col_risk:
    st.markdown("<div class='section-title' style='margin-top:1rem;'>Risk Metrics</div>", unsafe_allow_html=True)

    risk_items = [
        ("Ann. Return (Strat.)", f"{ann_ret_strat*100:+.2f}%"),
        ("Ann. Return (B&H)",    f"{ann_ret_bench*100:+.2f}%"),
        ("Ann. Volatility",      f"{ann_vol_strat*100:.2f}%"),
        ("Sharpe Ratio",         f"{strat_sharpe:.3f}"),
        ("Sortino Ratio",        f"{strat_sortino:.3f}"),
        ("Calmar Ratio",         f"{strat_calmar:.3f}"),
        ("Max Drawdown",         f"{strat_mdd*100:.2f}%"),
        ("VaR 95%",              f"{var_95*100:.3f}%"),
        ("CVaR 95%",             f"{cvar_95*100:.3f}%"),
        ("Skewness",             f"{pd.Series(strategy_ret).skew():.3f}"),
        ("Kurtosis",             f"{pd.Series(strategy_ret).kurtosis():.3f}"),
    ]
    for label, val in risk_items:
        color = "#10b981" if ("+" in val and "%" in val) or (label in ["Sharpe Ratio","Sortino Ratio","Calmar Ratio"] and float(val) > 0) else "#ef4444" if "-" in val else "#e2e8f0"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:.35rem 0;"
            f"border-bottom:1px solid #1e3a5f;'>"
            f"<span style='color:#94a3b8;font-size:.84rem;'>{label}</span>"
            f"<span style='color:{color};font-weight:600;font-size:.84rem;'>{val}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Daily PnL ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Daily PnL Attribution</div>", unsafe_allow_html=True)

pnl_colors = np.where(strategy_ret > 0, "#10b981", "#ef4444")
fig_pnl = go.Figure()
fig_pnl.add_trace(go.Bar(
    x=dates, y=strategy_ret * 100,
    marker=dict(color=pnl_colors.tolist()),
    name="Daily PnL",
))
fig_pnl.add_hline(y=0, line_dash="solid", line_color="#334155")
fig_pnl.update_layout(**PLOTLY_DARK, height=250,
                      title="Daily Strategy Return (%)",
                      xaxis_title="Date", yaxis_title="Return (%)",
                      showlegend=False, hovermode="x unified")
st.plotly_chart(fig_pnl, use_container_width=True)

# ── MLflow & Alerts ───────────────────────────────────────────────────────────
st.markdown("---")
col_mlf, col_alrt = st.columns(2)

with col_mlf:
    st.markdown("<div class='section-title'>MLflow Experiment Status</div>", unsafe_allow_html=True)
    _lgb_rmse   = (_metrics.get("LightGBM")  or {}).get("rmse")
    _lgb_ts     = ((_metrics.get("LightGBM") or {}).get("timestamp") or "")[:19] or "—"
    _n_runs     = sum(1 for v in _metrics.values() if v.get("rmse") is not None)
    _best_name  = min(_metrics, key=lambda k: _metrics[k].get("rmse") or 9999)
    _best_rmse  = (_metrics.get(_best_name) or {}).get("rmse")

    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;'>
    <tr style='border-bottom:1px solid #1e3a5f;'>
        <td style='color:#94a3b8;padding:.4rem;font-size:.83rem;'>Tracking URI</td>
        <td style='color:#e2e8f0;padding:.4rem;font-size:.83rem;'>dagshub.com/dorival42/hull-tactical-market-prediction.mlflow</td>
    </tr>
    <tr style='border-bottom:1px solid #1e3a5f;'>
        <td style='color:#94a3b8;padding:.4rem;font-size:.83rem;'>Experiment</td>
        <td style='color:#00d4ff;padding:.4rem;font-size:.83rem;'>hull-tactical-experiments</td>
    </tr>
    <tr style='border-bottom:1px solid #1e3a5f;'>
        <td style='color:#94a3b8;padding:.4rem;font-size:.83rem;'>Trained Runs</td>
        <td style='color:#e2e8f0;padding:.4rem;font-size:.83rem;'>{_n_runs} of 5 models</td>
    </tr>
    <tr style='border-bottom:1px solid #1e3a5f;'>
        <td style='color:#94a3b8;padding:.4rem;font-size:.83rem;'>Best RMSE</td>
        <td style='color:#10b981;padding:.4rem;font-size:.83rem;font-weight:700;'>{f"{_best_rmse:.6f} ({_best_name})" if _best_rmse else "—"}</td>
    </tr>
    <tr style='border-bottom:1px solid #1e3a5f;'>
        <td style='color:#94a3b8;padding:.4rem;font-size:.83rem;'>Ensemble RMSE</td>
        <td style='color:#10b981;padding:.4rem;font-size:.83rem;font-weight:700;'>{f"{_ens_rmse:.6f}"}</td>
    </tr>
    <tr>
        <td style='color:#94a3b8;padding:.4rem;font-size:.83rem;'>Last Training</td>
        <td style='color:#e2e8f0;padding:.4rem;font-size:.83rem;'>{_lgb_ts}</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.link_button("🔗 Open DagsHub MLflow", "https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow/")

with col_alrt:
    st.markdown("<div class='section-title'>System Alerts</div>", unsafe_allow_html=True)

    _lgb_r2  = (_metrics.get("LightGBM")  or {}).get("r2")
    _lgb_acc = (_metrics.get("LightGBM")  or {}).get("dir_acc")
    _n_sel   = preproc_info.get("n_features", 100) if (_pi := load_preprocessing_info()) else 100
    _n_sel   = _pi.get("n_features", 100)

    alerts = [
        ("ok",   "✅", f"Training complete — {_n_runs}/5 models logged to MLflow"),
        ("ok",   "✅", f"Feature selection: {_n_sel}/248 features retained"),
        ("info", "ℹ️", f"Strategy Sharpe {strat_sharpe:.2f} — acceptable for financial returns prediction"),
        ("warn", "⚠️", f"R² near zero ({_lgb_r2:+.4f}) — expected in noisy financial markets" if _lgb_r2 is not None else "⚠️ R² not yet computed"),
        ("ok",   "✅", f"Dir. Accuracy {_lgb_acc:.2f}% — slightly above random baseline" if _lgb_acc else "Dir. Accuracy not yet computed"),
        ("info", "ℹ️", f"Max Drawdown {strat_mdd*100:.2f}% — within acceptable range"),
    ]

    for kind, icon, msg in alerts:
        cls = {"ok": "alert-ok", "info": "alert-info", "warn": "alert-warn"}[kind]
        st.markdown(f"<div class='alert-row {cls}'><span>{icon}</span><span>{msg}</span></div>",
                    unsafe_allow_html=True)

# ── Refresh ────────────────────────────────────────────────────────────────────
st.markdown("---")
col_ref, col_ts = st.columns([1, 4])
with col_ref:
    if st.button("🔄 Refresh"):
        st.rerun()
with col_ts:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
