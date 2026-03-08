"""Predictions page — Hull Tactical Dark Dashboard."""

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

st.set_page_config(page_title="Predictions — Hull Tactical", page_icon="🔮", layout="wide")

# ── Dark CSS (shared) ─────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);border-right:1px solid #1e3a5f;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0f1f3d,#1a2f4e);border:1px solid #1e3a5f;border-radius:12px;padding:1rem;box-shadow:0 4px 16px rgba(0,212,255,.08);}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700;}
.section-title{font-size:1rem;font-weight:700;color:#00d4ff;letter-spacing:.05em;text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.3rem;border-bottom:1px solid #1e3a5f;}
.signal-box{border-radius:12px;padding:1.2rem;text-align:center;margin-bottom:.5rem;}
.bullish{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);}
.bearish{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);}
.neutral{background:rgba(100,116,139,.12);border:1px solid rgba(100,116,139,.3);}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.8)",
    font=dict(color="#e2e8f0"),
)

from utils.artifacts import load_train_data as load_data, load_model_pkl, ARTIFACTS_DIR  # noqa: E402

_PKL_MAP = {
    "LightGBM":     "lightgbm_final.pkl",
    "XGBoost":      "xgboost_final.pkl",
    "CatBoost":     "catboost_final.pkl",
    "RandomForest": "random_forest_final.pkl",
    "Ensemble":     "ensemble_final.pkl",
}


# ── Real prediction helper ────────────────────────────────────────────────────

def _get_predictions(df: pd.DataFrame, model_name: str) -> np.ndarray | None:
    """
    Run real inference using the saved model pkl.

    Steps:
    1. Load the pkl (model + feature_names).
    2. Intersect saved feature_names with columns available in df.
    3. fillna(0) to match training behaviour (training used fillna(0) after pipeline).
    4. Call model.predict().

    Returns None if the model file is missing or inference fails.
    """
    model_data = load_model_pkl(model_name)
    if model_data is None or "_load_error" in model_data:
        return None

    try:
        # Ensemble stores a list of sub-models; individual models store the raw model directly.
        if "models" in model_data:
            # Ensemble: aggregate predictions from sub-models using saved weights
            weights_map: dict = model_data.get("weights", {})
            sub_preds = []
            sub_weights = []
            for sub in model_data["models"]:
                feat_names = sub.get("feature_names") or model_data.get("feature_names", [])
                avail = [f for f in feat_names if f in df.columns]
                if not avail:
                    continue
                X = df[avail].fillna(0)
                pred = sub["model"].predict(X)
                sub_preds.append(pred)
                sub_weights.append(weights_map.get(sub["name"], 1.0))
            if not sub_preds:
                return None
            weights_arr = np.array(sub_weights)
            weights_arr = weights_arr / weights_arr.sum()
            return np.average(np.array(sub_preds), axis=0, weights=weights_arr)
        else:
            # Individual model
            feat_names = model_data.get("feature_names") or []
            avail = [f for f in feat_names if f in df.columns]
            if not avail:
                return None
            X = df[avail].fillna(0)
            return model_data["model"].predict(X)
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
date_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 30 days", "Last 90 days", "Last 252 days", "Full history"],
    index=1,
)
model_type = st.sidebar.selectbox(
    "Model", ["Ensemble", "LightGBM", "XGBoost", "CatBoost", "RandomForest"]
)

# Map to n recent trading days
range_map = {"Last 30 days": 30, "Last 90 days": 90, "Last 252 days": 252, "Full history": 9000}
n_days = range_map[date_range]

# ── Load data ─────────────────────────────────────────────────────────────────
df_full = load_data()

st.title("🔮 Predictions & Signal Analysis")
st.markdown("Historical predictions, allocation signals, and error analysis based on real trained models.")
st.markdown("---")

if df_full is not None:
    df_slice = df_full.tail(n_days).copy().reset_index(drop=True)
    target = df_slice["market_forward_excess_returns"].dropna()
    df_for_pred = df_slice.loc[target.index].reset_index(drop=True)
    target = target.reset_index(drop=True)

    # ── Run real inference ─────────────────────────────────────────────────────
    predictions = _get_predictions(df_for_pred, model_type)

    if predictions is None:
        pkl_path = ARTIFACTS_DIR / _PKL_MAP.get(model_type, "")
        if pkl_path.exists():
            model_data = load_model_pkl(model_type)
            load_error = model_data.get("_load_error") if isinstance(model_data, dict) else None
            st.error(
                f"⚠️ **{model_type}** artifact found but failed to load — "
                "likely a package version mismatch between training and deployment. "
                f"`{pkl_path.name}` was saved with the local environment "
                "(scikit-learn 1.7, xgboost 3.1, lightgbm 4.5). "
                "Check that `requirements.txt` matches those versions."
            )
            if load_error:
                with st.expander("🔍 Error details"):
                    st.code(load_error)
        else:
            st.warning(
                f"⚠️ Model artifact for **{model_type}** not found in `artifacts/`. "
                "Run `python scripts/retrain.py` to train the models first."
            )
        st.stop()

    predictions = predictions[: len(target)]
    errors = target.values - predictions
    correct_dir = np.sign(predictions) == np.sign(target.values)

    # Allocations clipped to [0, 2]
    allocations = np.clip(1.0 + predictions * 80, 0.0, 2.0)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    last_pred = predictions[-1]
    last_alloc = allocations[-1]
    dir_acc = correct_dir.mean() * 100
    rmse = np.sqrt((errors ** 2).mean())
    direction = "📈 Bullish" if last_pred > 0 else "📉 Bearish"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Latest Signal", f"{last_pred*100:+.4f}%",
                  delta=f"{(last_pred - predictions[-2])*100:+.4f}%")
    with c2:
        st.metric("Direction", direction)
    with c3:
        st.metric("Allocation", f"{last_alloc:.3f}×",
                  delta=f"{last_alloc - 1:.3f}× vs neutral", delta_color="normal")
    with c4:
        st.metric("Dir. Accuracy", f"{dir_acc:.1f}%")
    with c5:
        st.metric("RMSE", f"{rmse:.5f}")

    st.markdown("---")

    # ── Signal + Allocation side by side ─────────────────────────────────────
    col_sig, col_alloc = st.columns([1, 2])
    signal_class = "bullish" if last_pred > 0 else "bearish"
    with col_sig:
        st.markdown("<div class='section-title'>Current Signal</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='signal-box {signal_class}'>
            <div style='font-size:3rem;'>{'📈' if last_pred > 0 else '📉'}</div>
            <div style='font-size:1.8rem;font-weight:800;
                color:{"#10b981" if last_pred > 0 else "#ef4444"};'>
                {last_pred*100:+.4f}%
            </div>
            <div style='color:#94a3b8;font-size:.85rem;margin-top:.4rem;'>Predicted excess return</div>
            <hr style='border-color:#1e3a5f;margin:.8rem 0;'>
            <div style='font-size:1.2rem;font-weight:700;color:#e2e8f0;'>Alloc: {last_alloc:.3f}×</div>
            <div style='color:#94a3b8;font-size:.78rem;'>0.0 (cash) → 1.0 (neutral) → 2.0 (leveraged)</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=last_alloc,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "×", "font": {"color": "#e2e8f0", "size": 28}},
            gauge={
                "axis": {"range": [0, 2], "tickcolor": "#64748b"},
                "bar": {"color": "#00d4ff"},
                "bgcolor": "#111827",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 0.7],  "color": "rgba(239,68,68,.25)"},
                    {"range": [0.7, 1.3], "color": "rgba(100,116,139,.15)"},
                    {"range": [1.3, 2],  "color": "rgba(16,185,129,.25)"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 3}, "value": 1.0},
            },
        ))
        fig_gauge.update_layout(**PLOTLY_DARK, height=220, margin=dict(t=20, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_alloc:
        st.markdown("<div class='section-title'>Predictions vs Actual Returns</div>", unsafe_allow_html=True)
        x_axis = list(range(len(target)))
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=x_axis, y=target.values * 100, name="Actual",
            line=dict(color="#10b981", width=1.2), opacity=0.8,
        ))
        fig_pred.add_trace(go.Scatter(
            x=x_axis, y=predictions * 100, name="Predicted",
            line=dict(color="#00d4ff", width=2),
        ))
        # Confidence band (±1 std of prediction errors)
        std_err = np.std(errors) * 1.96
        fig_pred.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=list((predictions + std_err) * 100) + list((predictions - std_err) * 100)[::-1],
            fill="toself", fillcolor="rgba(0,212,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True,
        ))
        fig_pred.add_hline(y=0, line_dash="dot", line_color="#334155")
        fig_pred.update_layout(**PLOTLY_DARK, height=370, hovermode="x unified",
                               title=f"{model_type} — {date_range}",
                               xaxis_title="Trading Day (recent)", yaxis_title="Return (%)",
                               legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_pred, use_container_width=True)

    # ── Allocation History ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Allocation Signal History</div>", unsafe_allow_html=True)
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Bar(
        x=list(range(len(allocations))),
        y=allocations - 1,  # Center on 0
        marker=dict(color=np.where(allocations > 1, "#10b981", "#ef4444")),
        name="Allocation offset (vs 1.0×)",
    ))
    fig_alloc.add_hline(y=0, line_dash="dash", line_color="#64748b",
                        annotation_text="Neutral (1.0×)", annotation_font_color="#64748b")
    fig_alloc.update_layout(**PLOTLY_DARK, height=250,
                            title="Daily Allocation Offset from Neutral (1.0×)",
                            xaxis_title="Trading Day", yaxis_title="Allocation - 1.0",
                            showlegend=False)
    st.plotly_chart(fig_alloc, use_container_width=True)

    # ── Error analysis ────────────────────────────────────────────────────────
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("<div class='section-title'>Prediction Error Distribution</div>", unsafe_allow_html=True)
        fig_err_hist = go.Figure()
        fig_err_hist.add_trace(go.Histogram(
            x=errors * 100, nbinsx=40,
            marker=dict(color="#7b5ea7", line=dict(width=0)),
            name="Error",
        ))
        fig_err_hist.add_vline(x=0, line_dash="dash", line_color="#00d4ff")
        fig_err_hist.update_layout(**PLOTLY_DARK, height=260,
                                   title="Error (Actual − Predicted) Distribution",
                                   xaxis_title="Error (%)", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_err_hist, use_container_width=True)

    with col_e2:
        st.markdown("<div class='section-title'>Directional Accuracy Rolling (20d)</div>", unsafe_allow_html=True)
        rolling_acc = pd.Series(correct_dir.astype(float)).rolling(20, min_periods=1).mean() * 100
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=list(range(len(rolling_acc))), y=rolling_acc.values,
            line=dict(color="#00d4ff", width=2), fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)", name="Rolling Dir. Acc.",
        ))
        fig_roll.add_hline(y=50, line_dash="dash", line_color="#ef4444",
                           annotation_text="Random baseline 50%", annotation_font_color="#ef4444")
        fig_roll.update_layout(**PLOTLY_DARK, height=260, showlegend=False,
                               title="20-Day Rolling Directional Accuracy (%)",
                               xaxis_title="Trading Day", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig_roll, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Recent Predictions Detail</div>", unsafe_allow_html=True)
    n_show = min(30, len(target))
    table_df = pd.DataFrame({
        "Day":        range(len(target) - n_show, len(target)),
        "Predicted":  [f"{p*100:+.4f}%" for p in predictions[-n_show:]],
        "Actual":     [f"{a*100:+.4f}%" for a in target.values[-n_show:]],
        "Error":      [f"{e*100:+.4f}%" for e in errors[-n_show:]],
        "Direction":  ["✅" if c else "❌" for c in correct_dir[-n_show:]],
        "Allocation": [f"{a:.3f}×" for a in allocations[-n_show:]],
    })
    st.dataframe(table_df.iloc[::-1], use_container_width=True, hide_index=True)

    csv = pd.DataFrame({
        "prediction": predictions,
        "actual": target.values,
        "error": errors,
        "allocation": allocations,
    }).to_csv(index=False)
    st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

else:
    st.error("Training data not found. Run the download script first.")
