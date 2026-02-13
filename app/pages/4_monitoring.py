"""Monitoring page for Hull Tactical Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Monitoring - Hull Tactical",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Performance Monitoring")
st.markdown("Monitor model performance and system health in real-time.")


def generate_performance_data(n_days: int = 90) -> pd.DataFrame:
    """Generate sample performance data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")

    cumulative_returns = np.cumsum(np.random.randn(n_days) * 0.005) + 1
    benchmark_returns = np.cumsum(np.random.randn(n_days) * 0.004) + 1
    predictions = np.random.randn(n_days) * 0.005
    actuals = predictions + np.random.randn(n_days) * 0.003
    rolling_rmse = pd.Series((predictions - actuals) ** 2).rolling(20).mean().apply(np.sqrt)

    return pd.DataFrame({
        "date": dates,
        "portfolio_value": cumulative_returns,
        "benchmark_value": benchmark_returns,
        "daily_return": np.random.randn(n_days) * 0.01,
        "rolling_rmse": rolling_rmse,
        "prediction_error": predictions - actuals,
    })


# Generate data
perf_df = generate_performance_data()

# System status
st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model Status",
        "🟢 Active",
    )

with col2:
    st.metric(
        "Last Prediction",
        "2 min ago",
    )

with col3:
    st.metric(
        "API Latency",
        "45 ms",
        delta="-5 ms",
        delta_color="inverse",
    )

with col4:
    st.metric(
        "Uptime",
        "99.9%",
    )

st.markdown("---")

# Performance chart
st.subheader("Portfolio Performance")

fig_perf = go.Figure()

fig_perf.add_trace(go.Scatter(
    x=perf_df["date"],
    y=perf_df["portfolio_value"],
    name="Portfolio",
    line=dict(color="#1f77b4", width=2),
))

fig_perf.add_trace(go.Scatter(
    x=perf_df["date"],
    y=perf_df["benchmark_value"],
    name="Benchmark (S&P 500)",
    line=dict(color="#7f7f7f", width=2, dash="dash"),
))

fig_perf.update_layout(
    title="Cumulative Returns: Portfolio vs Benchmark",
    xaxis_title="Date",
    yaxis_title="Value (Normalized)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig_perf, use_container_width=True)

# Rolling metrics
st.subheader("Rolling Metrics")

col1, col2 = st.columns(2)

with col1:
    fig_rmse = go.Figure()

    fig_rmse.add_trace(go.Scatter(
        x=perf_df["date"],
        y=perf_df["rolling_rmse"],
        name="Rolling RMSE",
        line=dict(color="#2ca02c"),
        fill="tozeroy",
        fillcolor="rgba(44, 160, 44, 0.2)",
    ))

    fig_rmse.update_layout(
        title="20-Day Rolling RMSE",
        xaxis_title="Date",
        yaxis_title="RMSE",
    )

    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    fig_error = go.Figure()

    fig_error.add_trace(go.Scatter(
        x=perf_df["date"],
        y=perf_df["prediction_error"],
        name="Prediction Error",
        line=dict(color="#d62728"),
        fill="tozeroy",
        fillcolor="rgba(214, 39, 40, 0.2)",
    ))

    fig_error.add_hline(y=0, line_dash="dash", line_color="gray")

    fig_error.update_layout(
        title="Daily Prediction Error",
        xaxis_title="Date",
        yaxis_title="Error (Predicted - Actual)",
    )

    st.plotly_chart(fig_error, use_container_width=True)

# Return distribution
st.subheader("Return Distribution")

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        perf_df,
        x="daily_return",
        nbins=30,
        title="Daily Return Distribution",
        color_discrete_sequence=["#1f77b4"],
    )

    fig_hist.update_layout(
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
    )

    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Risk metrics
    st.markdown("### Risk Metrics")

    returns = perf_df["daily_return"]

    metrics = {
        "Mean Daily Return": f"{returns.mean()*100:.3f}%",
        "Daily Volatility": f"{returns.std()*100:.3f}%",
        "Max Drawdown": f"{(returns.cumsum().cummax() - returns.cumsum()).max()*100:.2f}%",
        "Value at Risk (95%)": f"{np.percentile(returns, 5)*100:.3f}%",
        "Skewness": f"{returns.skew():.3f}",
        "Kurtosis": f"{returns.kurtosis():.3f}",
    }

    for metric, value in metrics.items():
        st.metric(metric, value)

# Alerts
st.subheader("Alerts & Notifications")

alert_data = [
    {"time": "10:32 AM", "type": "Info", "message": "Model prediction completed successfully"},
    {"time": "10:30 AM", "type": "Info", "message": "Daily data update received"},
    {"time": "09:00 AM", "type": "Warning", "message": "Prediction RMSE above threshold (0.012)"},
    {"time": "Yesterday", "type": "Info", "message": "Weekly model retraining completed"},
]

for alert in alert_data:
    if alert["type"] == "Warning":
        st.warning(f"**{alert['time']}** - {alert['message']}")
    else:
        st.info(f"**{alert['time']}** - {alert['message']}")

# MLflow integration status
st.subheader("MLflow Integration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Experiment Tracking")
    st.markdown(
        """
        | Metric | Value |
        |--------|-------|
        | Active Experiment | hull-tactical-experiments |
        | Total Runs | 42 |
        | Best RMSE | 0.0085 |
        | Last Run | 2 hours ago |
        """
    )

with col2:
    st.markdown("### Model Registry")
    st.markdown(
        """
        | Model | Stage | Version |
        |-------|-------|---------|
        | hull-tactical-ensemble | Production | 3 |
        | hull-tactical-lightgbm | Staging | 5 |
        | hull-tactical-xgboost | Archived | 4 |
        """
    )

# Footer with refresh
st.markdown("---")
if st.button("🔄 Refresh Data"):
    st.rerun()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
