"""Predictions page for Hull Tactical Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Predictions - Hull Tactical",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Predictions")
st.markdown("View current and historical predictions.")


def generate_sample_predictions(n_days: int = 30) -> pd.DataFrame:
    """Generate sample prediction data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")

    predictions = np.random.randn(n_days) * 0.005
    actuals = predictions + np.random.randn(n_days) * 0.003

    # Calculate directional accuracy
    correct_direction = np.sign(predictions) == np.sign(actuals)

    return pd.DataFrame({
        "date": dates,
        "prediction": predictions,
        "actual": actuals,
        "error": actuals - predictions,
        "correct_direction": correct_direction,
    })


# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
    index=1,
)

model_type = st.sidebar.selectbox(
    "Model",
    ["Ensemble", "LightGBM", "XGBoost", "CatBoost"],
)

# Generate sample data
df = generate_sample_predictions(30)

# Current prediction card
st.subheader("Current Prediction")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted Return",
        f"{df['prediction'].iloc[-1]*100:.3f}%",
        delta=f"{(df['prediction'].iloc[-1] - df['prediction'].iloc[-2])*100:.3f}%",
    )

with col2:
    direction = "📈 Bullish" if df['prediction'].iloc[-1] > 0 else "📉 Bearish"
    st.metric("Market Direction", direction)

with col3:
    dir_acc = df['correct_direction'].mean() * 100
    st.metric("Directional Accuracy", f"{dir_acc:.1f}%")

with col4:
    rmse = np.sqrt((df['error'] ** 2).mean())
    st.metric("RMSE", f"{rmse:.4f}")

st.markdown("---")

# Prediction chart
st.subheader("Prediction vs Actual")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["prediction"] * 100,
    name="Prediction",
    line=dict(color="#1f77b4", width=2),
))

fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["actual"] * 100,
    name="Actual",
    line=dict(color="#2ca02c", width=2),
))

fig.update_layout(
    title="Predictions vs Actual Returns",
    xaxis_title="Date",
    yaxis_title="Return (%)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)

# Prediction Error chart
st.subheader("Prediction Error History")

fig_error = px.bar(
    df,
    x="date",
    y="error",
    color="error",
    color_continuous_scale=["red", "white", "green"],
    title="Daily Prediction Errors (Actual - Predicted)",
)

fig_error.update_layout(
    xaxis_title="Date",
    yaxis_title="Error",
)

st.plotly_chart(fig_error, use_container_width=True)

# Prediction table
st.subheader("Detailed Predictions")

display_df = df.copy()
display_df["prediction"] = display_df["prediction"].apply(lambda x: f"{x*100:.4f}%")
display_df["actual"] = display_df["actual"].apply(lambda x: f"{x*100:.4f}%")
display_df["error"] = display_df["error"].apply(lambda x: f"{x*100:.4f}%")
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    display_df.iloc[::-1],
    use_container_width=True,
    hide_index=True,
)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="Download Predictions CSV",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv",
)
