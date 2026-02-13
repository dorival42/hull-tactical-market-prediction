"""Model comparison page for Hull Tactical Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Model Comparison - Hull Tactical",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Model Comparison")
st.markdown("Compare performance metrics across different models.")


def get_sample_model_metrics() -> pd.DataFrame:
    """Generate sample model metrics for demonstration."""
    return pd.DataFrame({
        "Model": ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "Ensemble"],
        "RMSE": [0.0089, 0.0092, 0.0091, 0.0098, 0.0087],
        "MAE": [0.0068, 0.0071, 0.0070, 0.0076, 0.0066],
        "R²": [0.0245, 0.0198, 0.0215, 0.0112, 0.0278],
        "Directional Accuracy": [54.2, 53.8, 54.0, 52.5, 54.8],
    })


# Get model metrics
metrics_df = get_sample_model_metrics()

# Overall comparison
st.subheader("Overall Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    fig_rmse = px.bar(
        metrics_df,
        x="Model",
        y="RMSE",
        color="RMSE",
        color_continuous_scale="Reds_r",
        title="RMSE by Model (Lower is Better)",
    )
    fig_rmse.update_layout(showlegend=False)
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    fig_mae = px.bar(
        metrics_df,
        x="Model",
        y="MAE",
        color="MAE",
        color_continuous_scale="Oranges_r",
        title="MAE by Model (Lower is Better)",
    )
    fig_mae.update_layout(showlegend=False)
    st.plotly_chart(fig_mae, use_container_width=True)

# Radar chart
st.subheader("Multi-Metric Comparison")

# Normalize metrics for radar chart
metrics_normalized = metrics_df.copy()
for col in ["RMSE", "MAE", "R²", "Directional Accuracy"]:
    if col in ["RMSE", "MAE"]:  # Lower is better
        metrics_normalized[col] = 1 - (metrics_df[col] - metrics_df[col].min()) / (
            metrics_df[col].max() - metrics_df[col].min() + 1e-10
        )
    else:
        metrics_normalized[col] = (metrics_df[col] - metrics_df[col].min()) / (
            metrics_df[col].max() - metrics_df[col].min() + 1e-10
        )

# Create radar chart
categories = ["RMSE", "MAE", "R²", "Directional Accuracy"]

fig_radar = go.Figure()

colors = px.colors.qualitative.Set2

for i, model in enumerate(metrics_normalized["Model"]):
    values = metrics_normalized[metrics_normalized["Model"] == model][categories].values[0]
    values = np.append(values, values[0])  # Close the radar

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name=model,
        line_color=colors[i % len(colors)],
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    title="Normalized Performance Metrics",
)

st.plotly_chart(fig_radar, use_container_width=True)

# Detailed metrics table
st.subheader("Detailed Metrics")

# Format the dataframe for display
display_df = metrics_df.copy()
display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"{x:.4f}")
display_df["MAE"] = display_df["MAE"].apply(lambda x: f"{x:.4f}")
display_df["R²"] = display_df["R²"].apply(lambda x: f"{x:.4f}")
display_df["Directional Accuracy"] = display_df["Directional Accuracy"].apply(lambda x: f"{x:.1f}%")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# Model selection recommendation
st.subheader("Recommendation")

best_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
best_rmse = metrics_df["RMSE"].min()

st.success(
    f"**Recommended Model: {best_model}** with RMSE of {best_rmse:.4f}"
)

st.info(
    """
    **Selection Criteria:**
    - Primary metric: RMSE (prediction accuracy)
    - Secondary: MAE and Directional Accuracy
    - Model stability across walk-forward folds
    """
)

# Walk-forward validation results
st.subheader("Walk-Forward Validation Results")

# Sample walk-forward data
wf_data = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "LightGBM": [0.0091, 0.0088, 0.0087, 0.0090, 0.0089],
    "XGBoost": [0.0094, 0.0091, 0.0090, 0.0093, 0.0092],
    "CatBoost": [0.0093, 0.0090, 0.0089, 0.0092, 0.0091],
    "Ensemble": [0.0089, 0.0086, 0.0085, 0.0088, 0.0087],
})

fig_wf = go.Figure()

for model in ["LightGBM", "XGBoost", "CatBoost", "Ensemble"]:
    fig_wf.add_trace(go.Scatter(
        x=wf_data["Fold"],
        y=wf_data[model],
        mode="lines+markers",
        name=model,
    ))

fig_wf.update_layout(
    title="RMSE by Validation Fold",
    xaxis_title="Fold",
    yaxis_title="RMSE",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
)

st.plotly_chart(fig_wf, use_container_width=True)
