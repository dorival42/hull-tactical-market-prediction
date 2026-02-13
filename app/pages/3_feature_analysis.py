"""Feature analysis page for Hull Tactical Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Feature Analysis - Hull Tactical",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Feature Analysis")
st.markdown("Analyze feature importance and correlations.")


def get_sample_feature_importance() -> pd.DataFrame:
    """Generate sample feature importance data."""
    np.random.seed(42)
    features = [
        "feat_lagged_target", "feat_target_rolling_mean_5", "feat_v_mean",
        "feat_momentum_strength", "feat_target_zscore", "E19",
        "feat_lagged_target_2", "feat_m_mean", "feat_autocorr",
        "feat_target_volatility_20", "P4", "feat_s_mean",
        "feat_mean_reversion", "feat_i_spread", "V3",
        "feat_vol_momentum_interaction", "M5", "feat_day_sin",
        "feat_momentum_balance", "feat_e_mean",
    ]

    importance = np.random.exponential(scale=50, size=len(features))
    importance = sorted(importance, reverse=True)

    return pd.DataFrame({
        "feature": features,
        "importance": importance,
        "category": [
            "Target", "Target", "Volatility", "Momentum", "Target", "Economic",
            "Target", "Momentum", "Target", "Target", "Price", "Sentiment",
            "Target", "Interest", "Volatility", "Interaction", "Momentum",
            "Time", "Momentum", "Economic",
        ],
    })


# Get feature importance
importance_df = get_sample_feature_importance()

# Top features bar chart
st.subheader("Top 20 Feature Importance")

fig_importance = px.bar(
    importance_df.head(20),
    x="importance",
    y="feature",
    color="category",
    orientation="h",
    title="Feature Importance (LightGBM)",
    color_discrete_sequence=px.colors.qualitative.Set2,
)

fig_importance.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    legend_title="Category",
)

st.plotly_chart(fig_importance, use_container_width=True)

# Feature category breakdown
st.subheader("Importance by Category")

col1, col2 = st.columns(2)

with col1:
    category_importance = importance_df.groupby("category")["importance"].sum().reset_index()
    category_importance = category_importance.sort_values("importance", ascending=False)

    fig_category = px.pie(
        category_importance,
        values="importance",
        names="category",
        title="Feature Importance Distribution by Category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    category_count = importance_df.groupby("category").size().reset_index(name="count")

    fig_count = px.bar(
        category_count,
        x="category",
        y="count",
        color="category",
        title="Number of Features by Category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_count.update_layout(showlegend=False)
    st.plotly_chart(fig_count, use_container_width=True)

# Feature correlation heatmap
st.subheader("Feature Correlations")

# Generate sample correlation matrix
np.random.seed(42)
n_features = 15
feature_names = importance_df["feature"].head(n_features).tolist()
corr_matrix = np.random.randn(n_features, n_features) * 0.3
corr_matrix = (corr_matrix + corr_matrix.T) / 2
np.fill_diagonal(corr_matrix, 1)
corr_matrix = np.clip(corr_matrix, -1, 1)

fig_corr = px.imshow(
    corr_matrix,
    x=feature_names,
    y=feature_names,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Top 15 Features Correlation Matrix",
    zmin=-1,
    zmax=1,
)

fig_corr.update_layout(
    xaxis=dict(tickangle=45),
)

st.plotly_chart(fig_corr, use_container_width=True)

# Feature statistics
st.subheader("Feature Statistics")

# Generate sample statistics
stats_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean": np.random.randn(n_features) * 0.1,
    "Std": np.random.exponential(scale=0.1, size=n_features),
    "Min": np.random.randn(n_features) - 2,
    "Max": np.random.randn(n_features) + 2,
    "Missing %": np.random.uniform(0, 5, n_features),
})

# Format for display
display_stats = stats_df.copy()
for col in ["Mean", "Std", "Min", "Max"]:
    display_stats[col] = display_stats[col].apply(lambda x: f"{x:.4f}")
display_stats["Missing %"] = display_stats["Missing %"].apply(lambda x: f"{x:.2f}%")

st.dataframe(
    display_stats,
    use_container_width=True,
    hide_index=True,
)

# Feature selection
st.subheader("Feature Selection Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Features", "150")

with col2:
    st.metric("Original Columns", "95")

with col3:
    st.metric("Engineered Features", "~200")

st.info(
    """
    **Feature Selection Method:** LightGBM Importance

    Features are selected based on:
    1. Train a LightGBM model on the full feature set
    2. Rank features by importance score
    3. Select top N features (default: 150)
    4. Verify no data leakage from excluded columns
    """
)

# Download feature list
csv = importance_df.to_csv(index=False)
st.download_button(
    label="Download Feature Importance CSV",
    data=csv,
    file_name="feature_importance.csv",
    mime="text/csv",
)
