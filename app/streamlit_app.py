"""Hull Tactical Market Prediction - Streamlit Dashboard."""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Hull Tactical - Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application entry point."""
    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/150x50?text=Hull+Tactical",
            use_container_width=True,
        )
        st.title("Navigation")

        st.markdown("---")

        st.markdown("### Model Selection")
        model_options = ["LightGBM", "XGBoost", "CatBoost", "Ensemble"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=3,  # Default to Ensemble
        )

        st.markdown("---")

        st.markdown("### Settings")
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        auto_refresh = st.checkbox("Auto Refresh", value=False)

        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=30,
                max_value=300,
                value=60,
            )

        st.markdown("---")

        st.markdown("### Quick Links")
        st.markdown("- [GitHub Repository](https://github.com)")
        st.markdown("- [MLflow Dashboard](https://dagshub.com)")
        st.markdown("- [Kaggle Competition](https://kaggle.com)")

    # Main content
    st.markdown('<p class="main-header">Hull Tactical Market Prediction</p>', unsafe_allow_html=True)

    st.markdown(
        """
        Welcome to the Hull Tactical Market Prediction Dashboard. This application
        provides real-time predictions for the S&P 500 excess returns.
        """
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Latest Prediction",
            value="+0.35%",
            delta="0.05%",
            delta_color="normal",
        )

    with col2:
        st.metric(
            label="R² Score",
            value="0.0278",
            delta="0.002",
            delta_color="normal",
        )

    with col3:
        st.metric(
            label="RMSE",
            value="0.0087",
            delta="-0.0002",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            label="Directional Accuracy",
            value="54.8%",
            delta="1.2%",
            delta_color="normal",
        )

    st.markdown("---")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Overview", "🔮 Predictions", "📈 Performance"]
    )

    with tab1:
        st.subheader("Market Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Recent Predictions")
            st.info(
                "Navigate to the **Predictions** page for detailed "
                "prediction analysis and historical data."
            )

        with col2:
            st.markdown("### Model Status")
            st.success(f"**{selected_model}** model is active and running.")

            st.markdown(
                """
                | Metric | Value |
                |--------|-------|
                | Last Update | 2 minutes ago |
                | Predictions Today | 1 |
                | Constraint Status | ✅ OK |
                """
            )

    with tab2:
        st.subheader("Current Predictions")
        st.info(
            "Visit the **Predictions** page from the sidebar for "
            "detailed prediction analysis."
        )

    with tab3:
        st.subheader("Performance Metrics")
        st.info(
            "Visit the **Monitoring** page from the sidebar for "
            "detailed performance tracking."
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Hull Tactical Market Prediction | Built with Streamlit |
        <a href='https://github.com'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
