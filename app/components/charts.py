"""Chart components for Hull Tactical Dashboard."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_prediction_chart(
    dates: List,
    predictions: List[float],
    actuals: Optional[List[float]] = None,
    title: str = "Predictions vs Actuals",
) -> go.Figure:
    """
    Create a line chart comparing predictions and actuals.

    Args:
        dates: List of dates.
        predictions: List of prediction values.
        actuals: Optional list of actual values.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=[p * 100 for p in predictions],
        name="Prediction",
        line=dict(color="#1f77b4", width=2),
    ))

    if actuals:
        fig.add_trace(go.Scatter(
            x=dates,
            y=[a * 100 for a in actuals],
            name="Actual",
            line=dict(color="#2ca02c", width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_performance_chart(
    dates: List,
    portfolio_values: List[float],
    benchmark_values: Optional[List[float]] = None,
    title: str = "Portfolio Performance",
) -> go.Figure:
    """
    Create a performance comparison chart.

    Args:
        dates: List of dates.
        portfolio_values: Portfolio cumulative values.
        benchmark_values: Optional benchmark values.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        name="Portfolio",
        line=dict(color="#1f77b4", width=2),
    ))

    if benchmark_values:
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            name="Benchmark",
            line=dict(color="#7f7f7f", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    return fig


def create_feature_importance_chart(
    features: List[str],
    importance: List[float],
    categories: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Feature Importance",
) -> go.Figure:
    """
    Create a horizontal bar chart for feature importance.

    Args:
        features: Feature names.
        importance: Importance scores.
        categories: Optional feature categories for coloring.
        top_n: Number of top features to show.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    df = pd.DataFrame({
        "feature": features,
        "importance": importance,
    })

    if categories:
        df["category"] = categories

    df = df.nlargest(top_n, "importance")

    if categories:
        fig = px.bar(
            df,
            x="importance",
            y="feature",
            color="category",
            orientation="h",
            title=title,
        )
    else:
        fig = px.bar(
            df,
            x="importance",
            y="feature",
            orientation="h",
            title=title,
        )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Importance Score",
        yaxis_title="Feature",
    )

    return fig


def create_correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: List[str],
    title: str = "Feature Correlations",
) -> go.Figure:
    """
    Create a correlation heatmap.

    Args:
        corr_matrix: Correlation matrix.
        labels: Feature labels.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = px.imshow(
        corr_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=title,
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(
        xaxis=dict(tickangle=45),
    )

    return fig


def create_model_comparison_chart(
    models: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison",
) -> go.Figure:
    """
    Create a grouped bar chart comparing models.

    Args:
        models: Model names.
        metrics: Dictionary of metric_name -> values.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for metric_name, values in metrics.items():
        fig.add_trace(go.Bar(
            name=metric_name,
            x=models,
            y=values,
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
    )

    return fig


def create_radar_chart(
    categories: List[str],
    values_dict: Dict[str, List[float]],
    title: str = "Multi-Metric Comparison",
) -> go.Figure:
    """
    Create a radar chart for multi-metric comparison.

    Args:
        categories: Metric categories.
        values_dict: Dictionary of model_name -> normalized values.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, (model_name, values) in enumerate(values_dict.items()):
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=model_name,
            line_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title,
    )

    return fig
