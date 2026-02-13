"""Metric display components for Hull Tactical Dashboard."""

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


def display_key_metrics(
    metrics: Dict[str, Tuple[Any, Optional[Any]]],
    columns: int = 4,
) -> None:
    """
    Display key metrics in a row of columns.

    Args:
        metrics: Dictionary of {label: (value, delta)}.
        columns: Number of columns.
    """
    cols = st.columns(columns)

    for i, (label, (value, delta)) in enumerate(metrics.items()):
        with cols[i % columns]:
            if delta is not None:
                st.metric(label=label, value=value, delta=delta)
            else:
                st.metric(label=label, value=value)


def display_model_metrics(
    model_name: str,
    metrics: Dict[str, float],
) -> None:
    """
    Display metrics for a single model.

    Args:
        model_name: Name of the model.
        metrics: Dictionary of metric values.
    """
    st.subheader(f"{model_name} Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "rmse" in metrics:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")

    with col2:
        if "r2" in metrics:
            st.metric("R²", f"{metrics['r2']:.4f}")

    with col3:
        if "mae" in metrics:
            st.metric("MAE", f"{metrics['mae']:.4f}")

    with col4:
        if "directional_accuracy" in metrics:
            st.metric("Dir. Accuracy", f"{metrics['directional_accuracy']:.1f}%")


def display_status_indicator(
    status: str,
    label: str = "Status",
) -> None:
    """
    Display a status indicator.

    Args:
        status: Status value ('ok', 'warning', 'error').
        label: Label for the indicator.
    """
    if status == "ok":
        st.success(f"✅ {label}: OK")
    elif status == "warning":
        st.warning(f"⚠️ {label}: Warning")
    elif status == "error":
        st.error(f"❌ {label}: Error")
    else:
        st.info(f"ℹ️ {label}: {status}")


def display_comparison_table(
    data: List[Dict[str, Any]],
    highlight_best: Optional[str] = None,
) -> None:
    """
    Display a comparison table with optional highlighting.

    Args:
        data: List of dictionaries with comparison data.
        highlight_best: Column to highlight best value.
    """
    import pandas as pd

    df = pd.DataFrame(data)

    if highlight_best and highlight_best in df.columns:
        # Find best value (depends on metric)
        if highlight_best in ["rmse", "mae"]:
            best_idx = df[highlight_best].idxmin()
        else:
            best_idx = df[highlight_best].idxmax()

        # Style the dataframe
        def highlight_row(row):
            if row.name == best_idx:
                return ["background-color: #90EE90"] * len(row)
            return [""] * len(row)

        styled_df = df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
