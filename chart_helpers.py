from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def diversity_bar_chart(
    df: pd.DataFrame,
    *,
    method_col: str = "method",
    metric_col: str = "metric",
    value_col: str = "value",
    title: str = "Diversity Comparison: VS vs Direct Prompting",
) -> go.Figure:
    """Build a grouped bar chart for diversity metrics.

    Accepts either:
    - long format: columns [method_col, metric_col, value_col]
    - wide format: [method_col, <metric1>, <metric2>, ...]
    """
    if df.empty:
        return go.Figure()

    if {method_col, metric_col, value_col}.issubset(df.columns):
        plot_df = df.copy()
    elif method_col in df.columns:
        metric_columns = [c for c in df.columns if c != method_col]
        if not metric_columns:
            raise ValueError("No metric columns found for diversity bar chart.")
        plot_df = df.melt(
            id_vars=method_col, value_vars=metric_columns, var_name=metric_col, value_name=value_col
        )
    else:
        raise ValueError(
            f"DataFrame must include '{method_col}' and either long-format or wide-format metric columns."
        )

    fig = px.bar(
        plot_df,
        x=metric_col,
        y=value_col,
        color=method_col,
        barmode="group",
        title=title,
        text_auto=".3f",
        color_discrete_sequence=["#6366f1", "#14b8a6"],
    )
    fig.update_layout(height=460, xaxis_title="", yaxis_title="Score")
    return fig


def probability_distribution_chart(
    df: pd.DataFrame,
    *,
    probability_col: str = "probability",
    method_col: str = "method",
    nbins: int = 20,
    title: str = "Idea Probability Distribution",
) -> go.Figure:
    """Build an interactive histogram for idea probabilities."""
    if df.empty or probability_col not in df.columns:
        return go.Figure()

    plot_df = df.copy()
    plot_df[probability_col] = pd.to_numeric(plot_df[probability_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[probability_col])

    if plot_df.empty:
        return go.Figure()

    if method_col not in plot_df.columns:
        plot_df[method_col] = "Ideas"

    fig = px.histogram(
        plot_df,
        x=probability_col,
        color=method_col,
        nbins=nbins,
        barmode="overlay",
        opacity=0.75,
        title=title,
        color_discrete_sequence=["#6366f1", "#14b8a6", "#f59e0b", "#ef4444"],
    )
    fig.update_layout(height=420, xaxis_title="Probability", yaxis_title="Count")
    return fig


def embedding_scatter_plot(
    df: pd.DataFrame,
    *,
    x_col: str = "pc1",
    y_col: str = "pc2",
    cluster_col: str = "cluster",
    hover_cols: Sequence[str] | None = None,
    title: str = "PCA Idea Map with Cluster Labels",
) -> go.Figure:
    """Build an interactive 2D scatter plot for embedding projections."""
    required = {x_col, y_col}
    if not required.issubset(df.columns):
        return go.Figure()

    plot_df = df.copy()
    if cluster_col not in plot_df.columns:
        plot_df[cluster_col] = "0"
    plot_df[cluster_col] = plot_df[cluster_col].astype(str)

    safe_hover: Iterable[str] | None = None
    if hover_cols:
        safe_hover = [c for c in hover_cols if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=cluster_col,
        hover_data=safe_hover,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=520, legend_title_text="Cluster")
    return fig
