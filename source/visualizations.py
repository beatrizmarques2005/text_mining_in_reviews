"""
Atlanta Restaurant Reviews — Visualization Utilities
====================================================

Clean and consistent plotting helpers for the Text Mining project.
All charts follow a shared visual language based on the viridis palette
and minimal typography for visual harmony across the report.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# VISUAL STYLE CONFIGURATION
# =============================================================================

class ColorPalette:
    """Centralized color definitions (Viridis-inspired)."""
    DARK_TEXT = "#2D3436"
    MEDIUM_TEXT = "#636E72"
    LIGHT_TEXT = "#A0A4A8"
    WHITE = "white"
    LIGHT_GRAY = "#F8F9FA"
    BORDER_GRAY = "#DDD"
    VIRIDIS = ["#440154", "#3b528b", "#21908d", "#5dc962", "#fde725"]
    RATINGS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]


class ChartConfig:
    """Default layout configuration for all charts."""
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = 600
    LARGE_WIDTH = 1200
    LARGE_HEIGHT = 700
    TITLE_SIZE = 22
    AXIS_TITLE_SIZE = 16
    TICK_SIZE = 12
    TEXT_SIZE = 11
    LEGEND_SIZE = 12
    FONT_FAMILY = "Arial Black"


# =============================================================================
# SHARED LAYOUT FUNCTION
# =============================================================================

def _apply_base_layout(fig, title: str, x_label: str, y_label: str,
                       width: int, height: int) -> None:
    """Apply unified layout, colors, and typography to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=ChartConfig.TITLE_SIZE,
                      family=ChartConfig.FONT_FAMILY,
                      color=ColorPalette.DARK_TEXT)
        ),
        xaxis=dict(
            title=dict(text=x_label,
                       font=dict(size=ChartConfig.AXIS_TITLE_SIZE,
                                 color=ColorPalette.MEDIUM_TEXT)),
            tickangle=-45,
            tickfont=dict(size=ChartConfig.TICK_SIZE,
                          color=ColorPalette.LIGHT_TEXT),
            gridcolor=ColorPalette.LIGHT_GRAY,
            linecolor=ColorPalette.BORDER_GRAY
        ),
        yaxis=dict(
            title=dict(text=y_label,
                       font=dict(size=ChartConfig.AXIS_TITLE_SIZE,
                                 color=ColorPalette.MEDIUM_TEXT)),
            tickfont=dict(size=ChartConfig.TICK_SIZE,
                          color=ColorPalette.LIGHT_TEXT),
            gridcolor=ColorPalette.LIGHT_GRAY,
            linecolor=ColorPalette.BORDER_GRAY
        ),
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
        width=width,
        height=height,
        margin=dict(l=80, r=80, t=100, b=100),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)


# =============================================================================
# BASIC CHARTS
# =============================================================================

def bar_chart(data: pd.DataFrame, x: str, y: str, title: str,
              labels: Dict[str, str], top_n: int = 10) -> None:
    """Bar chart for the top-N records."""
    data = data.sort_values(by=y, ascending=False).head(top_n)
    fig = px.bar(
        data, x=x, y=y, title=title, labels=labels,
        color=y, color_continuous_scale=ColorPalette.VIRIDIS, text=y
    )
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_traces(
        texttemplate="%{text}", textposition="outside",
        marker=dict(line=dict(width=1.5, color=ColorPalette.MEDIUM_TEXT),
                    opacity=0.85)
    )
    fig.show()


def pie_chart(data: pd.DataFrame, names_col: str, values_col: str,
              title: str) -> None:
    """Pie chart using the Viridis palette."""
    fig = go.Figure(
        go.Pie(
            labels=data[names_col], values=data[values_col],
            textinfo="label+percent", hoverinfo="label+value+percent",
            marker=dict(colors=ColorPalette.VIRIDIS,
                        line=dict(color=ColorPalette.WHITE, width=2))
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=ChartConfig.DEFAULT_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
    )
    fig.show()


def histogram_chart(data: pd.DataFrame, column: str, title: str,
                    x_label: str, bins: int = 30) -> None:
    """Histogram with unified style."""
    fig = px.histogram(
        data, x=column, nbins=bins, title=title,
        color_discrete_sequence=[ColorPalette.VIRIDIS[0]]
    )
    _apply_base_layout(fig, title, x_label, "Frequency",
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_traces(marker=dict(line=dict(width=1.2, color=ColorPalette.BORDER_GRAY)))
    fig.show()


# =============================================================================
# COMPARATIVE CHARTS
# =============================================================================

def clustered_bar_chart(data: pd.DataFrame, x: str, y_columns: List[str],
                        title: str, labels: Dict[str, str]) -> None:
    """Grouped bar chart comparing multiple numeric series."""
    fig = go.Figure()
    for i, col in enumerate(y_columns):
        fig.add_trace(
            go.Bar(
                name=labels.get(col, col),
                x=data[x], y=data[col],
                text=data[col], textposition="outside",
                marker=dict(color=ColorPalette.VIRIDIS[i % len(ColorPalette.VIRIDIS)],
                            line=dict(width=1.2, color=ColorPalette.BORDER_GRAY))
            )
        )
    _apply_base_layout(fig, title, labels.get(x, x), "Count",
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1)
    )
    fig.show()


def clustered_bar_charts(data: pd.DataFrame, x: str, y_columns: List[str],
                         title: str, labels: Dict[str, str], top: int = 5) -> None:
    """Display two side-by-side clustered charts for quick comparison."""
    colors = ColorPalette.VIRIDIS
    top_a = data.sort_values(by=y_columns[0], ascending=False).head(top)
    top_b = data.sort_values(by=y_columns[1], ascending=False).head(top)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=[f"Top {top} {labels.get(y_columns[0], y_columns[0])}",
                        f"Top {top} {labels.get(y_columns[1], y_columns[1])}"]
    )

    for i, col in enumerate(y_columns):
        for j, subset in enumerate([top_a, top_b]):
            fig.add_trace(
                go.Bar(
                    x=subset[x], y=subset[col],
                    name=labels.get(col, col),
                    marker_color=colors[i % len(colors)],
                    text=subset[col], textposition="outside",
                    showlegend=(j == 0)
                ),
                row=1, col=j + 1
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="group",
        width=ChartConfig.LARGE_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.05),
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
    )
    fig.show()


# =============================================================================
# ADVANCED CHARTS
# =============================================================================

def scatter_plot(data: pd.DataFrame, x: str, y: str, title: str,
                 labels: Dict[str, str], size: Optional[str] = None) -> None:
    """Scatter plot using continuous Viridis scale."""
    fig = px.scatter(
        data, x=x, y=y, title=title, labels=labels, size=size,
        color_continuous_scale=ColorPalette.VIRIDIS
    )
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.LARGE_WIDTH, ChartConfig.LARGE_HEIGHT)
    fig.show()


def line_plot(data: pd.DataFrame, x: str, y: str, title: str,
              labels: Dict[str, str]) -> None:
    """Line plot for trend visualization."""
    fig = px.line(data, x=x, y=y, title=title, labels=labels,
                 color_discrete_sequence=ColorPalette.VIRIDIS)
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.LARGE_WIDTH, ChartConfig.LARGE_HEIGHT)
    fig.show()


def box_plot(data: pd.DataFrame, x: str, y: str, title: str,
             labels: Dict[str, str]) -> None:
    """Box plot for distribution comparison."""
    fig = px.box(data, x=x, y=y, title=title, labels=labels,
                 color_discrete_sequence=ColorPalette.VIRIDIS)
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.LARGE_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.show()


def treemap_chart(data: pd.DataFrame, path_col: str, value_col: str,
                  title: str) -> None:
    """Treemap for hierarchical data representation."""
    fig = px.treemap(
        data, path=[path_col], values=value_col, title=title,
        color_discrete_sequence=ColorPalette.VIRIDIS
    )
    fig.update_traces(textinfo="label+value")
    fig.show()


# =============================================================================
# GEO VISUALIZATION
# =============================================================================

def extract_coordinates(url: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (lat, lon) coordinates from a Google Maps URL."""
    match = re.search(r"@([-+]?\d*\.\d+),([-+]?\d*\.\d+)", str(url))
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)


def plot_restaurant_map(dataset: pd.DataFrame, color_by: str,
                        url_col: str = "url", title_col: str = "title",
                        category_col: str = "categoryName",
                        reviews_col: str = "reviewsCount",
                        zoom: int = 10, height: int = 700) -> None:
    """Map of restaurants colored by a chosen attribute."""
    dataset[["latitude", "longitude"]] = dataset[url_col].apply(
        lambda u: pd.Series(extract_coordinates(u))
    )
    dataset = dataset.dropna(subset=["latitude", "longitude"]).copy()

    color_args = (
        dict(color_continuous_scale=ColorPalette.RATINGS)
        if pd.api.types.is_numeric_dtype(dataset[color_by])
        else dict(color_discrete_sequence=px.colors.qualitative.Safe)
    )

    fig = px.scatter_mapbox(
        dataset, lat="latitude", lon="longitude", color=color_by,
        hover_name=title_col,
        hover_data={category_col: True, reviews_col: True, color_by: True},
        zoom=zoom, title=f"Restaurant Locations by {color_by}", **color_args
    )
    fig.update_layout(mapbox_style="carto-positron", height=height)
    fig.show()
