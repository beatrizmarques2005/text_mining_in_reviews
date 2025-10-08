"""
Atlanta Restaurant Reviews - Exploration Module

This module contains visualization functions for exploring restaurant review data.
All functions use a consistent VIRIDIS color palette and professional styling.

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional

# =============================================================================
# COLOR PALETTE & CONSTANTS
# =============================================================================

class ColorPalette:
    """
    All visualizations automatically use the beautiful predefined viridis color palette.
    No color customization is allowed - this ensures perfect visual consistency across all charts!
    """
    
    # Text colors
    DARK_TEXT = '#2D3436'     
    MEDIUM_TEXT = '#636E72'   
    LIGHT_TEXT = '#A0A4A8'     
    
    # Background colors
    LIGHT_BACKGROUND = '#FFEAA7' 
    BORDER_GRAY = '#DDD'
    WHITE = 'white'
    LIGHT_GRAY = '#F8F9FA' 

    # Palletes
    VIRIDIS_SEQUENCE = ['#440154', '#3b528b', '#21908d', '#5dc962', '#fde725', '#31688e']  # Smooth gradient
    MAIN_SEQUENCE = VIRIDIS_SEQUENCE  # Use viridis as default
    PASTEL_SEQUENCE = ['#b48ef7', '#81b1d4', '#8dd3c7', '#b9e769', '#fef3a0', '#f7b267']  # Soft viridis tones with separation
    VIBRANT_SEQUENCE = ['#482878', '#3e8ebd', '#1fa187', '#90d743', '#fde725', '#ff6b35']  # Vibrant, high-contrast viridis
    RATINGS_SEQUENCE = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']

class ChartConfig:
    """Default configuration for charts"""
    
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = 600
    LARGE_WIDTH = 1200
    LARGE_HEIGHT = 700
    
    TITLE_SIZE = 22
    AXIS_TITLE_SIZE = 16
    TICK_SIZE = 12
    TEXT_SIZE = 11
    LEGEND_SIZE = 12
    
    FONT_FAMILY = 'Arial Black'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _apply_base_layout(fig, title: str, x_label: str, y_label: str, width: int, height: int) -> None:
    """Apply consistent base layout styling to all charts"""
    fig.update_layout(
        # Title styling
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': ChartConfig.TITLE_SIZE, 'family': ChartConfig.FONT_FAMILY, 'color': ColorPalette.DARK_TEXT}
        },
        
        # Axis styling
        xaxis={
            'title': {'text': x_label, 'font': {'size': ChartConfig.AXIS_TITLE_SIZE, 'color': ColorPalette.MEDIUM_TEXT}},
            'tickangle': -45,
            'tickfont': {'size': ChartConfig.TICK_SIZE, 'color': ColorPalette.LIGHT_TEXT},
            'gridcolor': ColorPalette.LIGHT_GRAY,
            'linecolor': ColorPalette.BORDER_GRAY
        },
        
        yaxis={
            'title': {'text': y_label, 'font': {'size': ChartConfig.AXIS_TITLE_SIZE, 'color': ColorPalette.MEDIUM_TEXT}},
            'tickfont': {'size': ChartConfig.TICK_SIZE, 'color': ColorPalette.LIGHT_TEXT},
            'gridcolor': ColorPalette.LIGHT_GRAY,
            'linecolor': ColorPalette.BORDER_GRAY
        },
        
        # Overall layout
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
        width=width,
        height=height,
        margin=dict(l=80, r=80, t=100, b=100)
    )
    
    # Add subtle grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=ColorPalette.LIGHT_GRAY)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=ColorPalette.LIGHT_GRAY)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, labels: Dict[str, str]) -> None:
    """
    Create a beautiful and professional bar chart with enhanced styling.
    
    Args:
        data: DataFrame containing the data
        x: column name for x-axis
        y: column name for y-axis  
        title: chart title
        labels: dictionary for axis labels
    """
    color_palette = ColorPalette.MAIN_SEQUENCE
    width, height = ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT

    # Create the bar chart with enhanced styling
    fig = px.bar(
        data, 
        x=x, 
        y=y, 
        title=title, 
        labels=labels,
        color=y,  # Color bars by their values
        color_continuous_scale=color_palette,
        text=y  # Show values on bars
    )
    
    # Apply base layout
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y), width, height)
    
    # Additional styling specific to bar charts
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False
    )
    
    # Style the bars
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        textfont={'size': ChartConfig.TEXT_SIZE, 'color': ColorPalette.DARK_TEXT},
        marker={
            'line': {'width': 1.5, 'color': ColorPalette.MEDIUM_TEXT},
            'opacity': 0.8
        },
        hovertemplate='<b>%{x}</b><br>%{y}<br><extra></extra>'
    )
    
    fig.show()

def clustered_bar_chart(data: pd.DataFrame, x: str, y_columns: List[str], title: str, 
                       labels: Dict[str, str]) -> None:
    """
    Create a beautiful clustered (grouped) bar chart with multiple series.
    
    Args:
        data: DataFrame containing the data
        x: column name for x-axis (categories)
        y_columns: list of column names for y-axis (multiple series to compare)
        title: chart title
        labels: dictionary for axis labels
    """
    
    # Always use predefined viridis colors
    colors = ColorPalette.MAIN_SEQUENCE
    width, height = ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT
    
    # Create the clustered bar chart
    fig = go.Figure()
    
    # Add each series as a separate bar trace
    for i, col in enumerate(y_columns):
        fig.add_trace(go.Bar(
            name=labels.get(col, col),  # Legend name
            x=data[x],
            y=data[col],
            text=data[col],
            texttemplate='%{text}',
            textposition='outside',
            textfont={'size': ChartConfig.TEXT_SIZE, 'color': ColorPalette.DARK_TEXT},
            marker={
                'color': colors[i % len(colors)],
                'line': {'width': 1.5, 'color': ColorPalette.MEDIUM_TEXT},
                'opacity': 0.8
            },
            hovertemplate=f'<b>{labels.get(col, col)}</b><br>%{{x}}<br>%{{y}}<extra></extra>'
        ))
    
    # Apply base layout with custom y-label
    _apply_base_layout(fig, title, labels.get(x, x), 'Count', width, height)
    
    # Additional styling specific to clustered bar charts
    fig.update_layout(
        # Legend styling
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'size': ChartConfig.LEGEND_SIZE, 'color': ColorPalette.MEDIUM_TEXT}
        },
        
        # Bar mode for clustering
        barmode='group',
        bargap=0.15,      # Gap between groups
        bargroupgap=0.05, # Gap between bars in a group
        margin=dict(l=80, r=80, t=120, b=100)  # Extra top margin for legend
    )
    
    fig.show()

def clustered_bar_charts(data: pd.DataFrame, x: str, y_columns: list, title: str, labels: dict, top: int = 5):
    """
    Two clustered (grouped) bar charts side by side, sharing the y-axis, with a single legend.
    Left chart: top N sorted by first metric.
    Right chart: top N sorted by second metric.
    """
    # Always use predefined viridis colors
    colors = ColorPalette.MAIN_SEQUENCE
    width, height = ChartConfig.LARGE_WIDTH, ChartConfig.DEFAULT_HEIGHT
    
    # Prepare top N data for each metric
    top_data_total = data.sort_values(by=y_columns[0], ascending=False).head(top)
    top_data_analysed = data.sort_values(by=y_columns[1], ascending=False).head(top)
    
    # Create subplots with shared y-axis
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=[f"Top {top} by {labels.get(y_columns[0], y_columns[0])}",
                                        f"Top {top} by {labels.get(y_columns[1], y_columns[1])}"])
    
    # Add bars to left subplot (legend shown)
    for i, col in enumerate(y_columns):
        fig.add_trace(go.Bar(
            x=top_data_total[x],
            y=top_data_total[col],
            text=top_data_total[col],
            textposition='outside',
            name=labels.get(col, col),
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>{labels.get(col, col)}</b><br>%{{x}}<br>%{{y}}<extra></extra>"
        ), row=1, col=1)
    
    # Add bars to right subplot (legend hidden)
    for i, col in enumerate(y_columns):
        fig.add_trace(go.Bar(
            x=top_data_analysed[x],
            y=top_data_analysed[col],
            text=top_data_analysed[col],
            textposition='outside',
            name=labels.get(col, col),
            showlegend=False,  # Hide legend for right subplot
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>{labels.get(col, col)}</b><br>%{{x}}<br>%{{y}}<extra></extra>"
        ), row=1, col=2)
    
    # Layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': ChartConfig.TITLE_SIZE, 'family': ChartConfig.FONT_FAMILY, 'color': ColorPalette.DARK_TEXT}},
        barmode='group',  # clustered bars
        width=width,
        height=height,
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
        margin=dict(l=80, r=80, t=120, b=100),
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'center', 'x': 0.5, 'font': {'size': ChartConfig.LEGEND_SIZE, 'color': ColorPalette.MEDIUM_TEXT}}
    )
    
    # Update axes with consistent styling
    fig.update_xaxes(
        title_text=labels.get(x, x), 
        tickangle=-45,
        title_font={'size': ChartConfig.AXIS_TITLE_SIZE, 'color': ColorPalette.MEDIUM_TEXT},
        tickfont={'size': ChartConfig.TICK_SIZE, 'color': ColorPalette.LIGHT_TEXT},
        gridcolor=ColorPalette.LIGHT_GRAY,
        linecolor=ColorPalette.BORDER_GRAY,
        showgrid=True,
        gridwidth=1
    )
    fig.update_yaxes(
        title_text="Count",
        title_font={'size': ChartConfig.AXIS_TITLE_SIZE, 'color': ColorPalette.MEDIUM_TEXT},
        tickfont={'size': ChartConfig.TICK_SIZE, 'color': ColorPalette.LIGHT_TEXT},
        gridcolor=ColorPalette.LIGHT_GRAY,
        linecolor=ColorPalette.BORDER_GRAY,
        showgrid=True,
        gridwidth=1
    )
    
    fig.show()

def pie_chart(data: pd.DataFrame, names_col: str, values_col: str, title: str) -> None:
    """
    Create a pie chart with Plotly.

    Parameters:
    - data: DataFrame containing the data
    - names_col: column name for slice labels
    - values_col: column name for slice values
    - title: chart title
    - width, height: chart size
    """
    # Always use predefined extended viridis colors
    colors = ColorPalette.MAIN_SEQUENCE
    width, height = ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT

    fig = go.Figure(go.Pie(
        labels=data[names_col],
        values=data[values_col],
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        marker=dict(colors=colors, line=dict(color=ColorPalette.WHITE, width=2))
    ))
    
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': ChartConfig.TITLE_SIZE, 'family': ChartConfig.FONT_FAMILY, 'color': ColorPalette.DARK_TEXT}},
        width=width,
        height=height,
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE
    )
    
    fig.show()

def treemap_chart(data: pd.DataFrame, path_col: str, value_col: str, title: str) -> None:
    """
    Create and display a treemap using Plotly Express.

    Parameters:
    - data: DataFrame containing the data to plot
    - path_col: column name to use for category hierarchy (e.g., 'Category')
    - value_col: column name for numeric values (e.g., 'Number of Restaurants')
    - title: title of the treemap
    """
    fig = px.treemap(
        data,
        path=[path_col],
        values=value_col,
        title=title,
        color_discrete_sequence=ColorPalette.MAIN_SEQUENCE
    )
    
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': ChartConfig.TITLE_SIZE, 'family': ChartConfig.FONT_FAMILY, 'color': ColorPalette.DARK_TEXT}},
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE
    )
    
    fig.update_traces(
        textinfo='label+value',
        textfont={'size': ChartConfig.TEXT_SIZE, 'color': ColorPalette.DARK_TEXT}
    )
    fig.show()

def category_rating_analysis(dataset: pd.DataFrame, category_col: str, rating_col: str) -> pd.DataFrame:
    """
    Analyze whether certain cuisine categories tend to have higher or lower average ratings.

    Parameters:
    - dataset: DataFrame containing the data
    - category_col: column name for cuisine categories (e.g., 'categoryName')
    - rating_col: column name for ratings (e.g., 'stars')

    Returns:
    - DataFrame with mean, median, and count of ratings per category
    """

    # Compute summary statistics per category
    category_ratings = (
        dataset.groupby(category_col)[rating_col]
        .agg(['mean', 'median', 'count'])
        .reset_index()
        .sort_values(by='mean', ascending=False)
    )

    # Plot category averages (only if multiple observations exist)
    plt.figure(figsize=(10, 6))
    # Convert viridis colors to matplotlib format
    matplotlib_colors = [ColorPalette.MAIN_SEQUENCE[i % len(ColorPalette.MAIN_SEQUENCE)] for i in range(len(category_ratings))]
    sns.barplot(
        data=category_ratings[category_ratings['count'] > 5],  # filter categories with enough samples
        y=category_col,
        x='mean',
        palette=matplotlib_colors
    )
    plt.title('Average Rating per Cuisine Category', fontsize=ChartConfig.TITLE_SIZE, color=ColorPalette.DARK_TEXT)
    plt.xlabel('Average Rating', fontsize=ChartConfig.AXIS_TITLE_SIZE, color=ColorPalette.MEDIUM_TEXT)
    plt.ylabel('Category', fontsize=ChartConfig.AXIS_TITLE_SIZE, color=ColorPalette.MEDIUM_TEXT)
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()

    return category_ratings

# =============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS
# =============================================================================

def histogram_chart(data: pd.DataFrame, column: str, title: str, x_label: str, 
                   bins: int = 30, width: int = ChartConfig.DEFAULT_WIDTH, 
                   height: int = ChartConfig.DEFAULT_HEIGHT) -> None:
    """
    Create a professional histogram chart.
    
    Args:
        data: DataFrame containing the data
        column: column name to plot
        title: chart title
        x_label: x-axis label
        bins: number of bins for histogram
        width: chart width in pixels
        height: chart height in pixels
    """
    
    fig = px.histogram(
        data, 
        x=column, 
        nbins=bins,
        title=title,
        color_discrete_sequence=['#440154']  # Viridis dark purple
    )
    
    _apply_base_layout(fig, title, x_label, 'Frequency', width, height)
    
    # Style the bars
    fig.update_traces(
        marker={
            'line': {'width': 1.5, 'color': ColorPalette.MEDIUM_TEXT},
            'opacity': 0.8
        }
    )
    
    fig.show()


def scatter_plot(data: pd.DataFrame, x: str, y: str, title: str, labels: Dict[str, str],
                size: Optional[str] = None) -> None:
    """
    Create a professional scatter plot.
    
    Args:
        data: DataFrame containing the data
        x: column name for x-axis
        y: column name for y-axis
        title: chart title
        labels: dictionary for axis labels
        size: column name for bubble size (optional)
        color: column name for color mapping (optional)
        width: chart width in pixels
        height: chart height in pixels
    """
    color = ColorPalette.MAIN_SEQUENCE 
    width, height = ChartConfig.LARGE_WIDTH, ChartConfig.LARGE_HEIGHT
    
    fig = px.scatter(
        data,
        x=x,
        y=y,
        title=title,
        labels=labels,
        size=size,
        color=color,
        color_continuous_scale=color
    )
    
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y), width, height)
    
    # Style the markers
    fig.update_traces(
        marker={
            'line': {'width': 1, 'color': ColorPalette.MEDIUM_TEXT},
            'opacity': 0.7
        }
    )
    
    fig.show()


def box_plot(data: pd.DataFrame, x: str, y: str, title: str, labels: Dict[str, str]) -> None:
    """
    Create a professional box plot.
    
    Args:
        data: DataFrame containing the data
        x: column name for categories (x-axis)
        y: column name for values (y-axis)
        title: chart title
        labels: dictionary for axis labels
        width: chart width in pixels
        height: chart height in pixels
    """
    width, height = ChartConfig.LARGE_WIDTH, ChartConfig.DEFAULT_HEIGHT
    
    fig = px.box(
        data,
        x=x,
        y=y,
        title=title,
        labels=labels,
        color_discrete_sequence=ColorPalette.MAIN_SEQUENCE
    )
    
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y), width, height)
    
    fig.show()
