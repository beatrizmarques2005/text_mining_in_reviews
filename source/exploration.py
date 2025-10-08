
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

########################
#### VISUALIZATIONS ####
########################

def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, labels: dict, color_palette: str ="viridis", width: int =1000, height: int =600) -> None:
    """
    Create a beautiful and professional bar chart with enhanced styling.
    
    Parameters:
    - data: DataFrame containing the data
    - x: column name for x-axis
    - y: column name for y-axis  
    - title: chart title
    - labels: dictionary for axis labels
    - color_palette: color scheme (default: viridis)
    - width: chart width in pixels
    - height: chart height in pixels
    """
    
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
    
    # Enhanced layout and styling
    fig.update_layout(
        # Title styling
        title={
            'text': title,
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Arial Black', 'color': '#2c3e50'}
        },
        
        # Axis styling
        xaxis={
            'title': {'text': labels.get(x, x), 'font': {'size': 16, 'color': '#34495e'}},
            'tickangle': -45,  # Rotate x-axis labels for better readability
            'tickfont': {'size': 12, 'color': '#7f8c8d'},
            'gridcolor': '#ecf0f1',
            'linecolor': '#bdc3c7'
        },
        
        yaxis={
            'title': {'text': labels.get(y, y), 'font': {'size': 16, 'color': '#34495e'}},
            'tickfont': {'size': 12, 'color': '#7f8c8d'},
            'gridcolor': '#ecf0f1',
            'linecolor': '#bdc3c7'
        },
        
        # Overall layout
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height,
        margin=dict(l=80, r=80, t=100, b=100),
        
        # Remove color bar legend for cleaner look
        showlegend=False,
        coloraxis_showscale=False
    )
    
    # Style the bars
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        textfont={'size': 11, 'color': '#2c3e50'},
        marker={
            'line': {'width': 1.5, 'color': '#34495e'},  # Border around bars
            'opacity': 0.8
        },
        hovertemplate='<b>%{x}</b><br>%{y}<br><extra></extra>'  # Custom hover info
    )
    
    # Add subtle grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    
    fig.show()