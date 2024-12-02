import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
import numpy as np

def calculate_team_metrics(team_stats):
    """Calculate advanced metrics from basic stats."""
    metrics = {}
    
    # Get current week from the schedule or use a fixed number
    current_week = 13  # We can update this dynamically later
    
    # Offensive Efficiency
    metrics['yards_per_game'] = team_stats['total_yards'].iloc[0] / current_week
    metrics['pass_ratio'] = team_stats['passing_yards'].iloc[0] / team_stats['total_yards'].iloc[0]
    metrics['completion_rate'] = (team_stats['completions'].iloc[0] / team_stats['attempts'].iloc[0]) * 100
    
    # Ball Security
    metrics['turnover_ratio'] = team_stats['turnovers'].iloc[0] / current_week
    
    return metrics

def create_radar_chart(home_metrics, away_metrics, home_team, away_team):
    """Create a radar chart comparing team metrics."""
    categories = ['Yards/Game', 'Pass Ratio', 'Completion %', 'Ball Security']
    
    fig = go.Figure()
    
    # Home team trace
    fig.add_trace(go.Scatterpolar(
        r=[home_metrics['yards_per_game']/10, 
           home_metrics['pass_ratio']*100,
           home_metrics['completion_rate'],
           (1 - home_metrics['turnover_ratio'])*100],
        theta=categories,
        fill='toself',
        name=home_team,
        line=dict(color='#3b82f6'),  # Bright blue
        fillcolor='rgba(59, 130, 246, 0.3)'  # Transparent blue
    ))
    
    # Away team trace
    fig.add_trace(go.Scatterpolar(
        r=[away_metrics['yards_per_game']/10,
           away_metrics['pass_ratio']*100,
           away_metrics['completion_rate'],
           (1 - away_metrics['turnover_ratio'])*100],
        theta=categories,
        fill='toself',
        name=away_team,
        line=dict(color='#ef4444'),  # Bright red
        fillcolor='rgba(239, 68, 68, 0.3)'  # Transparent red
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255, 255, 255, 0.2)',  # Subtle grid
                linecolor='rgba(255, 255, 255, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'  # Transparent background
        ),
        showlegend=True,
        title="Team Performance Comparison",
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color='white'),  # White text
        title_font_color='white'
    )
    
    return fig

def create_head_to_head_bars(metric_name, home_value, away_value, home_team, away_team):
    """Create a horizontal bar chart for head-to-head metric comparison."""
    max_val = max(home_value, away_value)
    home_pct = (home_value / max_val) * 100
    away_pct = (away_value / max_val) * 100
    
    fig = go.Figure()
    
    # Home team bar
    fig.add_trace(go.Bar(
        y=[home_team],
        x=[home_pct],
        orientation='h',
        name=home_team,
        text=f"{home_value:.1f}",
        textposition='auto',
        marker_color='#3b82f6',  # Bright blue
        textfont=dict(color='white')
    ))
    
    # Away team bar
    fig.add_trace(go.Bar(
        y=[away_team],
        x=[away_pct],
        orientation='h',
        name=away_team,
        text=f"{away_value:.1f}",
        textposition='auto',
        marker_color='#ef4444',  # Bright red
        textfont=dict(color='white')
    ))
    
    fig.update_layout(
        title=metric_name,
        barmode='group',
        height=150,
        showlegend=False,
        xaxis_range=[0, 100],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        )
    )
    
    return fig 