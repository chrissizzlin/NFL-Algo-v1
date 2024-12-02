import streamlit as st # type: ignore
import pandas as pd
import nfl_data_py as nfl
from scripts.fetch_stats import fetch_team_stats, fetch_team_records
from scripts.fetch_injuries import fetch_current_injuries, filter_injuries_by_teams
from scripts.calculate_metrics import calculate_metrics, predict_outcome
from scripts.fetch_schedule import get_upcoming_games
from scripts.metrics_helper import (calculate_team_metrics, create_radar_chart, 
                                  create_head_to_head_bars)
import plotly.graph_objects as go # type: ignore
import numpy as np
from datetime import datetime

def display_game_details(home_team, away_team, game_info):
    """Display win probabilities and moneyline odds for both teams."""
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Format team records
    home_record = f"{game_info.get('home_wins', 0)}-{game_info.get('home_losses', 0)}"
    away_record = f"{game_info.get('away_wins', 0)}-{game_info.get('away_losses', 0)}"
    
    # Get moneyline odds
    home_ml = game_info.get('home_moneyline', 'N/A')
    away_ml = game_info.get('away_moneyline', 'N/A')
    
    with col1:
        st.markdown(f"""
            <div style='background-color: rgba(71, 85, 105, 0.95); padding: 20px; border-radius: 12px; text-align: center;
                      border: 1px solid rgba(148, 163, 184, 0.4); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 24px; font-weight: bold; color: #f8fafc; margin-bottom: 5px;'>{home_team}</div>
                <div style='color: #e2e8f0; font-size: 16px; margin-bottom: 10px;'>Record: {home_record}</div>
                <div style='font-size: 42px; font-weight: bold; color: #f8fafc;'>{game_info.get('home_win_prob', 0)*100:.1f}%</div>
                <div style='color: #e2e8f0; font-size: 18px;'>ML: {home_ml}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background-color: rgba(71, 85, 105, 0.95); padding: 20px; border-radius: 12px; text-align: center;
                      border: 1px solid rgba(148, 163, 184, 0.4); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 24px; font-weight: bold; color: #f8fafc; margin-bottom: 5px;'>{away_team}</div>
                <div style='color: #e2e8f0; font-size: 16px; margin-bottom: 10px;'>Record: {away_record}</div>
                <div style='font-size: 42px; font-weight: bold; color: #f8fafc;'>{game_info.get('away_win_prob', 0)*100:.1f}%</div>
                <div style='color: #e2e8f0; font-size: 18px;'>ML: {away_ml}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Display betting analysis
    display_betting_analysis(game_info)

def create_head_to_head_bars(metric_name, home_value, away_value, home_team, away_team):
    """Create a horizontal bar chart for head-to-head metric comparison."""
    max_val = max(home_value, away_value)
    home_pct = (home_value / max_val) * 100
    away_pct = (away_value / max_val) * 100
    
    fig = go.Figure()
    
    # Home team bar (now green)
    fig.add_trace(go.Bar(
        y=[home_team],
        x=[home_pct],
        orientation='h',
        name=home_team,
        text=f"{home_value:.1f}",
        textposition='auto',
        marker_color='#22c55e',  # Bright green
        textfont=dict(color='white')
    ))
    
    # Away team bar (brighter red)
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
        height=80,  # Reduced height
        showlegend=False,
        xaxis_range=[0, 100],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def display_team_stats(home_team, away_team, home_stats, away_stats):
    """Display team statistics comparison using charts and metrics."""
    
    # Calculate advanced metrics for both teams
    home_metrics = calculate_team_metrics(home_stats)
    away_metrics = calculate_team_metrics(away_stats)
    
    # Create columns for key stats comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Yards per game comparison
        yards_fig = create_head_to_head_bars(
            "Yards per Game",
            home_stats['yards_per_game'].iloc[0],
            away_stats['yards_per_game'].iloc[0],
            home_team,
            away_team
        )
        st.plotly_chart(yards_fig, use_container_width=True)
    
    with col2:
        # Points per game comparison
        points_fig = create_head_to_head_bars(
            "Points per Game",
            home_stats['points_per_game'].iloc[0],
            away_stats['points_per_game'].iloc[0],
            home_team,
            away_team
        )
        st.plotly_chart(points_fig, use_container_width=True)
    
    with col3:
        # Completion rate comparison
        comp_fig = create_head_to_head_bars(
            "Completion Rate (%)",
            home_stats['completion_rate'].iloc[0],
            away_stats['completion_rate'].iloc[0],
            home_team,
            away_team
        )
        st.plotly_chart(comp_fig, use_container_width=True)
    
    # Create and display radar chart
    radar_fig = create_radar_chart(home_metrics, away_metrics, home_team, away_team)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Detailed metric comparison section
    st.markdown("### Detailed Team Metrics")
    col1, col2 = st.columns(2)
    
    # Define metrics to compare with their properties
    metrics_to_compare = {
        'Offensive Metrics': {
            'Total Yards': ('total_yards', True),
            'Passing Yards': ('passing_yards', True),
            'Rushing Yards': ('rushing_yards', True),
            'Points per Game': ('points_per_game', True),
            'Yards per Game': ('yards_per_game', True),
            'Completion Rate': ('completion_rate', True),
            'First Downs': ('total_first_downs', True)
        },
        'Defensive Metrics': {
            'Sacks Taken': ('sacks', False),
            'Turnovers': ('turnovers', False),
            'Points Allowed': ('points_allowed_per_game', False),
            'Yards Allowed': ('yards_allowed_per_game', False)
        }
    }
    
    # Helper function to create metric box
    def create_metric_box(label, home_val, away_val, higher_is_better=True):
        # Determine which value is better
        is_home_better = home_val > away_val if higher_is_better else home_val < away_val
        
        # Set background colors - brighter green and red
        home_bg = 'rgba(34, 197, 94, 0.3)' if is_home_better else 'rgba(239, 68, 68, 0.3)'  # Increased opacity
        away_bg = 'rgba(34, 197, 94, 0.3)' if not is_home_better else 'rgba(239, 68, 68, 0.3)'  # Increased opacity
        
        # Format values
        if isinstance(home_val, float):
            home_val = f"{home_val:.1f}"
        if isinstance(away_val, float):
            away_val = f"{away_val:.1f}"
        
        return f"""
            <div style="margin-bottom: 10px;">
                <div style="font-weight: bold; color: #e2e8f0; margin-bottom: 5px;">{label}</div>
                <div style="display: flex; justify-content: space-between; gap: 10px;">
                    <div style="flex: 1; background: {home_bg}; padding: 8px; border-radius: 4px; text-align: center;">
                        {home_val}
                    </div>
                    <div style="flex: 1; background: {away_bg}; padding: 8px; border-radius: 4px; text-align: center;">
                        {away_val}
                    </div>
                </div>
            </div>
        """
    
    # Display metrics
    with col1:
        st.markdown(f"#### {home_team}")
    with col2:
        st.markdown(f"#### {away_team}")
    
    for category, metrics in metrics_to_compare.items():
        st.markdown(f"#### {category}")
        metric_html = ""
        for label, (metric, higher_is_better) in metrics.items():
            if metric in home_stats.columns and metric in away_stats.columns:
                home_val = home_stats[metric].iloc[0]
                away_val = away_stats[metric].iloc[0]
                metric_html += create_metric_box(label, home_val, away_val, higher_is_better)
        st.markdown(metric_html, unsafe_allow_html=True)

def display_injuries(home_team, away_team, home_injuries, away_injuries):
    """Display injury reports for both teams."""
    def filter_significant_injuries(injuries_df):
        """Filter for only significant injuries."""
        if injuries_df.empty:
            return injuries_df
            
        # Filter out full participation and non-injuries
        return injuries_df[
            (injuries_df['status'].isin(['Out', 'Doubtful', 'Questionable', 'Did Not Participate In Practice'])) &
            (~injuries_df['injury_type'].isin(['None', 'none', None, np.nan]))
        ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {away_team}")
        filtered_injuries = filter_significant_injuries(away_injuries)
        if filtered_injuries.empty:
            st.markdown("No significant injuries reported")
        else:
            # Sort injuries by severity and starter status
            sorted_injuries = filtered_injuries.sort_values(
                by=['severity_score', 'starter_status'],
                ascending=[False, True]
            )
            
            for _, injury in sorted_injuries.iterrows():
                status_color = {
                    'Out': '#ef4444',
                    'Doubtful': '#f97316',
                    'Questionable': '#eab308',
                    'Did Not Participate In Practice': '#f97316'
                }.get(injury['status'], '#94a3b8')
                
                st.markdown(
                    f"""<div style='background-color: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: {status_color}; font-weight: bold;'>{injury['starter_status']}</span>
                                <span style='color: #e2e8f0; font-weight: bold;'> {injury['full_name']}</span>
                                <span style='color: #94a3b8;'> ({injury['position']})</span>
                            </div>
                            <span style='color: {status_color}; font-weight: bold;'>{injury['status']}</span>
                        </div>
                        <div style='color: #94a3b8; margin-top: 4px;'>
                            {injury['injury_type']}
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )
    
    with col2:
        st.markdown(f"#### {home_team}")
        filtered_injuries = filter_significant_injuries(home_injuries)
        if filtered_injuries.empty:
            st.markdown("No significant injuries reported")
        else:
            # Sort injuries by severity and starter status
            sorted_injuries = filtered_injuries.sort_values(
                by=['severity_score', 'starter_status'],
                ascending=[False, True]
            )
            
            for _, injury in sorted_injuries.iterrows():
                status_color = {
                    'Out': '#ef4444',
                    'Doubtful': '#f97316',
                    'Questionable': '#eab308',
                    'Did Not Participate In Practice': '#f97316'
                }.get(injury['status'], '#94a3b8')
                
                st.markdown(
                    f"""<div style='background-color: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: {status_color}; font-weight: bold;'>{injury['starter_status']}</span>
                                <span style='color: #e2e8f0; font-weight: bold;'> {injury['full_name']}</span>
                                <span style='color: #94a3b8;'> ({injury['position']})</span>
                            </div>
                            <span style='color: {status_color}; font-weight: bold;'>{injury['status']}</span>
                        </div>
                        <div style='color: #94a3b8; margin-top: 4px;'>
                            {injury['injury_type']}
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )
    
    # Add explanation of injury statuses
    with st.expander("Understanding Injury Reports"):
        st.markdown("""
            ### How to Read Injury Reports
            
            **Player Status Indicators:**
            - ⭐⭐⭐ : Key starter (e.g., QB1)
            - ⭐⭐ : Important starter
            - ⭐ : Starter
            - 📋 : Backup
            
            **Injury Status Colors:**
            - 🔴 Red: Out
            - 🟡 Orange: Doubtful/DNP
            - 🟡 Yellow: Questionable
            
            The injury report shows only significant injuries that may impact team performance.
        """)

def filter_injuries_by_teams(injuries_df, teams):
    """Filter injuries for specific teams."""
    if injuries_df.empty:
        return pd.DataFrame()
    return injuries_df[injuries_df['team'].isin(teams)].copy()

def main():
    # Set page config
    st.set_page_config(
        page_title="NFL Game Predictor",
        page_icon="🏈",
        layout="wide"
    )

    # Custom CSS for better visibility
    st.markdown("""
        <style>
        /* Overall page styling */
        .stApp {
            background-color: #0f172a;
        }
        
        /* Main title styling */
        .title-container {
            background-color: rgba(71, 85, 105, 0.95);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(148, 163, 184, 0.5);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .main-title {
            color: #60a5fa;
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
        }
        
        .season-info {
            color: #f8fafc;
            font-size: 24px;
            font-weight: 500;
        }
        
        /* Game selector styling */
        .stSelectbox > div > div {
            background-color: rgba(71, 85, 105, 0.95) !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Selector text color */
        .stSelectbox > div > div > div {
            color: #f8fafc !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background-color: rgba(71, 85, 105, 0.95) !important;
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #4ade80, #16a34a) !important;  /* Even brighter green */
        }
        
        /* For red progress bars */
        .negative-progress > div > div > div > div {
            background: linear-gradient(90deg, #f87171, #dc2626) !important;  /* Even brighter red */
        }
        
        /* Button styling */
        .stButton > button {
            background: #2563eb !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
        }
        
        .stButton > button:hover {
            background: #1d4ed8 !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3) !important;
        }
        
        /* Section headers */
        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 600 !important;
            margin: 20px 0 !important;
        }
        
        /* Stats boxes */
        div[data-testid="stMetricValue"] {
            background-color: rgba(71, 85, 105, 0.95) !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
            border-radius: 8px !important;
            padding: 16px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Team comparison boxes */
        .team-box {
            background-color: rgba(71, 85, 105, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.5);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Betting analysis boxes */
        .betting-box {
            background-color: rgba(71, 85, 105, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.5);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Enhance text contrast */
        .stMarkdown {
            color: #f8fafc !important;
        }
        
        /* Metric labels */
        .metric-label {
            color: #f8fafc;
            font-weight: 500;
        }
        
        /* Text field styling */
        .stTextInput > div > div > input {
            color: #f8fafc !important;
            background-color: rgba(71, 85, 105, 0.95) !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
        }
        
        /* Dropdown text and options */
        .stSelectbox > div > div > div[data-baseweb="select"] > div {
            color: #f8fafc !important;
        }
        
        .stSelectbox > div > div > div[data-baseweb="select"] > div:hover {
            background-color: rgba(100, 116, 139, 0.95) !important;
        }
        
        .stSelectbox > div > div > div[data-baseweb="select"] > div:focus {
            background-color: rgba(100, 116, 139, 0.95) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display title and season info in a container
    st.markdown("""
        <div class="title-container">
            <div class="main-title">NFL Game Predictor</div>
            <div class="season-info">Week 13 - 2024 Season</div>
        </div>
    """, unsafe_allow_html=True)
    
    games_list, game_labels, current_season, current_week = get_upcoming_games()
    st.markdown(f"### Week {current_week} - {current_season} Season")
    
    selected_game = st.selectbox(
        "Choose a game:", 
        options=games_list,
        format_func=lambda x: game_labels[x]
    )

    if st.button("Analyze Matchup"):
        away_team, home_team = selected_game
        
        try:
            # Fetch team stats
            team_stats = fetch_team_stats([current_season])
            home_stats = team_stats[team_stats['team'] == home_team]
            away_stats = team_stats[team_stats['team'] == away_team]
            
            if home_stats.empty or away_stats.empty:
                st.error("Stats not available for one or both teams")
                return

            # Get game details
            schedule = nfl.import_schedules([current_season])
            game_info = schedule[
                (schedule['home_team'] == home_team) & 
                (schedule['away_team'] == away_team) &
                (schedule['week'].isin([current_week, current_week + 1]))
            ].iloc[0].to_dict()

            # Get injuries
            all_injuries = fetch_current_injuries()
            home_injuries = filter_injuries_by_teams(all_injuries, [home_team])
            away_injuries = filter_injuries_by_teams(all_injuries, [away_team])

            # Calculate prediction
            metrics = calculate_metrics(home_stats, away_stats, home_injuries, away_injuries)
            prediction = predict_outcome(metrics)
            
            if not prediction.empty:
                # Get win probabilities from the prediction
                home_team_data = prediction[prediction['team'] == home_team]
                away_team_data = prediction[prediction['team'] == away_team]
                
                if not home_team_data.empty and not away_team_data.empty:
                    game_info['home_win_prob'] = home_team_data['win_probability'].iloc[0] / 100
                    game_info['away_win_prob'] = away_team_data['win_probability'].iloc[0] / 100
            
            # Get team records
            team_records = fetch_team_records([current_season])
            if team_records:
                game_info['home_wins'] = team_records.get(home_team, {}).get('wins', 0)
                game_info['home_losses'] = team_records.get(home_team, {}).get('losses', 0)
                game_info['away_wins'] = team_records.get(away_team, {}).get('wins', 0)
                game_info['away_losses'] = team_records.get(away_team, {}).get('losses', 0)
            
            # Display game details
            display_game_details(home_team, away_team, game_info)
            
            # Display team stats comparison
            st.markdown("### Team Statistics")
            display_team_stats(home_team, away_team, home_stats, away_stats)
            
            # Display injury reports
            st.markdown("### Injury Reports")
            display_injuries(home_team, away_team, home_injuries, away_injuries)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise

def display_prediction(home_team, away_team, prediction):
    """
    Display game details including predictions, records, and betting information.
    """
    try:
        # Get current year
        current_year = datetime.now().year
        if datetime.now().month < 9:
            current_year -= 1
            
        # Get schedule data for additional game details
        schedule_data = nfl.import_schedules([current_year])
        game_details = schedule_data[
            (schedule_data['home_team'] == home_team) & 
            (schedule_data['away_team'] == away_team)
        ].iloc[0] if not schedule_data.empty else None
        
        # Get the win probabilities
        if 'win_probability' not in prediction.columns:
            st.error("Prediction data is missing win probability information")
            return
            
        home_team_data = prediction[prediction['team'] == home_team]
        away_team_data = prediction[prediction['team'] == away_team]
        
        if home_team_data.empty or away_team_data.empty:
            st.error(f"Missing prediction data for {home_team if home_team_data.empty else away_team}")
            return
            
        # Get raw probabilities and ensure they're between 0 and 1
        home_prob = float(home_team_data['win_probability'].iloc[0])
        away_prob = float(away_team_data['win_probability'].iloc[0])
        
        # Normalize probabilities if they're not already between 0 and 1
        if home_prob > 1 or away_prob > 1:
            total = home_prob + away_prob
            home_prob = (home_prob / total) * 100
            away_prob = (away_prob / total) * 100
        else:
            home_prob *= 100
            away_prob *= 100
        
        # Get team records
        try:
            team_records = fetch_team_records([current_year])
        except Exception as e:
            print(f"Error fetching team records: {str(e)}")
            team_records = {}
            st.warning("Unable to fetch team records. Displaying predictions only.")
        
        # Get records for both teams
        home_record = team_records.get(home_team, {'wins': 0, 'losses': 0, 'streak': 'N/A'})
        away_record = team_records.get(away_team, {'wins': 0, 'losses': 0, 'streak': 'N/A'})
        
        # Display header
        st.markdown("## Game Details")
        
        # Create three columns for team comparison and game details
        col1, col2, col3 = st.columns([4, 4, 3])
        
        # Style for the containers
        st.markdown("""
        <style>
        .team-container {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin: 10px;
        }
        .details-container {
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin: 10px;
        }
        .team-name {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
        }
        .win-prob {
            font-size: 42px;
            font-weight: bold;
            color: #ffffff;
            margin: 15px 0;
        }
        .team-record {
            font-size: 18px;
            color: #ffffff;
            margin: 5px 0;
        }
        .streak {
            font-size: 16px;
            color: #ffffff;
            margin: 5px 0;
            padding-bottom: 10px;
        }
        .detail-label {
            color: #9ca3af;
            font-size: 14px;
            margin-bottom: 2px;
        }
        .detail-value {
            color: #ffffff;
            font-size: 16px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with col1:
            st.markdown(f"""
                <div class="team-container">
                    <div class="team-name">{away_team}</div>
                    <div class="win-prob">{away_prob:.1f}%</div>
                    <div class="team-record">Record: {away_record['wins']}-{away_record['losses']}</div>
                    <div class="streak">Streak: {away_record['streak']}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="team-container">
                    <div class="team-name">{home_team}</div>
                    <div class="win-prob">{home_prob:.1f}%</div>
                    <div class="team-record">Record: {home_record['wins']}-{home_record['losses']}</div>
                    <div class="streak">Streak: {home_record['streak']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if game_details is not None:
                st.markdown(f"""
                    <div class="details-container">
                        <div class="detail-label">Game Time</div>
                        <div class="detail-value">{game_details['gameday']} {game_details['gametime']}</div>
                        
                        <div class="detail-label">Location</div>
                        <div class="detail-value">{game_details['stadium']}</div>
                        
                        <div class="detail-label">Weather</div>
                        <div class="detail-value">
                            {f"{game_details['temp']}°F, " if pd.notna(game_details['temp']) else ""}
                            {f"Wind: {game_details['wind']} mph" if pd.notna(game_details['wind']) else ""}
                            ({game_details['roof']})
                        </div>
                        
                        <div class="detail-label">Surface</div>
                        <div class="detail-value">{game_details['surface']}</div>
                        
                        <div class="detail-label">Quarterbacks</div>
                        <div class="detail-value">
                            {away_team}: {game_details['away_qb_name']}<br>
                            {home_team}: {game_details['home_qb_name']}
                        </div>
                        
                        <div class="detail-label">Betting Lines</div>
                        <div class="detail-value">
                            Spread: {home_team} {game_details['spread_line']}<br>
                            O/U: {game_details['total_line']}<br>
                            ML: {away_team} {game_details['away_moneyline']} / {home_team} {game_details['home_moneyline']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying game details: {str(e)}")
        print(f"Detailed error in display_prediction: {str(e)}")
        print(f"Error type: {type(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("Traceback:")
            traceback.print_tb(e.__traceback__)

def load_css():
    """Load all CSS styles"""
    st.markdown("""
        <style>
        /* Base styles */
        .stApp {
            background: linear-gradient(180deg, #0e1117 0%, #151922 100%);
        }
        
        /* Prediction container styles */
        .prediction-container {
            background: rgba(31, 41, 55, 0.7);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .prediction-title {
            color: #ffffff;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .prediction-winner {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }
        
        .prediction-probability {
            font-size: 1.1rem;
            color: #9ca3af;
            margin-bottom: 2rem;
        }
        
        /* Breakdown section styles */
        .breakdown-container {
            background: rgba(17, 24, 39, 0.4);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .breakdown-title {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #ffffff;
            font-weight: 500;
        }
        
        .breakdown-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 0.5rem;
        }
        
        .breakdown-table th {
            padding: 0.75rem;
            color: #9ca3af;
            font-weight: 500;
            text-align: left;
        }
        
        .breakdown-table td {
            padding: 0.75rem;
            color: #ffffff;
        }
        
        .breakdown-row {
            background: rgba(31, 41, 55, 0.3);
            border-radius: 6px;
        }
        
        .breakdown-total {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 6px;
            font-weight: 600;
        }
        
        /* Explanation section styles */
        .explanation-container {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(17, 24, 39, 0.4);
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .explanation-title {
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #ffffff;
        }
        
        .explanation-text {
            color: #9ca3af;
        }
        
        /* Additional styles for better visibility */
        .stMarkdown {
            color: #ffffff !important;
        }
        
        div[data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }
        
        /* Button and selectbox styling */
        .stButton > button {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div {
            background-color: rgba(31, 41, 55, 0.7) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

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
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color='white'),  # White text
        title_font_color='white'
    )
    
    return fig

def display_betting_analysis(game_info):
    """Display betting analysis including edge calculation and recommendations."""
    st.markdown("### Betting Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Get betting values
    home_ml = game_info.get('home_moneyline', 0)
    away_ml = game_info.get('away_moneyline', 0)
    home_prob = game_info.get('home_win_prob', 0)
    away_prob = game_info.get('away_win_prob', 0)
    
    # Calculate betting edge
    def calculate_edge(prob, odds):
        if odds > 0:
            implied_prob = 100 / (odds + 100)
        else:
            implied_prob = abs(odds) / (abs(odds) + 100)
        return (prob - implied_prob) * 100
    
    home_edge = calculate_edge(home_prob, home_ml)
    away_edge = calculate_edge(away_prob, away_ml)
    
    # Function to determine edge color and label
    def get_edge_color_and_label(edge):
        if edge > 3:
            return "#22c55e", "GREAT"  # Green
        elif edge > 2:
            return "#3b82f6", "GOOD"   # Blue
        elif edge > 1.01:
            return "#eab308", "OK"     # Yellow
        else:
            return "#ef4444", "NO VALUE"  # Red
    
    home_color, home_label = get_edge_color_and_label(home_edge)
    away_color, away_label = get_edge_color_and_label(away_edge)
    
    # Display color key
    st.markdown("""
        <div style='display: flex; justify-content: center; gap: 20px; margin-bottom: 15px; background-color: rgba(17, 25, 40, 0.75); padding: 10px; border-radius: 8px;'>
            <div style='text-align: center;'>
                <span style='color: #eab308; font-weight: bold;'>⬤</span>
                <span style='color: #e2e8f0;'> OK (>1.01%)</span>
            </div>
            <div style='text-align: center;'>
                <span style='color: #3b82f6; font-weight: bold;'>⬤</span>
                <span style='color: #e2e8f0;'> GOOD (>2%)</span>
            </div>
            <div style='text-align: center;'>
                <span style='color: #22c55e; font-weight: bold;'>⬤</span>
                <span style='color: #e2e8f0;'> GREAT (>3%)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with col1:
        st.markdown(f"""
            <div style='background-color: rgba(71, 85, 105, 0.95); padding: 20px; border-radius: 12px; text-align: center;
                      border: 1px solid rgba(148, 163, 184, 0.4); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 18px; color: #f8fafc; margin-bottom: 10px;'>Home Team Edge</div>
                <div style='font-size: 24px; font-weight: bold; color: {home_color};'>{home_edge:+.1f}%</div>
                <div style='color: #e2e8f0; font-size: 14px;'>Rating: {home_label}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background-color: rgba(71, 85, 105, 0.95); padding: 20px; border-radius: 12px; text-align: center;
                      border: 1px solid rgba(148, 163, 184, 0.4); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-size: 18px; color: #f8fafc; margin-bottom: 10px;'>Away Team Edge</div>
                <div style='font-size: 24px; font-weight: bold; color: {away_color};'>{away_edge:+.1f}%</div>
                <div style='color: #e2e8f0; font-size: 14px;'>Rating: {away_label}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Add betting recommendations if there's value
    if max(home_edge, away_edge) > 1.01:
        best_bet = "Home Team" if home_edge > away_edge else "Away Team"
        edge = max(home_edge, away_edge)
        edge_color, edge_label = get_edge_color_and_label(edge)
        
        st.markdown(f"""
            <div style='background-color: rgba(71, 85, 105, 0.95); padding: 15px; border-radius: 8px; margin-top: 20px;
                      border: 1px solid rgba(148, 163, 184, 0.4); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-weight: bold; color: {edge_color}; margin-bottom: 5px;'>Recommended Bet - {edge_label}</div>
                <div style='color: #f8fafc;'>
                    {best_bet} shows {edge:+.1f}% edge
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: rgba(71, 85, 105, 0.95); padding: 15px; border-radius: 8px; margin-top: 20px;
                      border: 1px solid rgba(148, 163, 184, 0.4); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <div style='font-weight: bold; color: #ef4444; margin-bottom: 5px;'>No Recommended Bets</div>
                <div style='color: #f8fafc;'>
                    Neither team shows sufficient betting value (minimum 1.01% edge required)
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
