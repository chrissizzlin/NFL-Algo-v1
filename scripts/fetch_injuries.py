import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import numpy as np

def fetch_current_injuries():
    """
    Fetches the latest NFL injury reports with detailed information.
    """
    try:
        # Import the most recent injury data
        injuries = nfl.import_injuries([datetime.now().year])
        
        # Get the most recent week
        current_week = injuries['week'].max()
        
        # Filter for current week only
        injuries = injuries[injuries['week'] == current_week]
        
        # Map injury status to severity score (aligned with 0-10 scale)
        severity_map = {
            'Out': 10.0,
            'Doubtful': 8.0,
            'Questionable': 5.0,
            'Limited Participation in Practice': 3.0,
            'Full Participation in Practice': 1.0,
            'Did Not Participate In Practice': 7.0
        }
        
        # Use report_status for status, fall back to practice_status if not available
        injuries['status'] = injuries['report_status'].fillna(injuries['practice_status'])
        injuries['severity_score'] = injuries['status'].map(severity_map).fillna(1.0)
        
        # Combine primary and secondary injuries if they exist
        injuries['injury_type'] = injuries.apply(
            lambda x: str(x['report_primary_injury']) if pd.isna(x['report_secondary_injury']) 
            else f"{x['report_primary_injury']}, {x['report_secondary_injury']}", 
            axis=1
        )
        
        # Clean and organize the data
        injuries = injuries[[
            'team', 'week', 'position', 
            'full_name', 'status', 'severity_score',
            'injury_type', 'practice_status'
        ]].copy()
        
        # Add starter status with position importance
        def determine_starter_status(position):
            key_positions = {
                'QB': '⭐⭐⭐',  # Triple star for QB
                'RB1': '⭐⭐', 'WR1': '⭐⭐', 'TE1': '⭐⭐',  # Double star for key offensive positions
                'DE1': '⭐⭐', 'MLB': '⭐⭐', 'CB1': '⭐⭐',  # Double star for key defensive positions
                'LT': '⭐', 'RT': '⭐',  # Single star for other important positions
                'C': '⭐', 'G': '⭐', 'DT': '⭐', 'LB': '⭐', 'S': '⭐'  # Single star for other starters
            }
            pos = position.split('-')[0] if '-' in position else position
            return key_positions.get(pos, '📋')
        
        injuries['starter_status'] = injuries['position'].apply(determine_starter_status)
        
        # Filter out non-impactful statuses and injuries
        injuries = injuries[
            (injuries['status'].isin(['Out', 'Doubtful', 'Questionable', 'Did Not Participate In Practice'])) &
            (~injuries['injury_type'].isin(['None', 'none', None, np.nan]))
        ]
        
        # Remove duplicates keeping most severe status
        injuries = injuries.sort_values(
            'severity_score', ascending=False
        ).drop_duplicates(subset=['full_name', 'team'], keep='first')
        
        return injuries
    except Exception as e:
        print(f"Error fetching injuries: {str(e)}")
        return pd.DataFrame()

def filter_injuries_by_teams(injuries, teams):
    """
    Filters injury reports for the specified teams.
    """
    if injuries.empty:
        return pd.DataFrame()
    
    # Filter for specified teams
    team_injuries = injuries[injuries['team'].isin(teams)].copy()
    
    return team_injuries

if __name__ == "__main__":
    # Example usage
    selected_teams = ['NE', 'BUF']  # Replace with user-selected teams
    current_injuries = fetch_current_injuries()
    relevant_injuries = filter_injuries_by_teams(current_injuries, selected_teams)
    relevant_injuries.to_csv('data/relevant_injuries.csv', index=False)
    print("Relevant injury reports have been fetched and saved.")
