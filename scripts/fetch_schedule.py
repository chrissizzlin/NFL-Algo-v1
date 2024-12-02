import nfl_data_py as nfl
import pandas as pd
from datetime import datetime, timedelta

def get_current_season_and_week():
    """
    Gets the current NFL season and week.
    """
    # Get current date
    today = datetime.now()
    
    # NFL season typically starts in September
    current_year = today.year
    season_start = datetime(today.year, 9, 1)
    
    # If we're before September, we're in the previous season
    if today < season_start:
        current_year -= 1
    
    # Get schedule for the current year
    schedule = nfl.import_schedules([current_year])
    
    # Convert gameday to datetime
    schedule['gameday'] = pd.to_datetime(schedule['gameday'])
    
    # Get the current week's games
    current_week = schedule[schedule['gameday'] > (today - timedelta(days=7))]['week'].min()
    
    return current_year, current_week

def get_upcoming_games():
    """
    Gets this week's and next week's NFL games.
    """
    current_year, current_week = get_current_season_and_week()
    
    # Get schedule for current season
    schedule = nfl.import_schedules([current_year])
    
    # Convert gameday to datetime
    schedule['gameday'] = pd.to_datetime(schedule['gameday'])
    
    # Get start of today (midnight)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Filter for current and next week's games, including today's games
    upcoming_games = schedule[
        (schedule['week'].isin([current_week, current_week + 1])) &
        (schedule['gameday'] >= today_start)
    ].copy()
    
    # Format games as tuples for the dropdown
    games_list = list(zip(upcoming_games['away_team'], upcoming_games['home_team']))
    
    # Create game labels with dates and times
    game_labels = {}
    for idx, row in upcoming_games.iterrows():
        game_key = (row['away_team'], row['home_team'])
        game_date = row['gameday'].strftime('%A, %B %d')
        game_time = row['gameday'].strftime('%I:%M %p ET')
        game_labels[game_key] = f"{row['away_team']} @ {row['home_team']} ({game_date} at {game_time})"
    
    return games_list, game_labels, current_year, current_week 