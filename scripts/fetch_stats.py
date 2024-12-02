import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

def fetch_team_stats(years):
    """
    Fetches team statistics for the specified years.
    """
    try:
        # Import weekly data up to current week
        weekly_data = nfl.import_weekly_data(years)
        current_week = weekly_data['week'].max()
        
        # Filter for completed games only
        weekly_data = weekly_data[weekly_data['week'] < current_week]
        
        # Group by team and season with available columns
        team_stats = weekly_data.groupby(['recent_team', 'season']).agg({
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'completions': 'sum',
            'attempts': 'sum',
            'interceptions': 'sum',
            'passing_tds': 'sum',
            'rushing_tds': 'sum',
            'sacks': 'sum',
            'sack_yards': 'sum',
            'rushing_fumbles_lost': 'sum',
            'passing_first_downs': 'sum',
            'rushing_first_downs': 'sum',
            'passing_epa': 'sum',
            'rushing_epa': 'sum'
        }).reset_index()
        
        # Rename columns for consistency
        team_stats = team_stats.rename(columns={
            'recent_team': 'team'
        })
        
        # Calculate derived metrics
        team_stats['total_yards'] = team_stats['passing_yards'] + team_stats['rushing_yards']
        team_stats['total_tds'] = team_stats['passing_tds'] + team_stats['rushing_tds']
        team_stats['turnovers'] = team_stats['interceptions'] + team_stats['rushing_fumbles_lost']
        team_stats['completion_rate'] = (team_stats['completions'] / team_stats['attempts'] * 100).round(1)
        team_stats['yards_per_game'] = (team_stats['total_yards'] / current_week).round(1)
        team_stats['total_first_downs'] = team_stats['passing_first_downs'] + team_stats['rushing_first_downs']
        team_stats['total_epa'] = team_stats['passing_epa'] + team_stats['rushing_epa']
        team_stats['points_per_game'] = ((team_stats['total_tds'] * 6) / current_week).round(1)  # Rough estimate based on TDs
        
        return team_stats
    except Exception as e:
        print(f"Error in fetch_team_stats: {str(e)}")
        raise

def fetch_player_stats(years):
    """
    Fetches player statistics for the specified years.

    Args:
        years (list): List of years (e.g., [2022, 2023]) to fetch data for.

    Returns:
        pd.DataFrame: DataFrame containing player statistics.
    """
    # Import weekly data for the specified years
    weekly_data = nfl.import_weekly_data(years)
    
    # Group by player and season, then calculate aggregate statistics
    player_stats = weekly_data.groupby(['player_id', 'season']).agg({
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'touchdowns': 'sum',
        # Add more aggregate statistics as needed
    }).reset_index()
    
    # Merge with player information to get player names
    player_info = nfl.import_rosters(years)
    player_stats = pd.merge(player_stats, player_info, on='player_id', how='left')
    
    return player_stats

def fetch_team_records(years):
    """
    Fetches team records and current streaks using schedule data.
    """
    try:
        # Import schedule data for current season
        current_year = datetime.now().year
        # If we're before September, use previous season
        if datetime.now().month < 9:
            current_year -= 1
            
        schedule_data = nfl.import_schedules([current_year])
        
        # Debug info
        print("\nSchedule data columns:")
        print(schedule_data.columns.tolist())
        print("\nSample game data:")
        print(schedule_data.iloc[0])
        
        # Filter for completed games only (games with scores)
        schedule_data = schedule_data[
            (schedule_data['game_type'] == 'REG') &
            pd.notna(schedule_data['home_score'])
        ]
        
        # Calculate wins and losses
        team_records = {}
        
        # Process all teams
        all_teams = set(schedule_data['home_team'].unique()) | set(schedule_data['away_team'].unique())
        
        for team in all_teams:
            # Get home and away games
            home_games = schedule_data[schedule_data['home_team'] == team]
            away_games = schedule_data[schedule_data['away_team'] == team]
            
            wins = 0
            losses = 0
            streak = 0
            streak_type = None
            
            # Combine and sort all games by week
            all_games = pd.concat([
                home_games.assign(is_home=True),
                away_games.assign(is_home=False)
            ]).sort_values('week')
            
            for _, game in all_games.iterrows():
                if game['is_home']:
                    team_score = game['home_score']
                    opp_score = game['away_score']
                else:
                    team_score = game['away_score']
                    opp_score = game['home_score']
                
                if pd.isna(team_score) or pd.isna(opp_score):
                    continue
                    
                if team_score > opp_score:
                    wins += 1
                    if streak_type == 'W':
                        streak += 1
                    else:
                        streak = 1
                        streak_type = 'W'
                elif team_score < opp_score:
                    losses += 1
                    if streak_type == 'L':
                        streak += 1
                    else:
                        streak = 1
                        streak_type = 'L'
            
            team_records[team] = {
                'wins': wins,
                'losses': losses,
                'streak': f"{streak_type}{streak}" if streak_type else "N/A"
            }
        
        print("\nTeam records calculated:", team_records)
        return team_records
        
    except Exception as e:
        print(f"Error in fetch_team_records: {str(e)}")
        print(f"Debug info: {e.__class__.__name__}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("\nTraceback:")
            traceback.print_tb(e.__traceback__)
        raise

if __name__ == "__main__":
    # Example usage
    years = [2022, 2023]
    team_stats = fetch_team_stats(years)
    player_stats = fetch_player_stats(years)
    
    # Save to CSV files
    team_stats.to_csv('data/team_stats.csv', index=False)
    player_stats.to_csv('data/player_stats.csv', index=False)
    
    print("Team and player statistics have been fetched and saved.")
