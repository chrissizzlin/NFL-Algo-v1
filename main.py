import pandas as pd
from scripts.fetch_stats import fetch_team_stats
from scripts.fetch_injuries import fetch_current_injuries
from scripts.calculate_metrics import calculate_metrics, predict_outcome
from scripts.fetch_schedule import get_current_season_and_week
import os

def main():
    # Get current season and week automatically
    current_season, current_week = get_current_season_and_week()
    print(f"Analyzing data for {current_season} Season, Week {current_week}")

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Fetch team statistics
    print("Fetching team statistics...")
    team_stats = fetch_team_stats([current_season])

    # Fetch current injury reports
    print("Fetching current injury reports...")
    injury_reports = fetch_current_injuries()

    # Calculate metrics for each team
    print("Calculating team metrics...")
    team_metrics = calculate_metrics(team_stats, injury_reports)

    # Predict outcomes for upcoming games
    print("Predicting game outcomes...")
    predictions = predict_outcome(team_metrics)

    # Display predictions
    print("\nGame Predictions:")
    print(predictions)

    # Save predictions to a CSV file
    predictions.to_csv('data/game_predictions.csv', index=False)
    print("Predictions have been saved to 'data/game_predictions.csv'.")

if __name__ == "__main__":
    main()
