import pandas as pd
import numpy as np

def load_data():
    """
    Loads team stats and injury reports from CSV files.

    Returns:
        tuple: DataFrames containing team stats and injury reports.
    """
    team_stats = pd.read_csv('data/team_stats.csv')
    injury_reports = pd.read_csv('data/relevant_injuries.csv')
    return team_stats, injury_reports

def calculate_metrics(home_stats, away_stats, home_injuries, away_injuries):
    """
    Calculates comprehensive metrics for prediction using weighted factors.
    """
    try:
        # Get current week from the first team's stats
        current_week = max(1, home_stats['week'].max() if 'week' in home_stats.columns else 13)
        
        # 1. Offensive Efficiency (40% of total score)
        def calculate_offensive_score(stats):
            points_per_game = stats['points_per_game'].iloc[0]
            yards_per_game = stats['yards_per_game'].iloc[0]
            completion_rate = stats['completion_rate'].iloc[0]
            first_downs = stats['total_first_downs'].iloc[0] / current_week
            pass_yards = stats['passing_yards'].iloc[0] / current_week
            rush_yards = stats['rushing_yards'].iloc[0] / current_week
            red_zone_pct = stats['red_zone_pct'].iloc[0] if 'red_zone_pct' in stats else 50.0
            third_down_pct = stats['third_down_pct'].iloc[0] if 'third_down_pct' in stats else 40.0
            
            # Normalize each metric (0-1 scale)
            norm_points = min(points_per_game / 35.0, 1.0)  # 35 points per game is excellent
            norm_yards = min(yards_per_game / 400.0, 1.0)  # 400 yards per game is excellent
            norm_completion = completion_rate / 100.0
            norm_first_downs = min(first_downs / 25.0, 1.0)  # 25 first downs per game is excellent
            norm_pass = min(pass_yards / 300.0, 1.0)  # 300 pass yards per game is excellent
            norm_rush = min(rush_yards / 150.0, 1.0)  # 150 rush yards per game is excellent
            norm_red_zone = red_zone_pct / 100.0
            norm_third_down = third_down_pct / 100.0
            
            # Weighted sum of offensive metrics
            return (norm_points * 0.30 +       # Points are most important
                   norm_yards * 0.15 +         # Total yards
                   norm_completion * 0.10 +    # Passing efficiency
                   norm_first_downs * 0.10 +   # Drive sustainability
                   norm_pass * 0.10 +          # Passing threat
                   norm_rush * 0.10 +          # Running threat
                   norm_red_zone * 0.10 +      # Red zone efficiency
                   norm_third_down * 0.05) * 40  # Third down efficiency
        
        # 2. Defensive Efficiency (30% of total score)
        def calculate_defensive_score(stats):
            sacks = stats['sacks'].iloc[0] / current_week
            turnovers = stats['turnovers'].iloc[0] / current_week
            points_allowed = stats['points_allowed_per_game'].iloc[0] if 'points_allowed_per_game' in stats else 24.0
            yards_allowed = stats['yards_allowed_per_game'].iloc[0] if 'yards_allowed_per_game' in stats else 350.0
            opp_completion_pct = stats['opp_completion_pct'].iloc[0] if 'opp_completion_pct' in stats else 65.0
            pressure_rate = stats['pressure_rate'].iloc[0] if 'pressure_rate' in stats else 25.0
            
            # Normalize each metric (0-1 scale)
            norm_sacks = min(sacks / 4.0, 1.0)  # 4 sacks per game is excellent
            norm_turnovers = min(turnovers / 2.5, 1.0)  # 2.5 turnovers per game is excellent
            norm_points_allowed = 1.0 - min(points_allowed / 30.0, 1.0)  # Lower points allowed is better
            norm_yards_allowed = 1.0 - min(yards_allowed / 400.0, 1.0)  # Lower yards allowed is better
            norm_opp_completion = 1.0 - (opp_completion_pct / 100.0)  # Lower completion % allowed is better
            norm_pressure = pressure_rate / 100.0
            
            # Weighted sum of defensive metrics
            return (norm_points_allowed * 0.35 +    # Points prevention is key
                   norm_yards_allowed * 0.20 +      # Yards prevention
                   norm_turnovers * 0.15 +          # Turnover creation
                   norm_sacks * 0.10 +              # Pass rush success
                   norm_opp_completion * 0.10 +     # Pass defense
                   norm_pressure * 0.10) * 30       # Consistent pressure
        
        # 3. EPA and Advanced Metrics (15% of total score)
        def calculate_advanced_score(stats):
            epa = stats['total_epa'].iloc[0]
            success_rate = stats['success_rate'].iloc[0] if 'success_rate' in stats else 0.45
            pass_epa = stats['passing_epa'].iloc[0] if 'passing_epa' in stats else 0
            rush_epa = stats['rushing_epa'].iloc[0] if 'rushing_epa' in stats else 0
            explosive_play_rate = stats['explosive_play_rate'].iloc[0] if 'explosive_play_rate' in stats else 0.10
            
            # Normalize EPA (-100 to 100 scale to 0-1)
            norm_total_epa = (epa + 100) / 200.0
            norm_pass_epa = (pass_epa + 50) / 100.0
            norm_rush_epa = (rush_epa + 50) / 100.0
            norm_explosive = min(explosive_play_rate / 0.15, 1.0)  # 15% explosive play rate is excellent
            
            # Weighted sum of advanced metrics
            return (norm_total_epa * 0.35 +
                   success_rate * 0.25 +
                   norm_pass_epa * 0.15 +
                   norm_rush_epa * 0.15 +
                   norm_explosive * 0.10) * 15
        
        # 4. Injury Impact (15% of total score)
        def calculate_injury_score(injuries):
            if injuries.empty:
                return 15.0  # Full score if no injuries
            
            impact = calculate_injury_impact(injuries)
            # Convert impact to 0-15 scale (lower impact is better)
            return max(0, 15.0 - min(impact, 15.0))
        
        # Calculate individual scores
        home_offensive = calculate_offensive_score(home_stats)
        away_offensive = calculate_offensive_score(away_stats)
        home_defensive = calculate_defensive_score(home_stats)
        away_defensive = calculate_defensive_score(away_stats)
        home_advanced = calculate_advanced_score(home_stats)
        away_advanced = calculate_advanced_score(away_stats)
        home_injury = calculate_injury_score(home_injuries)
        away_injury = calculate_injury_score(away_injuries)
        
        # Calculate final scores
        home_final = (home_offensive + home_defensive + home_advanced + home_injury)
        away_final = (away_offensive + away_defensive + away_advanced + away_injury)
        
        # Create metrics DataFrame
        metrics = pd.DataFrame({
            'team': [home_stats['team'].iloc[0], away_stats['team'].iloc[0]],
            'offensive_score': [home_offensive, away_offensive],
            'defensive_score': [home_defensive, away_defensive],
            'advanced_score': [home_advanced, away_advanced],
            'injury_score': [home_injury, away_injury],
            'final_score': [home_final, away_final]
        })
        
        return metrics
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        raise

def predict_outcome(metrics):
    """
    Predicts game outcomes based on calculated metrics using a sophisticated probability model.
    """
    try:
        # Get the score differential
        score_diff = metrics['final_score'].iloc[0] - metrics['final_score'].iloc[1]
        
        # Calculate win probability using a logistic function
        # This creates a more realistic S-curve probability distribution
        def logistic(x, k=0.1):
            return 1 / (1 + np.exp(-k * x))
        
        # Convert score difference to probability (centered at 50%)
        base_probability = logistic(score_diff) * 100
        
        # Add home field advantage (5% boost)
        home_probability = min(base_probability + 5, 99)
        away_probability = max(100 - home_probability, 1)
        
        # Update metrics with probabilities
        metrics.loc[0, 'win_probability'] = home_probability
        metrics.loc[1, 'win_probability'] = away_probability
        
        # Add predicted outcome
        metrics['predicted_outcome'] = metrics['win_probability']
        
        return metrics
    except Exception as e:
        print(f"Error in predict_outcome: {str(e)}")
        raise

def calculate_injury_impact(injuries):
    """
    Calculates the total injury impact on a team.
    """
    if injuries.empty:
        return 0.0
        
    # Weight injuries by position importance
    position_weights = {
        'QB': 10.0,    # Quarterbacks have highest impact
        'RB': 4.0,     # Running backs
        'WR': 3.5,     # Wide receivers
        'TE': 3.0,     # Tight ends
        'OL': 3.0,     # Offensive line
        'DL': 3.0,     # Defensive line
        'LB': 3.0,     # Linebackers
        'DB': 3.0,     # Defensive backs
        'K': 2.0,      # Kicker
        'P': 1.5       # Punter
    }
    
    # Weight based on starter status
    starter_weights = {
        '⭐⭐⭐': 1.0,  # Triple star (QB)
        '⭐⭐': 0.8,    # Double star (key positions)
        '⭐': 0.6,      # Single star (other starters)
        '📋': 0.3       # Depth player
    }
    
    # Weight based on injury status
    status_weights = {
        'Out': 1.0,
        'Doubtful': 0.8,
        'Questionable': 0.5,
        'Did Not Participate In Practice': 0.7,
        'Limited Participation in Practice': 0.3
    }
    
    total_impact = 0.0
    for _, injury in injuries.iterrows():
        # Get position weight (first two characters of position)
        pos = injury['position'][:2] if pd.notna(injury['position']) else 'OL'
        pos_weight = position_weights.get(pos, 2.0)
        
        # Get starter weight
        starter_status = injury['starter_status'] if pd.notna(injury['starter_status']) else '📋'
        starter_weight = starter_weights.get(starter_status, 0.3)
        
        # Get status weight
        status = injury['status'] if pd.notna(injury['status']) else 'Questionable'
        status_weight = status_weights.get(status, 0.5)
        
        # Calculate individual impact
        impact = pos_weight * starter_weight * status_weight
        total_impact += impact
    
    # Normalize total impact to 0-10 scale
    normalized_impact = min(total_impact, 10.0)
    return normalized_impact

def calculate_implied_probability(moneyline):
    """
    Convert moneyline odds to implied probability.
    Example: -110 -> 52.4%, +150 -> 40%
    """
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return (-moneyline) / (-moneyline + 100)

def calculate_betting_edge(win_probability, moneyline):
    """
    Calculate the edge percentage based on our predicted win probability vs. implied odds.
    Positive edge means potential value bet.
    """
    implied_prob = calculate_implied_probability(moneyline)
    edge = win_probability - implied_prob
    return edge * 100  # Convert to percentage

def analyze_betting_value(metrics, home_moneyline, away_moneyline):
    """
    Analyze betting value and provide recommendations based on:
    1. Win probability vs. implied odds (edge)
    2. Minimum win probability threshold
    3. Minimum edge percentage threshold
    4. Confidence level based on metrics
    
    Returns dict with betting analysis for both teams.
    """
    try:
        # Get win probabilities
        home_prob = metrics.loc[0, 'win_probability'] / 100  # Convert to decimal
        away_prob = metrics.loc[1, 'win_probability'] / 100
        
        # Calculate edges
        home_edge = calculate_betting_edge(home_prob, home_moneyline)
        away_edge = calculate_betting_edge(away_prob, away_moneyline)
        
        # Calculate confidence scores (0-1 scale)
        def calculate_confidence(team_idx):
            # Get team's scores
            offensive = metrics.loc[team_idx, 'offensive_score'] / 40  # Normalized by max possible
            defensive = metrics.loc[team_idx, 'defensive_score'] / 30
            advanced = metrics.loc[team_idx, 'advanced_score'] / 15
            injury = metrics.loc[team_idx, 'injury_score'] / 15
            
            # Weight the components
            confidence = (offensive * 0.4 +
                        defensive * 0.3 +
                        advanced * 0.2 +
                        injury * 0.1)
            return confidence
        
        home_confidence = calculate_confidence(0)
        away_confidence = calculate_confidence(1)
        
        # Define betting thresholds
        MIN_WIN_PROB = 0.60  # 60% minimum win probability
        MIN_EDGE = 5.0       # 5% minimum edge
        MIN_CONFIDENCE = 0.6  # 60% minimum confidence score
        
        # Analyze home team bet
        home_analysis = {
            'team': metrics.loc[0, 'team'],
            'win_probability': home_prob * 100,
            'implied_probability': calculate_implied_probability(home_moneyline) * 100,
            'edge': home_edge,
            'confidence': home_confidence * 100,
            'moneyline': home_moneyline,
            'recommended': False,
            'bet_rating': 0,
            'explanation': []
        }
        
        # Analyze away team bet
        away_analysis = {
            'team': metrics.loc[1, 'team'],
            'win_probability': away_prob * 100,
            'implied_probability': calculate_implied_probability(away_moneyline) * 100,
            'edge': away_edge,
            'confidence': away_confidence * 100,
            'moneyline': away_moneyline,
            'recommended': False,
            'bet_rating': 0,
            'explanation': []
        }
        
        # Analyze each team's betting value
        for analysis in [home_analysis, away_analysis]:
            # Calculate bet rating (0-10 scale)
            edge_score = min(analysis['edge'] / 2, 5)  # Up to 5 points for edge
            prob_score = min((analysis['win_probability'] - 50) / 10, 3)  # Up to 3 points for win probability
            conf_score = min(analysis['confidence'] / 50, 2)  # Up to 2 points for confidence
            
            analysis['bet_rating'] = round(edge_score + prob_score + conf_score, 1)
            
            # Build explanation
            if analysis['edge'] > MIN_EDGE:
                analysis['explanation'].append(f"Good value: {analysis['edge']:.1f}% edge")
            
            if analysis['win_probability'] > MIN_WIN_PROB * 100:
                analysis['explanation'].append(f"Strong win probability: {analysis['win_probability']:.1f}%")
            
            if analysis['confidence'] > MIN_CONFIDENCE * 100:
                analysis['explanation'].append(f"High confidence: {analysis['confidence']:.1f}%")
            
            # Determine if bet is recommended
            if (analysis['edge'] > MIN_EDGE and 
                analysis['win_probability'] > MIN_WIN_PROB * 100 and 
                analysis['confidence'] > MIN_CONFIDENCE * 100):
                analysis['recommended'] = True
                analysis['explanation'].append("✅ Bet meets all criteria")
            else:
                reasons = []
                if analysis['edge'] <= MIN_EDGE:
                    reasons.append(f"Edge ({analysis['edge']:.1f}%) below {MIN_EDGE}% threshold")
                if analysis['win_probability'] <= MIN_WIN_PROB * 100:
                    reasons.append(f"Win probability ({analysis['win_probability']:.1f}%) below {MIN_WIN_PROB*100}% threshold")
                if analysis['confidence'] <= MIN_CONFIDENCE * 100:
                    reasons.append(f"Confidence ({analysis['confidence']:.1f}%) below {MIN_CONFIDENCE*100}% threshold")
                analysis['explanation'].append("❌ " + "; ".join(reasons))
        
        return {
            'home': home_analysis,
            'away': away_analysis
        }
        
    except Exception as e:
        print(f"Error in analyze_betting_value: {str(e)}")
        raise

if __name__ == "__main__":
    # Load data
    team_stats, injury_reports = load_data()
    
    # Calculate metrics
    metrics = calculate_metrics(team_stats, injury_reports)
    
    # Predict outcomes
    predictions = predict_outcome(metrics)
    
    # Save predictions to CSV
    predictions.to_csv('data/predictions.csv', index=False)
    
    print("Predictions have been calculated and saved.")
