from flask import Flask, request, jsonify, send_from_directory
import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import os

app = Flask(__name__)

# Set last_week to 2 (static) since it's the last week with data
last_week = 2

# Get the current year and create a list of years from 2021 to current year
current_year = 2024
years = list(range(2021, current_year + 1))

# Load schedules for the years from 2021 to the current year (emphasizing 2024)
games = nfl.import_schedules(years)

# Load depth charts to prioritize starters
depth_charts = nfl.import_depth_charts(years)

# Load player stats for 2021 to the current year
player_stats = nfl.import_weekly_data(years, ['season', 'week', 'player_id', 'player_name', 'fantasy_points_ppr', 'fantasy_points', 'recent_team', 'opponent_team', 'passing_yards', 'rushing_yards', 'receiving_yards'])

# Filter to only get player stats for 2024 and use the most recent team in 2024
current_year_players = player_stats[player_stats['season'] == 2024].drop_duplicates(subset=['player_id'], keep='last')

# Load injury data for 2021 to the current year
injuries = nfl.import_injuries(years)

# Load team descriptions
teams = nfl.import_team_desc()
teams.to_csv('teams.csv', index=False)
# Calculate the league average defense up to last_week
def calculate_league_average_defense(games):
    # Filter games for the years 2021-2023 and for 2024 up to and including last_week
    all_games = games[(games['season'] < current_year) | 
                      ((games['season'] == current_year) & (games['week'] <= last_week))]
    
    if all_games.empty:
        return 1  # Default to 1 to avoid division by zero
    
    # Calculate points allowed by home and away teams separately
    home_points_allowed = all_games['away_score']
    away_points_allowed = all_games['home_score']
    
    # Combine the points allowed for all teams
    total_points_allowed = pd.concat([home_points_allowed, away_points_allowed])

    # Calculate the league average points allowed per game
    league_average = total_points_allowed.mean()
    
    return league_average

# Calculate the opponent strength based on data up to last_week
def calculate_opponent_strength(team_id, games):
    # Filter games for the years 2021-2023 and for 2024 up to and including last_week
    opponent_games = games[((games['season'] < current_year) |
                            ((games['season'] == current_year) & (games['week'] <= last_week))) &
                           ((games['home_team'] == team_id) | (games['away_team'] == team_id))]

    if opponent_games.empty:
        return 0  # Default to 0 if no past data is available for this team
    
    # Calculate points allowed by the opponent team in each game
    points_allowed = opponent_games.apply(
        lambda row: row['away_score'] if row['home_team'] == team_id else row['home_score'], axis=1
    )

    # Return the average points allowed by the opponent team
    return points_allowed.mean()

# Get the injury status of a player
def get_injury_status(player_id, injuries):
    player_injuries = injuries[injuries['gsis_id'] == player_id]
    if not player_injuries.empty:
        return 1  # Injured
    return 0  # Not Injured

# Function to assign weights based on season
def get_season_weight(season):
    if season == 2024:
        return 20  # Highest weight for the current season
    elif season == 2023:
        return 5  # Lower weight for the previous season
    else:
        return 1  # Minimal weight for older seasons

# Calculate player features based on past performance, injury status, and season weights
def calculate_player_features(player_id, team_id, games, player_stats, injuries):
    # Use data from 2021 until the last_week in the current year
    player_data = player_stats[
        (player_stats['player_id'] == player_id) &
        (player_stats['season'] >= 2021) &  # Use data starting from 2021
        (player_stats['season'] <= current_year) &  # Up to current_year
        (player_stats['week'] <= last_week)  # Use data up to last_week
    ]

    if player_data.empty:
        return [0, 0, 0]  # Return a default feature set if no data available

    # Apply weights based on the season
    player_data = player_data.copy()  # Create a copy to avoid the SettingWithCopyWarning
    player_data.loc[:, 'season_weight'] = player_data['season'].apply(get_season_weight)

    # Calculate weighted average performance
    past_performance_ppr = np.average(
        player_data['fantasy_points_ppr'],
        weights=player_data['season_weight']
    )
    past_performance_non_ppr = np.average(
        player_data['fantasy_points'],
        weights=player_data['season_weight']
    )

    # Get the injury status for the player
    injury_status = get_injury_status(player_id, injuries)

    # Features will include player's performance data and injury status, no opponent strength
    features = [past_performance_ppr, past_performance_non_ppr, injury_status]

    # Replace any NaN values with 0 (imputation)
    features = [0 if np.isnan(f) else f for f in features]

    return features

# Find the opponent for a team in a specific week
def find_opponent(team_id, week, games_df):
    week = int(week)
    # Query the games dataset for the specific week and team
    game = games_df[(games_df['week'] == week) & (games_df['season'] == current_year) & 
                    ((games_df['home_team'] == team_id) | (games_df['away_team'] == team_id))]

    if not game.empty:
        # If the team is the home team, the opponent is the away team, and vice versa
        return game['away_team'].iloc[0] if game['home_team'].iloc[0] == team_id else game['home_team'].iloc[0]
    else:
        return None  # Return None if no opponent is found

# Train models to predict PPR and non-PPR fantasy points based on weighted features
# Function to assign weights based on season
def get_season_weight(season):
    if season == 2024:
        return 20  # Highest weight for the current season
    elif season == 2023:
        return 5  # Lower weight for the previous season
    else:
        return 1  # Minimal weight for older seasons

# Train models to predict PPR and non-PPR fantasy points based on weighted features and targets
def train_models():
    player_ids = current_year_players['player_id'].unique()
    feature_data_ppr = []
    feature_data_non_ppr = []
    target_data_ppr = []
    target_data_non_ppr = []

    for player_id in player_ids:
        team_id = current_year_players[current_year_players['player_id'] == player_id]['recent_team'].iloc[0]
        
        # Calculate weighted features
        features = calculate_player_features(player_id, team_id, games, player_stats, injuries)
        feature_data_ppr.append([features[0], features[2]])  # PPR and injury status
        feature_data_non_ppr.append([features[1], features[2]])  # Non-PPR and injury status

        # Get player games for all seasons up to current_year and last_week
        player_games = player_stats[
            (player_stats['player_id'] == player_id) &
            ((player_stats['season'] < current_year) | 
            ((player_stats['season'] == current_year) & 
            (player_stats['week'] <= last_week)))
        ]

        # Apply season weights to target data
        if not player_games.empty:
            # Calculate season weights for target data
            player_games = player_games.copy()
            player_games['season_weight'] = player_games['season'].apply(get_season_weight)
            
            # Calculate weighted average target values
            weighted_target_ppr = np.average(
                player_games['fantasy_points_ppr'],
                weights=player_games['season_weight']
            )
            weighted_target_non_ppr = np.average(
                player_games['fantasy_points'],
                weights=player_games['season_weight']
            )
        else:
            weighted_target_ppr = 0
            weighted_target_non_ppr = 0

        # Append weighted target data
        target_data_ppr.append(weighted_target_ppr)
        target_data_non_ppr.append(weighted_target_non_ppr)

    # Train models with weighted target data
    X_train_ppr, X_test_ppr, y_train_ppr, y_test_ppr = train_test_split(
        feature_data_ppr, target_data_ppr, test_size=0.2
    )
    X_train_non_ppr, X_test_non_ppr, y_train_non_ppr, y_test_non_ppr = train_test_split(
        feature_data_non_ppr, target_data_non_ppr, test_size=0.2
    )

    model_ppr = RandomForestRegressor()
    model_non_ppr = RandomForestRegressor()

    model_ppr.fit(X_train_ppr, y_train_ppr)
    model_non_ppr.fit(X_train_non_ppr, y_train_non_ppr)

    return model_ppr, model_non_ppr

# Train both models
model_ppr, model_non_ppr = train_models()

# Endpoint to get the list of teams
@app.route('/teams', methods=['GET'])
def get_teams():
    # List of active team abbreviations for the current season
    active_team_abbrs = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
    ]
    
    # Filter the teams dataframe to only include active teams
    active_teams = teams[teams['team_abbr'].isin(active_team_abbrs)]
    active_teams_list = active_teams[['team_abbr', 'team_name']].drop_duplicates().to_dict(orient='records')
    
    return jsonify(active_teams_list)

# Endpoint to get the list of players based on the selected team
@app.route('/players/<team_id>', methods=['GET'])
def get_players(team_id):
    players = current_year_players[current_year_players['recent_team'] == team_id]['player_id'].unique()
    player_list = [{'player_id': player_id, 'player_name': current_year_players[current_year_players['player_id'] == player_id]['player_name'].iloc[0]} for player_id in players]
    return jsonify(player_list)

# Endpoint to get the list of weeks
@app.route('/weeks', methods=['GET'])
def get_weeks():
    weeks = list(range(3, 19))  # Future weeks only (from week 3 to 18)
    return jsonify(weeks)

# Serve index.html from the static folder
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/opponents/<team_id>/<week>', methods=['GET'])
def get_opponent(team_id, week):
    opponent = find_opponent(team_id, week, games)
    if opponent:
        return jsonify({'opponent': opponent})
    else:
        return jsonify({'error': 'No opponent found for the selected team and week'}), 404
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    player_id = data['player_id']
    format_type = data['format']  # 'ppr' or 'non_ppr'
    team_id = data['team_id']
    selected_week = data['week']  # User-selected week

    # Get base player features without scaling (two features: PPR/non-PPR performance and injury status)
    features = calculate_player_features(player_id, team_id, games, player_stats, injuries)

    # Get the opponent for the selected week and team
    opponent_id = find_opponent(team_id, selected_week, games)
    
    # Calculate opponent strength based on past games (up to last_week)
    if opponent_id:
        opponent_strength = calculate_opponent_strength(opponent_id, games)
        # Calculate the league-wide average defensive strength
        league_average_defense = calculate_league_average_defense(games)

        scaling_factor_weight = 0.5  # Modify this value to reduce the impact of the scaling factor
        raw_scaling_factor = league_average_defense / opponent_strength if opponent_strength else 1
        scaling_factor = 1 + scaling_factor_weight * (raw_scaling_factor - 1)

    else:
        scaling_factor = 1  # Default scaling if no opponent found
    print(scaling_factor)
    # Predict based on past performance and injury status
    if format_type == 'ppr':
        predicted_points = model_ppr.predict([[features[0], features[2]]])[0]  # PPR and injury status
        predicted_points *= scaling_factor  # Apply opponent strength scaling
    else:
        predicted_points = model_non_ppr.predict([[features[1], features[2]]])[0]  # Non-PPR and injury status
        predicted_points *= scaling_factor  # Apply opponent strength scaling

    return jsonify({
        'predicted_fantasy_points': predicted_points
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
