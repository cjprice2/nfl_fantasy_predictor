from flask import Flask, request, jsonify, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import os

app = Flask(__name__)

# Directory setup for saving CSV files
csv_directory = os.path.join(os.getcwd(), 'csv_data')
os.makedirs(csv_directory, exist_ok=True)

# Determine the current NFL week
nfl_season_start = datetime(2024, 9, 3)
today = datetime.now()
days_since_start = (today - nfl_season_start).days

current_nfl_week = max(0, (days_since_start // 7) + 1) if days_since_start >= 0 else 0
last_week = max(0, current_nfl_week - 1)
current_year = today.year
years = list(range(2021, current_year + 1))

# Function to load or import data
def load_or_import_data(filename, import_function, *args, **kwargs):
    file_path = os.path.join(csv_directory, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        data = import_function(*args, **kwargs)
        data.to_csv(file_path, index=False)
        return data

# Load necessary data
games = load_or_import_data('games.csv', nfl.import_schedules, years)
depth_charts = load_or_import_data('depth_charts.csv', nfl.import_depth_charts, years)
player_stats = load_or_import_data(
    'player_stats.csv', nfl.import_weekly_data, years,
    ['season', 'week', 'player_id', 'player_name', 'fantasy_points_ppr',
     'fantasy_points', 'recent_team', 'opponent_team', 'passing_yards',
     'rushing_yards', 'receiving_yards']
)
current_year_players = player_stats[player_stats['season'] == current_year].drop_duplicates(subset=['player_id'], keep='last')
injuries = load_or_import_data('injuries.csv', nfl.import_injuries, years)
teams = load_or_import_data('teams.csv', nfl.import_team_desc)

# Injury status mapping
injury_status_mapping = {
    'Out': 0,
    'Doubtful': 0.1,
    'Questionable': 0.5,
    'Healthy': 1
}
injuries['numeric_status'] = injuries['report_status'].map(injury_status_mapping).fillna(1)

# Update NFL data
def update_nfl_data():
    global games, depth_charts, player_stats, current_year_players, injuries, last_week
    print("Updating NFL Data...")
    years = list(range(2021, datetime.now().year + 1))
    games = nfl.import_schedules(years)
    games.to_csv(os.path.join(csv_directory, 'games.csv'), index=False)
    depth_charts = nfl.import_depth_charts(years)
    depth_charts.to_csv(os.path.join(csv_directory, 'depth_charts.csv'), index=False)
    player_stats = nfl.import_weekly_data(years, [
        'season', 'week', 'player_id', 'player_name', 'fantasy_points_ppr',
        'fantasy_points', 'recent_team', 'opponent_team', 'passing_yards',
        'rushing_yards', 'receiving_yards'
    ])
    player_stats.to_csv(os.path.join(csv_directory, 'player_stats.csv'), index=False)
    current_year_players = player_stats[player_stats['season'] == datetime.now().year].drop_duplicates(subset=['player_id'], keep='last')
    current_year_players.to_csv(os.path.join(csv_directory, 'current_year_players.csv'), index=False)
    injuries = nfl.import_injuries(years)
    injuries['numeric_status'] = injuries['report_status'].map(injury_status_mapping).fillna(1)
    injuries.to_csv(os.path.join(csv_directory, 'injuries.csv'), index=False)
    print("NFL Data updated successfully!")

# Helper functions
def find_opponent(team_id, week, games_df):
    week = int(week)
    game = games_df[
        (games_df['week'] == week) &
        (games_df['season'] == current_year) &
        ((games_df['home_team'] == team_id) | (games_df['away_team'] == team_id))
    ]
    if not game.empty:
        return game['away_team'].iloc[0] if game['home_team'].iloc[0] == team_id else game['home_team'].iloc[0]
    else:
        return None  # No opponent found for the given team and week

def calculate_league_average_defense(games):
    relevant_games = games[(games['season'] < current_year) | 
                           ((games['season'] == current_year) & (games['week'] <= last_week))]
    if relevant_games.empty:
        return 1
    points_allowed = pd.concat([relevant_games['away_score'], relevant_games['home_score']])
    return points_allowed.mean()

def calculate_opponent_strength(team_id, games):
    relevant_games = games[((games['season'] < current_year) |
                            ((games['season'] == current_year) & (games['week'] <= last_week))) &
                           ((games['home_team'] == team_id) | (games['away_team'] == team_id))]
    if relevant_games.empty:
        return 0
    points_allowed = relevant_games.apply(
        lambda row: row['away_score'] if row['home_team'] == team_id else row['home_score'], axis=1
    )
    return points_allowed.mean()

def get_season_weight(season):
    if season == current_year:
        return 20
    elif season == current_year - 1:
        return 5
    else:
        return 1

def calculate_player_features(player_id, player_stats, injuries):
    # Filter player stats
    player_data = player_stats[
        (player_stats['player_id'] == player_id) &
        (player_stats['season'] >= 2021) &
        (player_stats['season'] <= current_year) &
        (player_stats['week'] <= last_week)
    ].copy()  # Create a copy explicitly to avoid SettingWithCopyWarning

    if player_data.empty:
        return [0, 0, 1]

    # Apply season weight
    player_data.loc[:, 'season_weight'] = player_data['season'].apply(get_season_weight)

    # Calculate weighted averages
    past_performance_ppr = np.average(player_data['fantasy_points_ppr'], weights=player_data['season_weight'])
    past_performance_non_ppr = np.average(player_data['fantasy_points'], weights=player_data['season_weight'])

    # Get the latest injury status
    filtered_injuries = injuries[injuries['gsis_id'] == player_id]

    if not filtered_injuries.empty:
        # Sort by week and season in descending order to get the latest injury
        latest_injury = filtered_injuries.sort_values(by=['season', 'week'], ascending=[False, False]).iloc[0]

        # Check if the latest injury is from the current year
        if latest_injury['season'] == current_year:
            injury_status = latest_injury['numeric_status']
        else:
            # Check if the player is on the depth chart for the current week
            depth_chart_entry = depth_charts[
                (depth_charts['gsis_id'] == player_id) &
                (depth_charts['season'] == current_year) &
                (depth_charts['week'] == current_nfl_week)
            ]

            if not depth_chart_entry.empty:
                # Player is on the depth chart but not in the injuries list for the current year
                injury_status = 1  # Assume Healthy
            else:
                # Player is not in the injuries list and not on the depth chart
                injury_status = 0  # Assume Out
    else:
        # If no injuries are found, check the depth chart
        depth_chart_entry = depth_charts[
            (depth_charts['gsis_id'] == player_id) &
            (depth_charts['season'] == current_year) &
            (depth_charts['week'] == current_nfl_week)
        ]

        if not depth_chart_entry.empty:
            # Player is on the depth chart but not in the injuries list
            injury_status = 1  # Assume Healthy
        else:
            # Player is not in the injuries list and not on the depth chart
            injury_status = 0  # Assume Out
    
        
    return [past_performance_ppr, past_performance_non_ppr, injury_status]


# Train models
def train_models():
    feature_data, target_ppr, target_non_ppr = [], [], []
    for player_id in current_year_players['player_id']:
        features = calculate_player_features(player_id, player_stats, injuries)
        feature_data.append(features)

        # Filter player stats for training targets
        player_games = player_stats[
            (player_stats['player_id'] == player_id) &
            ((player_stats['season'] < current_year) |
             ((player_stats['season'] == current_year) & (player_stats['week'] <= last_week)))
        ].copy()  # Explicitly copy to avoid SettingWithCopyWarning

        if not player_games.empty:
            player_games.loc[:, 'season_weight'] = player_games['season'].apply(get_season_weight)
            target_ppr.append(np.average(player_games['fantasy_points_ppr'], weights=player_games['season_weight']))
            target_non_ppr.append(np.average(player_games['fantasy_points'], weights=player_games['season_weight']))
        else:
            target_ppr.append(0)
            target_non_ppr.append(0)

    X_train_ppr, X_test_ppr, y_train_ppr, y_test_ppr = train_test_split(feature_data, target_ppr, test_size=0.2, random_state=42)
    X_train_non_ppr, X_test_non_ppr, y_train_non_ppr, y_test_non_ppr = train_test_split(feature_data, target_non_ppr, test_size=0.2, random_state=42)
    model_ppr = RandomForestRegressor(random_state=42).fit(X_train_ppr, y_train_ppr)
    model_non_ppr = RandomForestRegressor(random_state=42).fit(X_train_non_ppr, y_train_non_ppr)
    return model_ppr, model_non_ppr

model_ppr, model_non_ppr = train_models()

# Flask API Endpoints
@app.route('/teams', methods=['GET'])
def get_teams():
    active_teams = teams[['team_abbr', 'team_name']].drop_duplicates().to_dict(orient='records')
    return jsonify(active_teams)

@app.route('/players/<team_id>', methods=['GET'])
def get_players(team_id):
    players = current_year_players[current_year_players['recent_team'] == team_id][['player_id', 'player_name']].drop_duplicates()
    return jsonify(players.to_dict(orient='records'))

@app.route('/weeks', methods=['GET'])
def get_weeks():
    return jsonify(list(range(current_nfl_week, 19)))  # NFL season weeks 1-18

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
    selected_week = int(data['week'])

    # Check if the player is a QB and is a starter in the depth chart
    qb_depth = depth_charts[
        (depth_charts['gsis_id'] == player_id) &
        (depth_charts['season'] == current_year) &
        (depth_charts['week'] == current_nfl_week) &
        (depth_charts['position'] == 'QB')
    ]
    if not qb_depth.empty:
        # If player is not the starting QB (depth != 1), return 0 points
        if qb_depth['depth_team'].iloc[0] != 1:
            return jsonify({'predicted_fantasy_points': 0})

    # Calculate player features
    features = calculate_player_features(player_id, player_stats, injuries)

    # Get the opponent for the selected week and calculate opponent strength
    opponent_id = find_opponent(team_id, selected_week, games)
    opponent_strength = calculate_opponent_strength(opponent_id, games) if opponent_id else 0

    # Calculate league average defense
    league_average_defense = calculate_league_average_defense(games)
    scaling_factor = (
        1 + 0.5 * ((league_average_defense / opponent_strength) - 1)
        if opponent_strength else 1
    )

    # Predict based on past performance and injury status
    if format_type == 'ppr':
        predicted_points = model_ppr.predict([features])[0] * scaling_factor
    else:
        predicted_points = model_non_ppr.predict([features])[0] * scaling_factor

    # Apply injury status adjustments
    injury_status = features[2]
    if injury_status == 0:
        predicted_points = 0  # Out
    elif injury_status == 0.1 and selected_week == last_week + 1:
        predicted_points *= 0.1  # Doubtful
    elif injury_status == 0.5 and selected_week == last_week + 1:
        predicted_points *= 0.5  # Questionable

    return jsonify({'predicted_fantasy_points': predicted_points})

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(update_nfl_data, 'cron', day_of_week='tue', hour=0, minute=0)
scheduler.start()

# Serve static files
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

