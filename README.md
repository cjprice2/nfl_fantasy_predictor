# NFL Fantasy Predictor

## Overview
NFL Fantasy Predictor is a machine learning-based application designed to predict weekly fantasy points for NFL players. The tool leverages historical player performance, injury data, and opponent strength to provide dynamic predictions for PPR and non-PPR formats. By emphasizing recent performance data and accounting for player injuries, the predictor aims to offer more accurate projections for fantasy football enthusiasts.

## Features
- **Dynamic Fantasy Predictions**: Provides weekly predictions for both PPR and non-PPR formats based on historical player data and current season trends.
- **Injury Adjustments**: Excludes games where players were injured to ensure accurate predictions.
- **Opponent Strength Adjustment**: Scales predictions based on the relative strength of the opposing defense.
- **Weighted Historical Data**: Emphasizes recent performances while considering past seasons, with adjustable weighting to prioritize the most relevant data.
- **User-Friendly Interface**: Allows users to select teams, players, and weeks to generate predictions interactively.

## Technologies Used
- **Python**: Core logic and data processing.
- **Flask**: Backend framework to handle API endpoints.
- **scikit-learn**: Machine learning model training and prediction.
- **nfl_data_py**: NFL data acquisition and preprocessing.
- **HTML/CSS/JavaScript**: Frontend interface for user interaction.

## Usage
1. Select an NFL team and player.
2. Choose the week for which you want the prediction.
3. Get instant predictions for the selected player and week, adjusted based on opponent strength and injury history.

## Future Enhancements
- Implement additional models for more precise predictions.
- Integrate real-time data updates during the NFL season.
- Add support for custom scoring settings.