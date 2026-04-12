""" 
File: model.py

Handles model training, feature selection, and data preparation for NBA game predictions.
    - Defines training feature sets
    - Converts data into feature matrices
    - Train Logisitic Regression and Random Forest models
    - Return trained models and scalers
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# WINDOW is how many past games to use for rolling averages (with min_periods=3)
WINDOW = 10
FEATURES = ["netRatingDiff",
            "b2bDiff",
            "closeGame",
            "netRating_b2b"]

def load_data():
    """
    Load Games.csv and TeamStatistic.csv
    Build rolling/team-based features
    returns prepared game-level dataframe used for modeling
    """
    
    games = pd.read_csv("data/Games.csv", low_memory=False)
    stats = pd.read_csv("data/TeamStatistics.csv", low_memory=False)

    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
    stats["gameDateTimeEst"] = pd.to_datetime(stats["gameDateTimeEst"])

    # Model is only trained on regular season games because preseason/postseason data is not always predictive of regular season performance
    games = games[games["gameType"] == "Regular Season"].copy()
    stats = stats[stats["gameId"].isin(games["gameId"])].copy()

    # Calculate estimated possessions per game
    # Basic Possession Formula=0.96*[(Field Goal Attempts)+(Turnovers)+0.44*(Free Throw Attempts)-(Offensive Rebounds)]
    # Source: https://www.nbastuffer.com/analytics101/possession/
    stats["possessions"] = (
        0.96 * (stats["fieldGoalsAttempted"] - stats["reboundsOffensive"]
        + stats["turnovers"] + 0.44 * stats["freeThrowsAttempted"])
    ).replace(0, np.nan)

    # Calculate offensive rating, defensive rating, and net rating
    stats["offRating"] = (stats["teamScore"] / stats["possessions"]) * 100
    stats["defRating"] = (stats["opponentScore"] / stats["possessions"]) * 100
    stats["netRating"] = stats["offRating"] - stats["defRating"]

    # Sort the dataframe by chronological order for each team
    stats.sort_values(["teamId", "gameDateTimeEst"], inplace=True)
    # Reset index after sorting to keep it clean
    stats.reset_index(drop=True, inplace=True)

    # For each team, compute a rolling average of the shifted values over a window of size WINDOW
    # - Uses only previous games (because of shift)
    # - Requires at least 3 prior games to produce a value (min_periods=3)
    # - Stores result in a new column like "roll_offRating", "roll_defRating", etc.
    for col in ["offRating", "defRating", "netRating"]:
        # this shift prevents data leakage by ensuring we only use stats from previous games
        shifted = stats.groupby("teamId")[col].shift(1)
        stats[f"roll_{col}"] = (
            shifted.groupby(stats["teamId"])
                .transform(lambda s: s.rolling(WINDOW, min_periods=3).mean())
        )

    # Calculate rest days and back-to-back status
    stats["prevGameDate"] = stats.groupby("teamId")["gameDateTimeEst"].shift(1)
    stats["restDays"] = (
        (stats["gameDateTimeEst"] - stats["prevGameDate"]).dt.total_seconds() / 86400
    ).fillna(3)
    stats["isB2B"] = (stats["restDays"] < 1.5).astype(int)

    # Separate home and away stats, then merge on gameId to get one row per game with both teams' stats
    home_df = stats[stats["home"] == 1][["gameId", "teamId", "roll_offRating", "roll_defRating", "roll_netRating", "restDays", "isB2B", "win"]].copy()
    away_df = stats[stats["home"] == 0][["gameId", "teamId", "roll_offRating", "roll_defRating", "roll_netRating", "restDays", "isB2B"]].copy()
    home_df.columns = ["gameId", "homeTeamId", "home_offRtg", "home_defRtg", "home_netRtg", "home_rest", "home_b2b", "homeWin"]
    away_df.columns = ["gameId", "awayTeamId", "away_offRtg", "away_defRtg", "away_netRtg", "away_rest", "away_b2b"]

    # Merge home and away data into a single row per game using gameId
    df = home_df.merge(away_df, on="gameId", how="inner")

    # Positive values mean home team has the advantage in that metric
    df["offRatingDiff"] = df["home_offRtg"] - df["away_offRtg"]
    df["defRatingDiff"] = df["home_defRtg"] - df["away_defRtg"]
    df["netRatingDiff"] = df["home_netRtg"] - df["away_netRtg"]
    df["restDiff"]      = df["home_rest"]   - df["away_rest"]
    df["b2bDiff"]       = df["home_b2b"]   - df["away_b2b"]
    
    #interaction features
    df["netRating_b2b"] = df["netRatingDiff"] * df["b2bDiff"]
    df["netRating_rest"] = df["netRatingDiff"] * df["restDiff"]
    df["absNetRatingDiff"] = df["netRatingDiff"].abs()
    df["closeGame"] = (df["absNetRatingDiff"] < 5).astype(int)
    df["close_b2b"] = df["closeGame"] * df["b2bDiff"]
    
    # Remove rows with missing values in the selected feature columns. This ensures clean input for model training
    df.dropna(subset=FEATURES, inplace=True)    
    
    return games, stats, df

def build_team_name_map(games):
    """
    Build current team name map from games CSV
    """
    home_names = games[["hometeamId", "hometeamCity", "hometeamName", "gameDateTimeEst"]].rename(
        columns={"hometeamId": "teamId", "hometeamCity": "teamCity", "hometeamName": "teamName"})
    away_names = games[["awayteamId", "awayteamCity", "awayteamName", "gameDateTimeEst"]].rename(
        columns={"awayteamId": "teamId", "awayteamCity": "teamCity", "awayteamName": "teamName"})
    
    all_names = pd.concat([home_names, away_names])
    all_names["fullName"] = all_names["teamCity"] + " " + all_names["teamName"]
    
    # Most recent name per teamId
    name_map = (
        all_names.sort_values("gameDateTimeEst")
                .groupby("teamId").last()["fullName"]
                .to_dict()
    )
    
    return name_map, all_names

def build_team_list(all_names):
    """
    Team list for selection (active in last 2 seasons)
    """
    latest_game = (
        all_names.sort_values("gameDateTimeEst")
                .groupby("teamId").last()[["gameDateTimeEst", "fullName"]]
                .reset_index()
    )
    team_list = (
        latest_game[latest_game["gameDateTimeEst"] >= "2023-10-01"]
        .sort_values("fullName")
        .reset_index(drop=True)
    )
    
    return team_list
    
def get_feature_matrix(df, feature_cols = FEATURES):
    # Extract feature matrix (X) using the selected feature columns
    # .values converts the DataFrame into a NumPy array for model input
    X = df[feature_cols].values
    # Extract target variable (y): whether the home team won (1) or lost (0)
    y = df["homeWin"].astype(int).values
    return X, y
    
def train_logistic_model(X_train, y_train):
    """
    Initialize logistic regression model:
    penalty="l2": ridge regularization to prevent overfitting
    C=1.0: regularization strength (lower = stronger regularization)
    solver="lbfgs": optimization algorithm
    max_iter=1000: increase iterations to ensure convergence
    random_state=42: ensures reproducibility
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model  = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def train_random_forest_model(X_train, y_train):
    """
    Train random forest and return fitted model
    estimators = 200: more trees will improve stability and reduce variance
    max_depth = 10: limits complexity to prevent overfitting
    min_samples_split=10: avoids overly specific splits
    random_state=42: ensures reproducibility
    """
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def get_rest(stats, team_id, before_date):
    played = stats[
        (stats["teamId"] == team_id) &
        (stats["gameDateTimeEst"] < before_date)
    ]["gameDateTimeEst"]
    
    if played.empty:
        return 3.0
    
    return (before_date - played.max()).total_seconds() / 86400

def get_rolling_stats(stats, team_id, before_date):
    played = stats[
        (stats["teamId"] == team_id) &
        (stats["gameDateTimeEst"] < before_date)
    ].sort_values("gameDateTimeEst")
    
    if len(played) < 3:
        return None
    
    return played[["roll_offRating", "roll_defRating", "roll_netRating"]].iloc[-1]