""" 
File: model.py

Handles model training, feature selection, and data preparation for NBA game predictions.
    - Defines training feature sets
    - Converts data into feature matrices
    - Train Logistic Regression, Random Forest, XGBoost, and ensemble models
    - Return trained models and scalers
"""
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# WINDOW is how many past games to use for rolling averages (with min_periods=3)
WINDOW = 10

# These are the features from our original model - used for evaluation
BASE_FEATURES = [
    "netRatingDiff",
    "b2bDiff",
    "closeGame",
    "netRating_b2b"
]

# map raw stats to home-away difference features
ROLLING_FEATURES = {
    "offRating": "offRatingDiff",
    "defRating": "defRatingDiff",
    "netRating": "netRatingDiff",
    "fieldGoalsPercentage": "fieldGoalPctDiff",
    "threePointersPercentage": "threePointPctDiff",
    "freeThrowsPercentage": "freeThrowPctDiff",
    "reboundsTotal": "reboundsTotalDiff",
    "reboundsOffensive": "offensiveReboundsDiff",
    "reboundsDefensive": "defensiveReboundsDiff",
    "turnovers": "turnoversDiff",
    "assists": "assistsDiff",
    "steals": "stealsDiff",
    "blocks": "blocksDiff",
    "benchPoints": "benchPointsDiff",
    "pointsFastBreak": "fastBreakPointsDiff",
    "pointsInThePaint": "paintPointsDiff",
    "pointsSecondChance": "secondChancePointsDiff",
    "seasonWinPct": "seasonWinPctDiff",
}

# Additional features extracted from csv, used for evaluation
ADDITIONAL_FEATURES = [
    "offRatingDiff",
    "defRatingDiff",
    "restDiff",
    "netRating_rest",
    "absNetRatingDiff",
    "absRating_b2b",
    "close_b2b",
    "fieldGoalPctDiff",
    "threePointPctDiff",
    "freeThrowPctDiff",
    "reboundsTotalDiff",
    "offensiveReboundsDiff",
    "defensiveReboundsDiff",
    "turnoversDiff",
    "assistsDiff",
    "stealsDiff",
    "blocksDiff",
    "benchPointsDiff",
    "fastBreakPointsDiff",
    "paintPointsDiff",
    "secondChancePointsDiff",
    "seasonWinPctDiff",
]

# This is the entire feature set extracted from the teamstats csv
EXPANDED_FEATURES = BASE_FEATURES + [
    feature for feature in ADDITIONAL_FEATURES if feature not in BASE_FEATURES
]

# These are the selected features that the model runs on
FEATURES = [
    "seasonWinPctDiff",
    "absNetRatingDiff",
    "defRatingDiff",
    "fieldGoalPctDiff",
    "benchPointsDiff",
    "turnoversDiff",
    "assistsDiff",
    "freeThrowPctDiff"
]

def load_data():
    """
    Load Games.csv and TeamStatistic.csv
    Build rolling/team-based features
    returns prepared game-level dataframe used for modeling
    """
    
    games = pd.read_csv(DATA_DIR / "Games.csv", low_memory=False)
    stats = pd.read_csv(DATA_DIR / "TeamStatistics.csv", low_memory=False)

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
    stats["seasonGames"] = stats["seasonWins"] + stats["seasonLosses"]
    stats["seasonWinPct"] = (stats["seasonWins"] / stats["seasonGames"].replace(0, np.nan)).fillna(0.5)

    # Sort the dataframe by chronological order for each team
    stats.sort_values(["teamId", "gameDateTimeEst"], inplace=True)
    # Reset index after sorting to keep it clean
    stats.reset_index(drop=True, inplace=True)

    # For each team, compute a rolling average of the shifted values over a window of size WINDOW
    # - Uses only previous games (because of shift)
    # - Requires at least 3 prior games to produce a value (min_periods=3)
    # - Stores result in a new column like "roll_offRating", "roll_defRating", etc.
    for col in ROLLING_FEATURES:
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
    roll_cols = [f"roll_{col}" for col in ROLLING_FEATURES]
    home_cols = ["gameId", "gameDateTimeEst", "teamId", *roll_cols, "restDays", "isB2B", "win"]
    away_cols = ["gameId", "teamId", *roll_cols, "restDays", "isB2B"]
    home_df = stats[stats["home"] == 1][home_cols].copy()
    away_df = stats[stats["home"] == 0][away_cols].copy()
    home_df.rename(
        columns={
            "teamId": "homeTeamId",
            "restDays": "home_rest",
            "isB2B": "home_b2b",
            "win": "homeWin",
            **{f"roll_{col}": f"home_{col}" for col in ROLLING_FEATURES},
        },
        inplace=True,
    )
    away_df.rename(
        columns={
            "teamId": "awayTeamId",
            "restDays": "away_rest",
            "isB2B": "away_b2b",
            **{f"roll_{col}": f"away_{col}" for col in ROLLING_FEATURES},
        },
        inplace=True,
    )

    # Merge home and away data into a single row per game using gameId
    df = home_df.merge(away_df, on="gameId", how="inner")
    df.sort_values("gameDateTimeEst", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Positive values mean home team has the advantage in that metric
    for source_col, diff_col in ROLLING_FEATURES.items():
        df[diff_col] = df[f"home_{source_col}"] - df[f"away_{source_col}"]
    df["restDiff"]      = df["home_rest"]   - df["away_rest"]
    df["b2bDiff"]       = df["home_b2b"]   - df["away_b2b"]
    
    #interaction features
    df["netRating_b2b"] = df["netRatingDiff"] * df["b2bDiff"]
    df["netRating_rest"] = df["netRatingDiff"] * df["restDiff"]
    df["absNetRatingDiff"] = df["netRatingDiff"].abs()
    df["absRating_b2b"] = df["absNetRatingDiff"] * df["b2bDiff"]
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
    
    model  = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def train_random_forest_model(X_train, y_train):
    """
    Train random forest and return fitted model
    estimators = 200: enough trees for stability without making comparisons too slow
    max_depth = 12: limits complexity to reduce overfitting
    min_samples_leaf=5: avoids overly specific leaves
    random_state=42: ensures reproducibility
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    
    return model

def train_xgboost_model(X_train, y_train):
    """
    Train a conservative XGBoost classifier for tabular game features.
    """
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    return model

class SoftVotingEnsemble:
    """
    Average predicted probabilities from fitted models.
    Each member is a tuple of (model, optional_scaler).
    """
    def __init__(self, members):
        self.members = members

    def predict_proba(self, X):
        probabilities = []
        for fitted_model, scaler in self.members:
            X_model = scaler.transform(X) if scaler is not None else X
            probabilities.append(fitted_model.predict_proba(X_model))
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def train_soft_voting_ensemble(log_model, log_scaler, rf_model, xgb_model):
    """
    Build a soft voting ensemble from already fitted base models.
    """
    return SoftVotingEnsemble([
        (log_model, log_scaler),
        (rf_model, None),
        (xgb_model, None),
    ])

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
    
    return played[[f"roll_{col}" for col in ROLLING_FEATURES]].iloc[-1]
