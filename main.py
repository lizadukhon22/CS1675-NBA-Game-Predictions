"""
NBA Game Predictor — Terminal Interface
Run: python predict_game.py

Options:
  1. Predict a single game (pick 2 teams + date)
  2. Predict a full season for a team (pick team + season year)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# WINDOW is how many past games to use for rolling averages (with min_periods=3)
WINDOW = 10
FEATURES = ["offRatingDiff", "defRatingDiff", "netRatingDiff", "restDiff", "b2bDiff"]

# Train model
print("Loading data and training model...")

games = pd.read_csv("Games.csv", low_memory=False)
stats = pd.read_csv("TeamStatistics.csv", low_memory=False)

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

# Remove rows with missing values in the selected feature columns. This ensures clean input for model training
df.dropna(subset=FEATURES, inplace=True)

# Start the model training !
# Extract feature matrix (X) using the selected feature columns
# .values converts the DataFrame into a NumPy array for model input
X = df[FEATURES].values
# Extract target variable (y): whether the home team won (1) or lost (0)
y = df["homeWin"].astype(int).values
scaler = StandardScaler()

# Initialize logistic regression model:
# - penalty="l2": ridge regularization to prevent overfitting
# - C=1.0: regularization strength (lower = stronger regularization)
# - solver="lbfgs": optimization algorithm
# - max_iter=1000: increase iterations to ensure convergence
# - random_state=42: ensures reproducibility
model  = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
model.fit(scaler.fit_transform(X), y)

# Build current team name map from games CSV (most recent name per teamId) 
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

# Team list for selection (active in last 2 seasons)
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

print("Model ready.\n")

# Shared helpers 
def pick_team(prompt, exclude_id=None):
    teams = team_list[team_list["teamId"] != exclude_id].reset_index(drop=True)
    print(prompt)
    for i, row in teams.iterrows():
        print(f"  {i+1:2}. {row['fullName']}")
    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(teams):
                return teams.iloc[choice - 1]
        except ValueError:
            pass
        print("  Invalid — try again.")

def get_rest(team_id, before_date):
    played = stats[
        (stats["teamId"] == team_id) &
        (stats["gameDateTimeEst"] < before_date)
    ]["gameDateTimeEst"]
    if played.empty:
        return 3.0
    return (before_date - played.max()).total_seconds() / 86400

def get_rolling_stats(team_id, before_date):
    played = stats[
        (stats["teamId"] == team_id) &
        (stats["gameDateTimeEst"] < before_date)
    ].sort_values("gameDateTimeEst")
    if len(played) < 3:
        return None
    return played[["roll_offRating", "roll_defRating", "roll_netRating"]].iloc[-1]

def predict_game_prob(home_id, away_id, game_date):
    h = get_rolling_stats(home_id, game_date)
    a = get_rolling_stats(away_id, game_date)
    if h is None or a is None:
        return None
    home_rest = get_rest(home_id, game_date)
    away_rest = get_rest(away_id, game_date)
    features = np.array([[
        h["roll_offRating"] - a["roll_offRating"],
        h["roll_defRating"] - a["roll_defRating"],
        h["roll_netRating"] - a["roll_netRating"],
        home_rest - away_rest,
        (1 if home_rest < 1.5 else 0) - (1 if away_rest < 1.5 else 0),
    ]])
    return model.predict_proba(scaler.transform(features))[0, 1]

# Mode 1: Single game
def mode_single_game():
    home_team = pick_team("\nSelect HOME team:")
    away_team = pick_team("\nSelect AWAY team:", exclude_id=home_team["teamId"])

    while True:
        date_str = input("\nGame date (YYYY-MM-DD): ").strip()
        try:
            game_date = pd.Timestamp(date_str)
            break
        except Exception:
            print("  Invalid date — try again.")

    prob = predict_game_prob(home_team["teamId"], away_team["teamId"], game_date)

    if prob is None:
        print("  Not enough historical data for one or both teams before that date.")
        return

    winner   = home_team["fullName"] if prob >= 0.5 else away_team["fullName"]
    win_prob = prob if prob >= 0.5 else 1 - prob

    print(f"""
--- PREDICTION ---
  {home_team['fullName']} (Home) vs {away_team['fullName']} (Away)
  Home win probability : {prob:.1%}
  Away win probability : {1-prob:.1%}
  Predicted winner     : {winner} ({win_prob:.1%})
""")

# Mode 2: Full season 
def mode_season():
    team = pick_team("\nSelect team:")
    team_id = team["teamId"]

    # Find seasons where this team played
    team_games_all = games[
        (games["hometeamId"] == team_id) |
        (games["awayteamId"] == team_id)
    ].copy()

    # NBA season year = the calendar year the season ends (Oct-Jun)
    team_games_all["season"] = team_games_all["gameDateTimeEst"].apply(
        lambda d: d.year if d.month >= 10 else d.year
    )
    available_seasons = sorted(team_games_all["season"].unique())[-10:]

    print("\nAvailable seasons:")
    for i, s in enumerate(available_seasons):
        print(f"  {i+1:2}. {s-1}-{str(s)[2:]} season")

    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(available_seasons):
                season_end_year = available_seasons[choice - 1]
                break
        except ValueError:
            pass
        print("  Invalid — try again.")

    season_start = pd.Timestamp(f"{season_end_year - 1}-10-01")
    season_end   = pd.Timestamp(f"{season_end_year}-06-30")

    season_games = games[
        (games["gameDateTimeEst"] >= season_start) &
        (games["gameDateTimeEst"] <= season_end)   &
        ((games["hometeamId"] == team_id) | (games["awayteamId"] == team_id))
    ].sort_values("gameDateTimeEst").copy()

    if season_games.empty:
        print("  No games found for that season.")
        return

    season_label = f"{season_end_year-1}-{str(season_end_year)[2:]}"
    print(f"\nPredicting {len(season_games)} games for {team['fullName']} in the {season_label} season...\n")

    print(f"{'Date':<12} {'Matchup':<34} {'Pred':>5} {'Prob':>7} {'Actual':>7} {'':>3}")
    print("─" * 72)

    correct        = 0
    skipped        = 0
    predicted_wins = 0
    actual_wins    = 0

    for _, game in season_games.iterrows():
        home_id   = game["hometeamId"]
        away_id   = game["awayteamId"]
        game_date = game["gameDateTimeEst"]
        actual_winner_id = game["winner"]

        prob = predict_game_prob(home_id, away_id, game_date)
        if prob is None:
            skipped += 1
            continue

        # Use current names for display
        home_name = name_map.get(home_id, str(home_id))
        away_name = name_map.get(away_id, str(away_id))
        matchup   = f"{home_name} vs {away_name}"

        pred_winner_id   = home_id if prob >= 0.5 else away_id
        team_wins_pred   = pred_winner_id   == team_id
        team_wins_actual = actual_winner_id == team_id

        pred_label   = "W" if team_wins_pred   else "L"
        actual_label = "W" if team_wins_actual else "L"
        team_prob    = prob if home_id == team_id else 1 - prob
        check        = "✓" if pred_winner_id == actual_winner_id else "✗"

        if pred_winner_id == actual_winner_id:
            correct += 1
        if team_wins_pred:
            predicted_wins += 1
        if team_wins_actual:
            actual_wins += 1

        print(f"{game_date.strftime('%Y-%m-%d'):<12} {matchup:<34} {pred_label:>5} {team_prob:>6.1%} {actual_label:>7} {check:>4}")

    total = len(season_games) - skipped
    print("─" * 72)
    print(f"\n--- SEASON SUMMARY: {team['fullName']} {season_label} ---")
    print(f"  Games predicted : {total}  ({skipped} skipped — insufficient early-season data)")
    print(f"  Correct         : {correct} / {total}  ({correct/total:.1%} accuracy)")
    print(f"  Predicted W-L   : {predicted_wins}-{total - predicted_wins}")
    print(f"  Actual W-L      : {actual_wins}-{total - actual_wins}")
    print()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    print("=" * 50)
    print("\nWhat would you like to do?")
    print("  1. Predict a single game")
    print("  2. Predict a full season for a team")
    print("  3. Quit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        mode_single_game()
    elif choice == "2":
        mode_season()
    elif choice == "3":
        break
    else:
        print("  Invalid — enter 1, 2, or 3.")