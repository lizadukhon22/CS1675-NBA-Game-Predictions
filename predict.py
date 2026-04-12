"""
NBA Game Predictor — Terminal Interface
Run: python predict.py

Options:
  1. Predict a single game (pick 2 teams + date)
  2. Predict a full season for a team (pick team + season year)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import model
from model import (
    load_data,
    build_team_name_map,
    build_team_list,
    get_feature_matrix,
    train_logistic_model,
    train_random_forest_model,
    get_rest,
    get_rolling_stats
)


print("Loading data and training model...")

games, stats, df = load_data()
name_map, all_names = build_team_name_map(games)
team_list = build_team_list(all_names)

X, y = get_feature_matrix(df)
log_model, log_scaler = train_logistic_model(X, y)
rf_model = train_random_forest_model(X, y)

# to be updated when user chooses model
curr_model = None
curr_scaler = None
curr_model_name = None

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


"""
def predict_game_prob(home_id, away_id, game_date):
    h = get_rolling_stats(stats, home_id, game_date)
    a = get_rolling_stats(stats, away_id, game_date)
    if h is None or a is None:
        return None
    home_rest = get_rest(stats, home_id, game_date)
    away_rest = get_rest(stats, away_id, game_date)
    features = np.array([[
        h["roll_offRating"] - a["roll_offRating"],
        h["roll_defRating"] - a["roll_defRating"],
        h["roll_netRating"] - a["roll_netRating"],
        home_rest - away_rest,
        (1 if home_rest < 1.5 else 0) - (1 if away_rest < 1.5 else 0),
    ]])
    
    if curr_scaler is not None:
        features = curr_scaler.transform(features)
    
    return curr_model.predict_proba(features)[0, 1]
"""
def predict_game_prob(home_id, away_id, game_date):
    h = get_rolling_stats(stats, home_id, game_date)
    a = get_rolling_stats(stats, away_id, game_date)
    if h is None or a is None:
        return None

    home_rest = get_rest(stats, home_id, game_date)
    away_rest = get_rest(stats, away_id, game_date)

    feature_map = {
        "offRatingDiff": h["roll_offRating"] - a["roll_offRating"],
        "defRatingDiff": h["roll_defRating"] - a["roll_defRating"],
        "netRatingDiff": h["roll_netRating"] - a["roll_netRating"],
        "restDiff": home_rest - away_rest,
        "b2bDiff": (1 if home_rest < 1.5 else 0) - (1 if away_rest < 1.5 else 0),
    }

    feature_map["absNetRatingDiff"] = abs(feature_map["netRatingDiff"])
    feature_map["closeGame"] = int(feature_map["absNetRatingDiff"] < 5)
    feature_map["netRating_b2b"] = feature_map["netRatingDiff"] * feature_map["b2bDiff"]
    feature_map["netRating_rest"] = feature_map["netRatingDiff"] * feature_map["restDiff"]
    feature_map["close_b2b"] = feature_map["closeGame"] * feature_map["b2bDiff"]

    features = np.array([[feature_map[f] for f in model.FEATURES]])

    if curr_scaler is not None:
        features = curr_scaler.transform(features)

    return curr_model.predict_proba(features)[0, 1]

# switch between using logisitic regression prediciton
# or random forest prediction
def choose_model():
    global curr_model, curr_scaler, curr_model_name
    print("\nSelect predicition model:")
    print("    1. Logistic Regresion")
    print("    2. Random Forest")
    
    while True:
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            curr_model = log_model
            curr_scaler = log_scaler
            curr_model_name = "Logistic Regression"
            break
        elif choice == "2":
            curr_model = rf_model
            curr_scaler = None
            curr_model_name = "Random Forest"
            break
        else:
            print("\nInvalid input - enter 1 or 2.")

# create graph 
def plot_game_by_game_predictions(game_dates, probs, actual_results, team_name, season_label):
    """
    Line chart of model predictions vs actual results for each game.

    Parameters:
    - game_dates: list of datetime objects
    - probs: list of predicted win probabilities for the selected team
    - actual_results: list of 1 (win) or 0 (loss)
    - team_name: string
    - season_label: string
    """
    predictions = [1 if p >= 0.5 else 0 for p in probs]
    
    plt.figure(figsize=(12,8))

    # Model predicted probability line
    plt.plot(game_dates, probs, label="Predicted Win Probability")
    plt.scatter(game_dates, predictions, marker='x',label="Predicted Result")

    # Actual results (0 or 1)
    plt.scatter(game_dates, actual_results, marker='o', label="Actual Result (1=Win, 0=Loss)")

    plt.title(f"{team_name} {season_label}\nGame-by-Game Predictions vs Actual Results")
    plt.xlabel("Game Date")
    plt.ylabel("Win Probability / Result")

    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()

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
    game_dates = []
    probs = []
    actual_results = []

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
        
        game_dates.append(game_date)
        probs.append(team_prob)
        actual_results.append(1 if team_wins_actual else 0)

        print(f"{game_date.strftime('%Y-%m-%d'):<12} {matchup:<34} {pred_label:>5} {team_prob:>6.1%} {actual_label:>7} {check:>4}")

    total = len(season_games) - skipped
    print("─" * 72)
    print(f"\n--- SEASON SUMMARY: {team['fullName']} {season_label} ---")
    print(f"  Games predicted : {total}  ({skipped} skipped — insufficient early-season data)")
    print(f"  Correct         : {correct} / {total}  ({correct/total:.1%} accuracy)")
    print(f"  Predicted W-L   : {predicted_wins}-{total - predicted_wins}")
    print(f"  Actual W-L      : {actual_wins}-{total - actual_wins}")
    print()

    # =========================================================================
    # BASELINE 1: Always predict the home team wins
    # For each game, predict the selected team wins if they are the home team,
    # otherwise predict they lose. This is the simplest possible baseline —
    # it assumes home court advantage always decides the outcome.
    # =========================================================================
    baseline1_correct = 0
    for game_result, game in zip(actual_results, season_games.itertuples()):
        predicted_team_wins = (game.hometeamId == team_id)
        # if its the home team and they won, then add to correct 
        if predicted_team_wins == bool(game_result):
            baseline1_correct += 1
    baseline1_accuracy = baseline1_correct / total if total > 0 else 0

    # =========================================================================
    # BASELINE 2: Random prediction weighted by the training home win rate
    # home_win_rate is computed from df (all pre-season training games). 
    # For each game, flip a weighted coin: if the selected team is home, predict
    # win with probability = home_win_rate; if away, predict win with probability
    # = 1 - home_win_rate. random_state=42 ensures reproducibility.
    # home team usually wins, but not always... this adds this randomness to the baseline to make it more competitive. A good model should beat this by learning which features indicate when the home team is more or less likely to win.
    # =========================================================================
    home_win_rate = df["homeWin"].mean()
    rng = np.random.default_rng(42)
    baseline2_correct = 0
    for game_result, game in zip(actual_results, season_games.itertuples()):
        p = home_win_rate if game.hometeamId == team_id else (1 - home_win_rate)
        predicted_team_wins = rng.random() < p
        if predicted_team_wins == bool(game_result):
            baseline2_correct += 1
    baseline2_accuracy = baseline2_correct / total if total > 0 else 0

    print(f"--- BASELINE COMPARISONS ---")
    print(f"  Baseline 1 — Always Home (no model)        : {baseline1_correct} / {total}  ({baseline1_accuracy:.1%} accuracy)")
    print(f"  Baseline 2 — Random weighted ({home_win_rate:.0%} home rate) : {baseline2_correct} / {total}  ({baseline2_accuracy:.1%} accuracy)")
    print(f"  Our Model  — {curr_model_name:<29}: {correct} / {total}  ({correct/total:.1%} accuracy)")
    print()

    # graph
    plot_game_by_game_predictions(
        game_dates,
        probs,
        actual_results,
        team["fullName"],
        season_label
    )

def main():
    # allow users to use logistic regression or rf model
    choose_model()

    # Main loop
    while True:
        print("=" * 50)
        print(f"\nCurrent model: {curr_model_name}")
        print("\nWhat would you like to do?")
        print("  1. Predict a single game")
        print("  2. Predict a full season for a team")
        print("  3. Change model")
        print("  4. Quit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            mode_single_game()
        elif choice == "2":
            mode_season()
        elif choice == "3":
            choose_model()
        elif choice == "4":
            break
        else:
            print("  Invalid — enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
