import os
from datetime import datetime

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from model import (
    load_data, 
    get_feature_matrix,
    train_logistic_model,
    train_random_forest_model
)

ALL_FEATURES = ["offRatingDiff", "defRatingDiff", "netRatingDiff", "restDiff", "b2bDiff"] # baseline
TRAINING_FEATURES = None # set to None if you want to evaluate all combos, manual for specific features
EXPERIMENT_NOTES = "net & rest only" # change for each evaluation 
RESULTS_FILE = "evaluation_results.csv"
IMPORTANCE_FILE = "feature_importance.csv"


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics and return them in a dictionary.
    """
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def save_rows_to_csv(rows, file_path):
    """
    Append rows to a CSV file. Create the file with headers if it does not exist.
    """
    df_rows = pd.DataFrame(rows)

    if os.path.exists(file_path):
        df_rows.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_rows.to_csv(file_path, index=False)
        
def baseline_exists(file_path):
    """  
    Check if baseline evaluation exists in the file
    """
    if not os.path.exists(file_path):
        return False

    df_existing = pd.read_csv(file_path)

    if "model" not in df_existing.columns:
        return False

    baseline_models = {
        "Baseline 1 - Always Home Team",
        "Baseline 2 - Weighted Random"
    }

    existing_models = set(df_existing["model"].dropna().unique())
    return baseline_models.issubset(existing_models)

def save_baselines(y_train, y_test, train_df, test_df):
    """ 
    adds baseline evaluation to csv file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    home_win_rate = y_train.mean()

    base1_pred = np.ones_like(y_test)
    base1_prob = np.ones_like(y_test, dtype=float)

    rng = np.random.default_rng(42)
    base2_pred = (rng.random(len(y_test)) < home_win_rate).astype(int)
    base2_prob = np.full(len(y_test), home_win_rate, dtype=float)

    baseline_rows = []

    baseline_outputs = [
        ("Baseline 1 - Always Home Team", base1_pred, base1_prob),
        ("Baseline 2 - Weighted Random", base2_pred, base2_prob),
    ]

    for model_name, y_pred, y_prob in baseline_outputs:
        metrics = calculate_metrics(y_test, y_pred, y_prob)

        baseline_rows.append({
            "timestamp": timestamp,
            "model": model_name,
            "features": "BASELINE",
            "notes": "baseline",
            "train_size": len(train_df),
            "test_size": len(test_df),
            "home_win_rate_train": float(home_win_rate),
            **metrics
        })

    save_rows_to_csv(baseline_rows, RESULTS_FILE)
    
def get_feature_sets():
    """ 
    gets all combinations of current features to evaluate
    or manual feature set and notes
    """
    feature_sets = [] 
    
    if TRAINING_FEATURES is None:
        sorted_all_features = sorted(ALL_FEATURES)

        for r in range(1, len(sorted_all_features) + 1):
            for combo in combinations(sorted_all_features, r):
                feature_sets.append(list(combo))

        notes = "all combinations"
    else:
        feature_sets.append(sorted(TRAINING_FEATURES))
        notes = EXPERIMENT_NOTES

    return feature_sets, notes

def save_feature_importance(rf_model, training_features, notes):
    """ 
    evaluates feature importance and saves to csv
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feature_string = ",".join(training_features)

    importances = rf_model.feature_importances_
    importance_rows = []

    for feature, importance in sorted(
        zip(training_features, importances),
        key=lambda x: x[1],
        reverse=True
    ):
        importance_rows.append({
            "timestamp": timestamp,
            "model": "Random Forest",
            "features": feature_string,
            "notes": notes,
            "feature": feature,
            "importance": float(importance)
        })

    save_rows_to_csv(importance_rows, IMPORTANCE_FILE)

def run_models(train_df, test_df, training_features, y_train, y_test, notes):
    """ 
    runs the model with the given training features and saves items to csv
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feature_string = ",".join(training_features)
    home_win_rate = y_train.mean()

    X_train, _ = get_feature_matrix(train_df, training_features)
    X_test, _ = get_feature_matrix(test_df, training_features)

    # Train models  
    log_model, log_scaler = train_logistic_model(X_train, y_train)
    rf_model = train_random_forest_model(X_train, y_train)
    
    # Logistic predictions
    X_test_scaled = log_scaler.transform(X_test)
    log_pred = log_model.predict(X_test_scaled)
    log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    # Random forest predictions
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    result_rows = []

    model_outputs = [
        ("Logistic Regression", log_pred, log_prob),
        ("Random Forest", rf_pred, rf_prob),
    ]

    for model_name, y_pred, y_prob in model_outputs:
        metrics = calculate_metrics(y_test, y_pred, y_prob)

        result_rows.append({
            "timestamp": timestamp,
            "model": model_name,
            "features": feature_string,
            "notes": notes,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "home_win_rate_train": float(home_win_rate),
            **metrics
        })

    save_rows_to_csv(result_rows, RESULTS_FILE)
    save_feature_importance(rf_model, training_features, notes)
    
def main():
    # Load and split data
    games, stats, df = load_data()  

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    _, y_train = get_feature_matrix(train_df, ALL_FEATURES)
    _, y_test = get_feature_matrix(test_df, ALL_FEATURES)
    
    feature_sets, notes = get_feature_sets()
    
    existing_feature_sets = set()
    if os.path.exists(RESULTS_FILE):
        existing_results = pd.read_csv(RESULTS_FILE)
        if "features" in existing_results.columns:
            existing_feature_sets = set(existing_results["features"].dropna().unique())

    if not os.path.exists(RESULTS_FILE):
        save_baselines(y_train, y_test, train_df, test_df)

    for training_features in feature_sets:
        feature_string = ",".join(training_features)

        if feature_string in existing_feature_sets:
            print(f"Skipping existing feature set: {feature_string}")
            continue

        run_models(
            train_df,
            test_df,
            training_features,
            y_train,
            y_test,
            notes
        )

        existing_feature_sets.add(feature_string)
        print(f"Saved results for: {feature_string}")

    print(f"Saved experiment results to {RESULTS_FILE}")
    print(f"Saved feature importance to {IMPORTANCE_FILE}")


if __name__ == "__main__":
    main()