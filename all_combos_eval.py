"""
File: all_combos_eval.py

Runs exhaustive evaluation across all possible feature combinations.
    - Generates all non-empty feature subsets
    - Skips combinations already evaluated in the results CSV
    - Evaluates each feature set using shared evaluation logic
    - Saves evaluation results and Random Forest feature importance
"""

import os
from itertools import combinations

import pandas as pd

from evaluation import (
    load_eval_data,
    evaluate_feature_set,
    save_rows_to_csv,
    RESULTS_FILE,
    IMPORTANCE_FILE,
)

ALL_FEATURES = [
    "offRatingDiff",
    "defRatingDiff",
    "netRatingDiff",
    "restDiff",
    "b2bDiff",
    "netRating_b2b",
    "netRating_rest",
    "absNetRatingDiff",
    "closeGame",
    "close_b2b"
]

EXPERIMENT_NOTES = "all-combos evaluation"


def generate_feature_sets(all_features):
    """
    Generate all non-empty feature combinations from the provided list.
    """
    sorted_features = sorted(all_features)
    feature_sets = []

    for r in range(1, len(sorted_features) + 1):
        for combo in combinations(sorted_features, r):
            feature_sets.append(list(combo))

    return feature_sets


def get_existing_feature_sets(results_file):
    """
    Read the existing results CSV and return a set of feature strings
    that have already been evaluated.
    """
    if not os.path.exists(results_file):
        return set()

    df_existing = pd.read_csv(results_file)

    if "features" not in df_existing.columns:
        return set()

    # Ignore baseline rows if they exist
    existing = set(
        df_existing.loc[df_existing["features"] != "BASELINE", "features"]
        .dropna()
        .unique()
    )

    return existing


def main():
    # Load and split data once
    train_df, test_df = load_eval_data()

    # Generate all possible feature combinations
    feature_sets = generate_feature_sets(ALL_FEATURES)
    print(f"Generated {len(feature_sets)} feature combinations.")

    # Read previously evaluated combinations so we can skip duplicates
    existing_feature_sets = get_existing_feature_sets(RESULTS_FILE)

    completed = 0
    skipped = 0

    for training_features in feature_sets:
        feature_string = ",".join(training_features)

        if feature_string in existing_feature_sets:
            print(f"Skipping existing feature set: {feature_string}")
            skipped += 1
            continue

        print(f"Evaluating: {feature_string}")

        evaluation_rows, importance_rows = evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            training_features=training_features,
            notes=EXPERIMENT_NOTES,
            include_baselines=False,  # baselines do not depend on feature set
        )

        save_rows_to_csv(evaluation_rows, RESULTS_FILE)
        save_rows_to_csv(importance_rows, IMPORTANCE_FILE)

        existing_feature_sets.add(feature_string)
        completed += 1

    print("\n=== ALL-COMBOS RUN COMPLETE ===")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")


if __name__ == "__main__":
    main()