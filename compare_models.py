"""
File: compare_models.py

Compares a selected set of feature combinations using shared evaluation logic.
    - Manually define feature sets
    - Evaluate each feature set against baselines
    - Display summary tables for model comparison
"""

from evaluation import load_eval_data, evaluate_feature_set, plot_results_table

FEATURE_SETS = [
    ["offRatingDiff", "defRatingDiff", "netRatingDiff", "restDiff", "b2bDiff"],
    ["netRatingDiff", "b2bDiff"]
]

train_df, test_df = load_eval_data()

all_rows = []

for i, feature_set in enumerate(FEATURE_SETS):
    rows, importance_rows = evaluate_feature_set(
        train_df=train_df,
        test_df=test_df,
        training_features=feature_set,
        include_baselines= (i==len(FEATURE_SETS)-1) #only include baselines once
    )

    all_rows.extend(rows)

plot_results_table(all_rows)