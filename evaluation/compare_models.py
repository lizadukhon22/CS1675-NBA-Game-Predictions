"""
File: compare_models.py

Compares compact and expanded feature sets across all supported models.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluation import (
    EXPANDED_IMPORTANCE_FILE,
    UPGRADED_RESULTS_FILE,
    evaluate_feature_set,
    load_eval_data,
    plot_results_table,
)
from model import BASE_FEATURES, EXPANDED_FEATURES

FEATURE_SETS = [
    ("Base features", BASE_FEATURES),
    ("Expanded rolling features", EXPANDED_FEATURES),
]


def main():
    train_df, test_df = load_eval_data()

    all_rows = []
    expanded_importance_rows = []

    for i, (notes, feature_set) in enumerate(FEATURE_SETS):
        rows, importance_rows = evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            training_features=feature_set,
            notes=notes,
            include_baselines=(i == 0),  # only include baselines once
        )

        all_rows.extend(rows)
        if feature_set == EXPANDED_FEATURES:
            expanded_importance_rows.extend(importance_rows)

    if expanded_importance_rows:
        pd.DataFrame(expanded_importance_rows).to_csv(EXPANDED_IMPORTANCE_FILE, index=False)

    pd.DataFrame(all_rows).to_csv(UPGRADED_RESULTS_FILE, index=False)

    plot_results_table(all_rows)


if __name__ == "__main__":
    main()
