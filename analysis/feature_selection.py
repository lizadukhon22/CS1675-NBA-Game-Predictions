"""
File: feature_selection.py

Focused feature selection for the expanded NBA prediction model.
    - Ranks features using normalized Random Forest + XGBoost importance
    - Evaluates top-N ranked feature subsets
    - Runs one-feature ablation on the best subset
    - Saves results to reports/feature_selection_results.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluation import IMPORTANCE_FILE, evaluate_feature_set, load_eval_data
from model import EXPANDED_FEATURES

FEATURE_SELECTION_RESULTS_FILE = ROOT_DIR / "reports" / "feature_selection_results.csv"
TREE_MODELS = ["Random Forest", "XGBoost"]
SUBSET_SIZES = [4, 6, 8, 10, 12, 15, 20, len(EXPANDED_FEATURES)]
PRIMARY_METRIC = "roc_auc"
SECONDARY_METRIC = "f1"


def load_combined_importance():
    """
    Load feature importance values from the saved model comparison report
    This function combined RF and XGBoost feature importance for analysis, 
    and includes normalized  importances into a combined rating
    """
    if not Path(IMPORTANCE_FILE).exists():
        raise FileNotFoundError(
            f"Missing {IMPORTANCE_FILE}. Run python compare_models.py first."
        )

    importance = pd.read_csv(IMPORTANCE_FILE)
    importance = importance[importance["model"].isin(TREE_MODELS)].copy()

    importance["model_total"] = importance.groupby("model")["importance"].transform("sum")
    importance["normalized_importance"] = importance["importance"] / importance["model_total"]

    combined = (
        importance.pivot_table(
            index="feature",
            columns="model",
            values="normalized_importance",
            fill_value=0,
        )
        .reset_index()
    )
    for model_name in TREE_MODELS:
        if model_name not in combined:
            combined[model_name] = 0

    combined["combined_importance"] = combined[TREE_MODELS].mean(axis=1)

    ranked = (
        combined[combined["feature"].isin(EXPANDED_FEATURES)]
        .sort_values("combined_importance", ascending=False)
        .reset_index(drop=True)
    )

    # Preserve any expanded features absent from the importance file at the end.
    missing_features = [feature for feature in EXPANDED_FEATURES if feature not in set(ranked["feature"])]
    if missing_features:
        ranked = pd.concat(
            [
                ranked,
                pd.DataFrame(
                    {
                        "feature": missing_features,
                        "Random Forest": 0.0,
                        "XGBoost": 0.0,
                        "combined_importance": 0.0,
                    }
                ),
            ],
            ignore_index=True,
        )

    print("\n=== COMBINED FEATURE RANKING ===")
    print(ranked[["feature", "combined_importance"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return ranked


def feature_set_string(features):
    
    return ",".join(features)


def add_selection_metadata(rows, selection_type, subset_name, feature_list):
    """ 
    Add identifying metadata to evaluation results
    """
    feature_string = feature_set_string(feature_list)
    for row in rows:
        row["selection_type"] = selection_type
        row["subset_name"] = subset_name
        row["selected_features"] = feature_string
    return rows


def evaluate_ranked_subsets(train_df, test_df, ranked_features):
    """
    Evaluate progressively larger subsets of the ranked feature list.

    The goal is to test whether a smaller number of high-importance features can
    perform as well as, or better than, the full expanded feature set.

    For example, it evaluates the top 4, top 6, top 8, etc. features based on
    the combined RF + XGBoost importance ranking.
    """
    all_rows = []
    evaluated_sizes = set()

    for size in SUBSET_SIZES:
        actual_size = min(size, len(ranked_features))
        if actual_size in evaluated_sizes:
            continue
        evaluated_sizes.add(actual_size)

        subset_features = ranked_features[:actual_size]
        subset_name = f"top_{actual_size}"

        print(f"\nEvaluating ranked subset: {subset_name}")
        rows, _ = evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            training_features=subset_features,
            notes=f"feature selection - {subset_name}",
            include_baselines=False,
        )
        all_rows.extend(add_selection_metadata(rows, "ranked_subset", subset_name, subset_features))

    return all_rows


def choose_best_subset(selection_rows):
    """
    Choose the best-performing feature subset from the ranked-subset experiments.

    The best subset is selected first by the primary metric, ROC-AUC, and then by
    the secondary metric, F1, as a tiebreaker.

    Returns:
        best_row: The result row for the best model/subset combination.
        best_features: List of features used in that best subset.
    """
    results = pd.DataFrame(selection_rows)
    model_rows = results[
        results["model"].isin(["Logistic Regression", "Random Forest", "XGBoost", "Soft Voting Ensemble"])
    ].copy()

    best_row = model_rows.sort_values([PRIMARY_METRIC, SECONDARY_METRIC], ascending=[False, False]).iloc[0]
    best_features = [feature for feature in best_row["selected_features"].split(",") if feature]

    print("\n=== BEST RANKED SUBSET ===")
    print(
        best_row[
            ["subset_name", "model", "num_features", PRIMARY_METRIC, SECONDARY_METRIC, "accuracy"]
        ].to_string()
    )

    return best_row, best_features


def evaluate_ablation(train_df, test_df, best_features, best_subset_name):
    """
    Run one-feature ablation tests on the best selected feature subset.

    Each test removes one feature from the best subset and re-evaluates the models.
    If performance drops after removing a feature, that suggests the removed
    feature was useful.

    Returns:
        rows containing evaluation results for each ablated subset.
    """
    if len(best_features) <= 1:
        return []

    rows = []
    for removed_feature in best_features:
        ablated_features = [feature for feature in best_features if feature != removed_feature]
        subset_name = f"{best_subset_name}_minus_{removed_feature}"

        print(f"\nAblation: removing {removed_feature}")
        eval_rows, _ = evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            training_features=ablated_features,
            notes=f"feature selection ablation - remove {removed_feature}",
            include_baselines=False,
        )
        for row in eval_rows:
            row["removed_feature"] = removed_feature
        rows.extend(add_selection_metadata(eval_rows, "ablation", subset_name, ablated_features))

    return rows


def redundancy_checks(train_df, test_df):
    """
    Evaluate hand-picked feature groups that may contain overlapping information.

    These checks test whether certain groups of related features are redundant.
    For example, this compares net rating alone against offensive and defensive
    rating together, and total rebounds against offensive/defensive rebounds.
    """
    checks = [
        ("net_rating_only", ["netRatingDiff"]),
        ("off_def_rating_only", ["offRatingDiff", "defRatingDiff"]),
        ("net_plus_off_def_rating", ["netRatingDiff", "offRatingDiff", "defRatingDiff"]),
        ("total_rebounds_only", ["reboundsTotalDiff"]),
        ("off_def_rebounds_only", ["offensiveReboundsDiff", "defensiveReboundsDiff"]),
        ("all_rebounds", ["reboundsTotalDiff", "offensiveReboundsDiff", "defensiveReboundsDiff"]),
        ("base_interactions", ["closeGame", "netRating_b2b"]),
        ("expanded_interactions", ["closeGame", "netRating_b2b", "close_b2b", "absRating_b2b", "netRating_rest"]),
    ]

    rows = []
    for subset_name, features in checks:
        valid_features = [feature for feature in features if feature in EXPANDED_FEATURES]
        if not valid_features:
            continue

        print(f"\nRedundancy check: {subset_name}")
        eval_rows, _ = evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            training_features=valid_features,
            notes=f"feature selection redundancy - {subset_name}",
            include_baselines=False,
        )
        rows.extend(add_selection_metadata(eval_rows, "redundancy_check", subset_name, valid_features))

    return rows


def summarize_results(rows):
    """
    Print a summary of the feature-selection experiments.

    This reports:
    - the best-performing subsets by ROC-AUC
    - the smallest subset close to the best result
    - which ablation removals caused the largest performance drop

    The summary helps determine whether the full expanded feature set is needed
    or whether a smaller subset performs nearly as well.
    """
    results = pd.DataFrame(rows)
    model_rows = results[
        results["model"].isin(["Logistic Regression", "Random Forest", "XGBoost", "Soft Voting Ensemble"])
    ].copy()

    best_auc = model_rows[PRIMARY_METRIC].max()
    near_best = model_rows[model_rows[PRIMARY_METRIC] >= best_auc - 0.002].copy()
    smallest_near_best = near_best.sort_values(["num_features", PRIMARY_METRIC], ascending=[True, False]).iloc[0]

    print("\n=== FEATURE SELECTION SUMMARY ===")
    print("\nBest by ROC-AUC:")
    print(
        model_rows.sort_values([PRIMARY_METRIC, SECONDARY_METRIC], ascending=[False, False])
        .head(10)[["selection_type", "subset_name", "model", "num_features", PRIMARY_METRIC, SECONDARY_METRIC, "accuracy"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nSmallest model within 0.002 ROC-AUC of best:")
    print(
        smallest_near_best[
            ["selection_type", "subset_name", "model", "num_features", PRIMARY_METRIC, SECONDARY_METRIC, "accuracy"]
        ].to_string()
    )

    if "removed_feature" in model_rows.columns:
        ablation = model_rows[model_rows["selection_type"] == "ablation"].copy()
        if not ablation.empty:
            best_subset_rows = model_rows[model_rows["selection_type"] == "ranked_subset"]
            reference = best_subset_rows.sort_values(
                [PRIMARY_METRIC, SECONDARY_METRIC], ascending=[False, False]
            ).iloc[0]
            ablation["roc_auc_drop_vs_best"] = reference[PRIMARY_METRIC] - ablation[PRIMARY_METRIC]
            print("\nMost important ablation removals by ROC-AUC drop:")
            print(
                ablation.sort_values("roc_auc_drop_vs_best", ascending=False)
                .head(12)[["removed_feature", "model", "num_features", PRIMARY_METRIC, "roc_auc_drop_vs_best"]]
                .to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )


def main():
    """
    Run the full feature-selection workflow.

    Steps:
    1. Load combined RF + XGBoost feature rankings.
    2. Evaluate top-N ranked feature subsets.
    3. Choose the best subset.
    4. Run ablation tests on that subset.
    5. Run redundancy checks on related feature groups.
    6. Save all results and print a summary.
    """
    ranked = load_combined_importance()
    ranked_features = ranked["feature"].tolist()

    train_df, test_df = load_eval_data()

    rows = []
    rows.extend(evaluate_ranked_subsets(train_df, test_df, ranked_features))

    best_row, best_features = choose_best_subset(rows)
    rows.extend(evaluate_ablation(train_df, test_df, best_features, best_row["subset_name"]))
    rows.extend(redundancy_checks(train_df, test_df))

    results = pd.DataFrame(rows)
    results.to_csv(FEATURE_SELECTION_RESULTS_FILE, index=False)
    print(f"\nSaved feature selection results to {FEATURE_SELECTION_RESULTS_FILE}")

    summarize_results(rows)


if __name__ == "__main__":
    main()
