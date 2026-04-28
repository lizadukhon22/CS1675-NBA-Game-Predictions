"""
File: upgraded_analysis_report.py

Analyzes the clean upgraded model comparison outputs:
    - Base vs expanded feature performance
    - Metric gains from expanded features
    - Random Forest and XGBoost feature importance
    - Combined normalized tree importance
    - Feature-group importance
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluation import EXPANDED_IMPORTANCE_FILE, UPGRADED_RESULTS_FILE

METRICS = ["roc_auc", "f1", "accuracy", "precision", "recall"]
BASE_NOTE = "Base features"
EXPANDED_NOTE = "Expanded rolling features"
TREE_MODELS = ["Random Forest", "XGBoost"]
FEATURE_SELECTION_RESULTS_FILE = ROOT_DIR / "reports" / "feature_selection_results.csv"


def show_plot():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def require_report_files():
    missing = [
        path for path in [UPGRADED_RESULTS_FILE, EXPANDED_IMPORTANCE_FILE]
        if not Path(path).exists()
    ]
    if missing:
        print("Missing required report file(s):")
        for path in missing:
            print(f"  {path}")
        print("\nRun this first:")
        print("  python compare_models.py")
        return False
    return True


def load_reports():
    results = pd.read_csv(UPGRADED_RESULTS_FILE)
    importance = pd.read_csv(EXPANDED_IMPORTANCE_FILE)
    return results, importance


def print_base_vs_expanded(results):
    cols = ["notes", "model", *METRICS]
    df = results[results["notes"].isin([BASE_NOTE, EXPANDED_NOTE])].copy()
    df = df[cols].sort_values(["notes", "model"])

    print("\n=== BASE VS EXPANDED MODEL PERFORMANCE ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def plot_metric_by_model(performance_df, metric="roc_auc"):
    model_df = performance_df[
        performance_df["model"].isin(["Logistic Regression", "Random Forest", "XGBoost", "Soft Voting Ensemble"])
    ].copy()
    pivot = model_df.pivot(index="model", columns="notes", values=metric)
    pivot = pivot[[BASE_NOTE, EXPANDED_NOTE]]

    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel(metric.upper())
    plt.title(f"Base vs Expanded {metric.upper()} by Model")
    plt.xticks(rotation=25, ha="right")
    show_plot()

    return pivot


def expanded_improvement(results):
    model_rows = results[
        results["notes"].isin([BASE_NOTE, EXPANDED_NOTE])
        & results["model"].isin(["Logistic Regression", "Random Forest", "XGBoost", "Soft Voting Ensemble"])
    ].copy()

    base = model_rows[model_rows["notes"] == BASE_NOTE].set_index("model")
    expanded = model_rows[model_rows["notes"] == EXPANDED_NOTE].set_index("model")
    common_models = base.index.intersection(expanded.index)

    rows = []
    for model_name in common_models:
        row = {"model": model_name}
        for metric in METRICS:
            row[f"{metric}_gain"] = expanded.loc[model_name, metric] - base.loc[model_name, metric]
        rows.append(row)

    gains = pd.DataFrame(rows).sort_values("roc_auc_gain", ascending=False)

    print("\n=== EXPANDED FEATURE GAINS (EXPANDED - BASE) ===")
    print(gains.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plt.figure(figsize=(9, 5))
    plt.barh(gains["model"][::-1], gains["roc_auc_gain"][::-1])
    plt.xlabel("ROC-AUC Gain")
    plt.title("ROC-AUC Gain from Expanded Features")
    show_plot()

    return gains


def top_feature_importance(importance, model_name, top_n=15):
    df = importance[importance["model"] == model_name].copy()
    df = df.sort_values("importance", ascending=False).head(top_n)

    print(f"\n=== TOP {top_n} {model_name.upper()} FEATURE IMPORTANCES ===")
    print(df[["feature", "importance"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plt.figure(figsize=(9, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} {model_name} Feature Importances")
    show_plot()

    return df


def combined_feature_importance(importance):
    df = importance[importance["model"].isin(TREE_MODELS)].copy()
    df["model_total"] = df.groupby("model")["importance"].transform("sum")
    df["normalized_importance"] = df["importance"] / df["model_total"]

    combined = (
        df.pivot_table(index="feature", columns="model", values="normalized_importance", fill_value=0)
        .reset_index()
    )
    for model_name in TREE_MODELS:
        if model_name not in combined:
            combined[model_name] = 0

    combined["combined_importance"] = combined[TREE_MODELS].mean(axis=1)
    combined = combined.sort_values("combined_importance", ascending=False)

    print("\n=== COMBINED NORMALIZED TREE FEATURE IMPORTANCE ===")
    print(
        combined[["feature", *TREE_MODELS, "combined_importance"]]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    top = combined.head(15)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"][::-1], top["combined_importance"][::-1])
    plt.xlabel("Average Normalized Importance")
    plt.title("Combined RF + XGBoost Feature Importance")
    show_plot()

    return combined


def feature_group(feature):
    if feature in ["offRatingDiff", "defRatingDiff", "netRatingDiff", "absNetRatingDiff"]:
        return "rating"
    if feature in ["restDiff", "b2bDiff"]:
        return "schedule/rest"
    if "Pct" in feature or feature in ["fieldGoalPctDiff", "threePointPctDiff", "freeThrowPctDiff"]:
        return "shooting"
    if "Rebounds" in feature or "rebounds" in feature:
        return "rebounding"
    if feature == "assistsDiff":
        return "ball movement"
    if feature in ["stealsDiff", "blocksDiff"]:
        return "defense"
    if feature in ["benchPointsDiff", "fastBreakPointsDiff", "paintPointsDiff", "secondChancePointsDiff"]:
        return "scoring style"
    if feature == "seasonWinPctDiff":
        return "season record"
    if feature in ["netRating_b2b", "netRating_rest", "absRating_b2b", "closeGame", "close_b2b"]:
        return "interaction"
    return "other"


def feature_group_importance(importance):
    df = importance[importance["model"].isin(TREE_MODELS)].copy()
    df["group"] = df["feature"].apply(feature_group)
    group_df = (
        df.groupby(["model", "group"], as_index=False)["importance"]
        .sum()
        .sort_values(["model", "importance"], ascending=[True, False])
    )

    print("\n=== FEATURE GROUP IMPORTANCE ===")
    print(group_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pivot = group_df.pivot(index="group", columns="model", values="importance").fillna(0)
    pivot = pivot.sort_values(by=TREE_MODELS[0], ascending=False)
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Total Importance")
    plt.title("Feature Importance by Basketball Category")
    plt.xticks(rotation=30, ha="right")
    show_plot()

    return group_df


def load_feature_selection_results():
    if not Path(FEATURE_SELECTION_RESULTS_FILE).exists():
        print("\nFeature selection results not found.")
        print("Run this to create them:")
        print("  python feature_selection.py")
        return None

    return pd.read_csv(FEATURE_SELECTION_RESULTS_FILE)


def feature_selection_summary(selection_results):
    if selection_results is None or selection_results.empty:
        return

    model_rows = selection_results[
        selection_results["model"].isin(["Logistic Regression", "Random Forest", "XGBoost", "Soft Voting Ensemble"])
    ].copy()
    ranked_rows = model_rows[model_rows["selection_type"] == "ranked_subset"].copy()
    if model_rows.empty:
        return

    best_auc = model_rows["roc_auc"].max()
    best_by_auc = model_rows.sort_values(["roc_auc", "f1"], ascending=[False, False]).iloc[0]
    best_by_f1 = model_rows.sort_values(["f1", "roc_auc"], ascending=[False, False]).iloc[0]
    smallest_near_best = (
        model_rows[model_rows["roc_auc"] >= best_auc - 0.002]
        .sort_values(["num_features", "roc_auc"], ascending=[True, False])
        .iloc[0]
    )

    print("\n=== FEATURE SELECTION SUMMARY ===")
    print("\nBest subset by ROC-AUC:")
    print(best_by_auc[["subset_name", "model", "num_features", "roc_auc", "f1", "accuracy"]].to_string())

    print("\nBest subset by F1:")
    print(best_by_f1[["subset_name", "model", "num_features", "roc_auc", "f1", "accuracy"]].to_string())

    print("\nSmallest subset within 0.002 ROC-AUC of best:")
    print(
        smallest_near_best[
            ["selection_type", "subset_name", "model", "num_features", "roc_auc", "f1", "accuracy"]
        ].to_string()
    )

    print("\nTop feature-selection results:")
    print(
        model_rows.sort_values(["roc_auc", "f1"], ascending=[False, False])
        .head(12)[["selection_type", "subset_name", "model", "num_features", "roc_auc", "f1", "accuracy"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    if not ranked_rows.empty:
        plot_feature_selection_metric(ranked_rows, "roc_auc")
        plot_feature_selection_metric(ranked_rows, "f1")
    print_top_selected_features(smallest_near_best)


def plot_feature_selection_metric(ranked_rows, metric):
    pivot = ranked_rows.pivot_table(
        index="num_features",
        columns="model",
        values=metric,
        aggfunc="max",
    ).sort_index()

    pivot.plot(marker="o", figsize=(9, 5))
    plt.xlabel("Number of Features")
    plt.ylabel(metric.upper())
    plt.title(f"Feature Count vs {metric.upper()}")
    show_plot()


def print_top_selected_features(row):
    features = [feature for feature in str(row["selected_features"]).split(",") if feature]
    print("\nRecommended selected features:")
    for feature in features:
        print(f"  {feature}")

    plt.figure(figsize=(8, max(3, len(features) * 0.28)))
    plt.barh(features[::-1], list(range(len(features), 0, -1))[::-1])
    plt.xlabel("Rank Position")
    plt.title("Recommended Feature Subset")
    show_plot()


def main():
    if not require_report_files():
        return

    results, importance = load_reports()

    performance_df = print_base_vs_expanded(results)
    plot_metric_by_model(performance_df, metric="roc_auc")
    expanded_improvement(results)

    for model_name in TREE_MODELS:
        top_feature_importance(importance, model_name)

    combined_feature_importance(importance)
    feature_group_importance(importance)
    feature_selection_summary(load_feature_selection_results())


if __name__ == "__main__":
    main()
