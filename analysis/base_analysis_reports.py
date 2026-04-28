""" 
File: analysis_reports.py

Analyzes evaluation results to determine feature importance and optimal model configurations.
    - Identifies top-performing models based on ROC-AUC
    - Computes feature frequency in top-tier models
    - Performs marginal contribution analysis
    - Evaluates feature redundancy and model stability
    - Generates plots and summary tables
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT_DIR / "reports" / "evaluation_results.csv"
PRIMARY_METRIC = "roc_auc"
MODEL_NAMES = ["Logistic Regression", "Random Forest"]
TOP_TIER_DELTA = 0.002
BASE_FEATURES = [
    "netRatingDiff",
    "b2bDiff"
]

def parse_features (feature_string):
    """ 
    converts comma-separated feature string into a storted list of features
    """
    # handle baseline rows or missing values
    if pd.isna(feature_string) or feature_string == "BASELINE":
        return[]
    
    # split string into individual feature names 
    return sorted([f.strip() for f in str(feature_string).split(",") if f.strip()])

def add_feature_columns (df):
    """
    add parsed feature helpers to the dataframe.
    """
    df = df.copy()
    
    # convert string representation to list
    df["feature_list"] = df["features"].apply(parse_features)
    
    # count number of features in each model 
    df["num_features"] = df["feature_list"].apply(len)

    # create boolean flags for each base feature
    for feature in BASE_FEATURES:
        df[f"has_{feature}"] = df["feature_list"].apply(lambda feats: feature in feats)

    return df

def load_data():
    df = pd.read_csv(RESULTS_FILE)

    df = df[df["model"].isin(MODEL_NAMES).copy()] # exclude baseline rows
    df = add_feature_columns(df) 
    return df

def get_top_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all rows within TOP_TIER_DELTA of the best PRIMARY_METRIC score.
    """

    best_score= df[PRIMARY_METRIC].max()    # best score across all models
    cutoff = best_score - TOP_TIER_DELTA    # cutoff range
    top_tier = df[df[PRIMARY_METRIC] >= cutoff].copy() # filter by models in cutoff

    print("\n=== TOP TIER SUMMARY ===")
    print(f"Best {PRIMARY_METRIC}: {best_score:.4f}")
    print(f"Top-tier cutoff:      {cutoff:.4f}")
    print(f"Top-tier rows:        {len(top_tier)}")

    return top_tier.sort_values([PRIMARY_METRIC, "f1"], ascending=[False, False])

def print_top_models(df: pd.DataFrame, n: int = 15) -> None:
    """
    Print top n models by ROC-AUC, then F1.
    """
    cols = ["model", "features", "num_features", "roc_auc", "f1", "accuracy", "precision", "recall"]
    top = df.sort_values(["roc_auc", "f1"], ascending=[False, False]).head(n)

    print("\n=== TOP MODELS ===")
    print(top[cols].to_string(index=False))

def roc_vs_num_features(df):
    plt.figure()

    for model in df["model"].unique():
        subset = df[df["model"] == model]
        plt.scatter(subset["num_features"], subset["roc_auc"], label=model)

    plt.xlabel("Number of Features")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC vs Number of Features")
    plt.legend()
    plt.show()


def feature_frequency_top_tier(top_tier: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which features appear msot in top-tier models
    """
    
    # count item frequency across top models
    counter = Counter()
    for feats in top_tier["feature_list"]:
        for f in feats:
            counter[f] += 1

    # convert counts into percentage
    freq_df = pd.DataFrame(
        [{"feature": f, "count": c} for f, c in counter.items()]
    ).sort_values("count", ascending=False)

    freq_df["pct_top_tier"] = freq_df["count"] / len(top_tier)

    print("\n=== FEATURE FREQUENCY IN TOP TIER ===")
    print(freq_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(freq_df["feature"][::-1], freq_df["count"][::-1])
    plt.xlabel("Count in Top-Tier Models")
    plt.title("Feature Frequency in Top-Tier Models")
    plt.tight_layout()
    plt.show()

    return freq_df

def marginal_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures the impact of each feature on model performance by comparing
    the average performance (ROC-AUC and F1) of models that includes a given
    features versus those that do not
    
    A larger positive difference indicates that the feature significantly improves
    model performance, while small or negative differences suggest limited usefullness
    """
    rows = []

    for feature in BASE_FEATURES:
        # split dataset into madels that include vs exclude feature
        with_feature = df[df[f"has_{feature}"]]
        without_feature = df[~df[f"has_{feature}"]]

        # compute average performance for both
        row = {
            "feature": feature,
            "with_count": len(with_feature),
            "without_count": len(without_feature),
            f"with_avg_{PRIMARY_METRIC}": with_feature[PRIMARY_METRIC].mean(),
            f"without_avg_{PRIMARY_METRIC}": without_feature[PRIMARY_METRIC].mean(),
            "metric_gain": with_feature[PRIMARY_METRIC].mean() - without_feature[PRIMARY_METRIC].mean(),
            "with_avg_f1": with_feature["f1"].mean(),
            "without_avg_f1": without_feature["f1"].mean(),
            "f1_gain": with_feature["f1"].mean() - without_feature["f1"].mean(),
        }
        rows.append(row)

    contrib_df = pd.DataFrame(rows).sort_values("metric_gain", ascending=False)

    print("\n=== MARGINAL CONTRIBUTION ANALYSIS ===")
    print(contrib_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(contrib_df["feature"][::-1], contrib_df["metric_gain"][::-1])
    plt.xlabel(f"Average {PRIMARY_METRIC.upper()} Gain (WITH - WITHOUT)")
    plt.title("Marginal Contribution by Feature")
    plt.tight_layout()
    plt.show()

    return contrib_df

def best_by_feature_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines how model performance changes as the number of features increases.
    Groups models by number of features and records the best-performing models
    in each group
    
    Helps identify the optimal level of model complexity by showing whether 
    adding more features leads to meaningful improvements
    """
    rows = []

    for num in sorted(df["num_features"].unique()):
        subset = df[df["num_features"] == num]  # all models with this num features
        best_row = subset.sort_values([PRIMARY_METRIC, "f1"], ascending=[False, False]).iloc[0]     # select best performers

        rows.append({
            "num_features": num,
            "model": best_row["model"],
            "features": best_row["features"],
            "roc_auc": best_row["roc_auc"],
            "f1": best_row["f1"],
            "accuracy": best_row["accuracy"],
        })

    best_count_df = pd.DataFrame(rows)

    print("\n=== BEST MODEL BY FEATURE COUNT ===")
    print(best_count_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.plot(best_count_df["num_features"], best_count_df["roc_auc"], marker="o")
    plt.xlabel("Number of Features")
    plt.ylabel("Best ROC-AUC")
    plt.title("Best ROC-AUC by Feature Count")
    plt.tight_layout()
    plt.show()

    return best_count_df


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Logistic vs Random Forest on the same feature sets.
    """
    # reshape data so each feature set has both model scores side-by-side
    pivot = df.pivot_table(
        index="features",
        columns="model",
        values=PRIMARY_METRIC
    ).dropna()

    # computer difference in performance
    pivot["rf_minus_log"] = pivot["Random Forest"] - pivot["Logistic Regression"]

    print("\n=== LOGISTIC VS RANDOM FOREST ===")
    print(pivot.sort_values("rf_minus_log", ascending=False).head(10).to_string())
    print("\nMost Logistic-favored feature sets:")
    print(pivot.sort_values("rf_minus_log", ascending=True).head(10).to_string())

    plt.figure(figsize=(6, 6))
    plt.scatter(pivot["Logistic Regression"], pivot["Random Forest"])
    min_val = min(pivot["Logistic Regression"].min(), pivot["Random Forest"].min())
    max_val = max(pivot["Logistic Regression"].max(), pivot["Random Forest"].max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Logistic Regression ROC-AUC")
    plt.ylabel("Random Forest ROC-AUC")
    plt.title("Model Comparison by Feature Set")
    plt.tight_layout()
    plt.show()

    return pivot

def stability_analysis(top_tier: pd.DataFrame) -> None:
    """
    Assesses the consisitency of feature importance across top models
    
    Consistent patterns across top models indicate reliable feature importance,
    while high variability suggests weaker or interchangeable features
    """
    print("\n=== STABILITY ANALYSIS ===")

    all_feature_sets = top_tier["feature_list"].tolist()
    feature_counter = Counter()

    for feats in all_feature_sets:
        for f in feats:
            feature_counter[f] += 1

    print("Top-tier feature counts:")
    for feature, count in feature_counter.most_common():
        print(f"  {feature}: {count}/{len(top_tier)}")

    print("\nTop-tier model counts:")
    print(top_tier["model"].value_counts().to_string())

    print("\nTop-tier feature-count distribution:")
    print(top_tier["num_features"].value_counts().sort_index().to_string()) 
    
def main():
    df = load_data()

    print_top_models(df, n=20)

    top_tier = get_top_tier(df)

    feature_frequency_top_tier(top_tier)
    marginal_contribution(df)
    best_by_feature_count(df)
    model_comparison(df)
    stability_analysis(top_tier)


if __name__ == "__main__":
    main()
