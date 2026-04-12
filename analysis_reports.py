import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

RESULTS_FILE = "reports/evaluation_results.csv"
PRIMARY_METRIC = "roc_auc"
MODEL_NAMES = ["Logistic Regression", "Random Forest"]
TOP_TIER_DELTA = 0.002
BASE_FEATURES = [
    "offRatingDiff",
    "defRatingDiff",
    "netRatingDiff",
    "restDiff",
    "b2bDiff"
]

def parse_features (feature_string):
    """ 
    converts comma-separated feature string into a storted list of features
    """
    if pd.isna(feature_string) or feature_string == "BASELINE":
        return[]
    return sorted([f.strip() for f in str(feature_string).split(",") if f.strip()])

def add_feature_columns (df):
    """
    Add parsed feature helpers to the dataframe.
    """
    df = df.copy()
    df["feature_list"] = df["features"].apply(parse_features)
    df["num_features"] = df["feature_list"].apply(len)

    for feature in BASE_FEATURES:
        df[f"has_{feature}"] = df["feature_list"].apply(lambda feats: feature in feats)

    return df

def load_data():
    df = pd.read_csv(RESULTS_FILE)

    # keep only real models
    df = df[df["model"].isin(MODEL_NAMES).copy()]
    df["num_features"] = df["features"].apply(lambda x:len(x.split(",")))
    df = add_feature_columns(df)
    return df

def get_top_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all rows within TOP_TIER_DELTA of the best PRIMARY_METRIC score.
    """
    best_score = df[PRIMARY_METRIC].max()
    cutoff = best_score - TOP_TIER_DELTA
    top_tier = df[df[PRIMARY_METRIC] >= cutoff].copy()

    print("\n=== TOP TIER SUMMARY ===")
    print(f"Best {PRIMARY_METRIC}: {best_score:.4f}")
    print(f"Top-tier cutoff:      {cutoff:.4f}")
    print(f"Top-tier rows:        {len(top_tier)}")

    return top_tier.sort_values([PRIMARY_METRIC, "f1"], ascending=[False, False])

def print_top_models(df: pd.DataFrame, n: int = 15) -> None:
    """
    Print top N models by ROC-AUC, then F1.
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
    Count how often each feature appears in top-tier models.
    """
    counter = Counter()
    for feats in top_tier["feature_list"]:
        for f in feats:
            counter[f] += 1

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
    Compare average model performance WITH vs WITHOUT each feature.
    """
    rows = []

    for feature in BASE_FEATURES:
        with_feature = df[df[f"has_{feature}"]]
        without_feature = df[~df[f"has_{feature}"]]

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
    Show the best model for each feature count.
    """
    rows = []

    for num in sorted(df["num_features"].unique()):
        subset = df[df["num_features"] == num]
        best_row = subset.sort_values([PRIMARY_METRIC, "f1"], ascending=[False, False]).iloc[0]

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
    pivot = df.pivot_table(
        index="features",
        columns="model",
        values=PRIMARY_METRIC
    ).dropna()

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

def redundancy_checks(df: pd.DataFrame) -> None:
    """
    Compare a few specific subsets to reason about redundancy among off/def/net.
    """
    print("\n=== REDUNDANCY CHECKS ===")

    checks = [
        ["netRatingDiff"],
        ["offRatingDiff", "defRatingDiff"],
        ["netRatingDiff", "offRatingDiff"],
        ["netRatingDiff", "defRatingDiff"],
        ["offRatingDiff", "defRatingDiff", "netRatingDiff"],
    ]

    df = df.copy()
    df["features_sorted_string"] = df["feature_list"].apply(lambda feats: ",".join(sorted(feats)))

    for feature_set in checks:
        key = ",".join(sorted(feature_set))
        subset = df[df["features_sorted_string"] == key]

        if subset.empty:
            print(f"{key}: not present")
            continue

        best = subset.sort_values([PRIMARY_METRIC, "f1"], ascending=[False, False]).iloc[0]
        print(
            f"{key:<45} "
            f"best model={best['model']:<20} "
            f"roc_auc={best['roc_auc']:.4f} "
            f"f1={best['f1']:.4f}"
        )

def stability_analysis(top_tier: pd.DataFrame) -> None:
    """
    Look at how similar the top-tier models are.
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
  
def export_top_tier(top_tier: pd.DataFrame, out_file: str = "reports/top_tier_models.csv") -> None:
    """
    Save the top-tier models for easy review in Excel.
    """
    cols = ["model", "features", "num_features", "roc_auc", "f1", "accuracy", "precision", "recall"]
    top_tier[cols].to_csv(out_file, index=False)
    print(f"\nSaved top-tier models to {out_file}")  
    
    
def find_rf_better(df):
    pivot = df.pivot_table(
        index = "features",
        columns = "model", 
        values = "roc_auc"
    ).dropna()
    
    pivot["roc_auc_diff"] = pivot["Random Forest"] - pivot["Logistic Regression"]
    
    rf_better = pivot[pivot["roc_auc_diff"] > 0]
    
    print("\nFeature sets where Random Forest outperforms Logistic:\n")
    print(rf_better.sort_values("roc_auc_diff", ascending=False).head(10))
    
    return rf_better

def main():
    df = load_data()

    print_top_models(df, n=20)

    top_tier = get_top_tier(df)
    export_top_tier(top_tier)

    feature_frequency_top_tier(top_tier)
    marginal_contribution(df)
    best_by_feature_count(df)
    model_comparison(df)
    redundancy_checks(df)
    stability_analysis(top_tier)


if __name__ == "__main__":
    main()