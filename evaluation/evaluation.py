"""
File: evaluation.py

Provides reusable functions for evaluating NBA prediction models.
    - Compute evaluation metrics
    - Generate baseline results
    - Evaluate Logistic Regression, Random Forest, XGBoost, and ensemble models
    - Build tables and CSV-ready result rows
    
"""
import os
import sys

from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import (
    load_data, 
    get_feature_matrix,
    train_logistic_model,
    train_random_forest_model,
    train_xgboost_model,
    train_soft_voting_ensemble,
)

RESULTS_FILE = REPORTS_DIR / "evaluation_results.csv"
IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
UPGRADED_RESULTS_FILE = REPORTS_DIR / "upgraded_evaluation_results.csv"
EXPANDED_IMPORTANCE_FILE = REPORTS_DIR / "expanded_feature_importance.csv"

def load_eval_data(split_ratio=0.8):
    game, stats, df = load_data()  

    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


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
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(file_path):
        df_rows.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_rows.to_csv(file_path, index=False)

def get_baselines(y_train, y_test, training_features=None, notes=""):
    """ 
    builds baseline evaluation rows
    """
    rows = []

    home_win_rate = y_train.mean()
    feature_string = ",".join(training_features) if training_features else ""

    # Baseline 1: always predict home team win
    base1_pred = np.ones_like(y_test)
    base1_prob = np.ones_like(y_test, dtype=float)

    rows.append({
        "model": "Baseline 1 - Always Home Team",
        "features": feature_string,
        "notes": notes,
        **calculate_metrics(y_test, base1_pred, base1_prob)
    })

    # Baseline 2: weighted random using training home-win rate
    rng = np.random.default_rng(42)
    base2_pred = (rng.random(len(y_test)) < home_win_rate).astype(int)
    base2_prob = np.full(len(y_test), home_win_rate, dtype=float)

    rows.append({
        "model": "Baseline 2 - Weighted Random",
        "features": feature_string,
        "notes": notes,
        **calculate_metrics(y_test, base2_pred, base2_prob)
    })

    return rows

def build_evaluation_rows(y_train, y_test, model_outputs, training_features=None, notes="", include_baselines=True):
    """" 
    builds evaluation result rows for baselines and trained models
    """
    rows = []
    if include_baselines:
        rows.extend(get_baselines(
            y_train=y_train,
            y_test=y_test,
            training_features=training_features,
            notes = notes
        ))
    
    feature_string = ",".join(training_features) if training_features else ""

    for model_name, y_pred, y_prob in model_outputs:
        rows.append({
            "model": model_name,
            "features": feature_string,
            "notes": notes,
            **calculate_metrics(y_test, y_pred, y_prob)
        })

    return rows

def build_feature_importance_rows(fitted_model, model_name, training_features, notes=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feature_string = ",".join(training_features)

    if not hasattr(fitted_model, "feature_importances_"):
        return []

    importances = fitted_model.feature_importances_
    importance_rows = []

    for feature, importance in sorted(
        zip(training_features, importances),
        key=lambda x: x[1],
        reverse=True
    ):
        importance_rows.append({
            "timestamp": timestamp,
            "model": model_name,
            "features": feature_string,
            "notes": notes,
            "feature": feature,
            "importance": float(importance)
        })
        
    return importance_rows
    
def evaluate_feature_set(train_df, test_df, training_features, notes ="", include_baselines=True):
    """ 
    runs the model with the given training features and saves items to csv
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
   # Build train/test matrices for this feature set
    X_train, y_train = get_feature_matrix(train_df, training_features)
    X_test, y_test = get_feature_matrix(test_df, training_features)

    # Train Logistic Regression
    log_model, log_scaler = train_logistic_model(X_train, y_train)
    X_test_scaled = log_scaler.transform(X_test)

    log_pred = log_model.predict(X_test_scaled)
    log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    # Train Random Forest
    rf_model = train_random_forest_model(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    # Train XGBoost
    xgb_model = train_xgboost_model(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Train soft voting ensemble from the fitted base models
    ensemble_model = train_soft_voting_ensemble(log_model, log_scaler, rf_model, xgb_model)
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_prob = ensemble_model.predict_proba(X_test)[:, 1]

    model_outputs = [
        ("Logistic Regression", log_pred, log_prob),
        ("Random Forest", rf_pred, rf_prob),
        ("XGBoost", xgb_pred, xgb_prob),
        ("Soft Voting Ensemble", ensemble_pred, ensemble_prob),
    ]

    evaluation_rows = build_evaluation_rows(
        y_train=y_train,
        y_test=y_test,
        model_outputs=model_outputs,
        training_features=training_features,
        notes=notes,
        include_baselines=include_baselines
    )

    # Add metadata shared by all rows from this run
    for row in evaluation_rows:
        row["timestamp"] = timestamp
        row["train_size"] = len(train_df)
        row["test_size"] = len(test_df)
        row["num_features"] = len(training_features)

    importance_rows = []
    importance_rows.extend(build_feature_importance_rows(
        fitted_model=rf_model,
        model_name="Random Forest",
        training_features=training_features,
        notes=notes
    ))
    importance_rows.extend(build_feature_importance_rows(
        fitted_model=xgb_model,
        model_name="XGBoost",
        training_features=training_features,
        notes=notes
    ))

    return evaluation_rows, importance_rows

def plot_results_table(result_rows):
    """
    Display a comparison table across multiple feature sets and models.
    """
    if not result_rows:
        print("No results to display.")
        return

    df_table = pd.DataFrame(result_rows).copy()

    # Rename features column to make table clearer
    if "features" in df_table.columns:
        df_table["feature_set"] = df_table["features"]

    # Keep only columns we want to compare
    cols = ["feature_set", "model", "roc_auc", "f1", "accuracy", "precision", "recall"]
    df_table = df_table[[c for c in cols if c in df_table.columns]]

    # Round numeric columns
    for col in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
        if col in df_table.columns:
            df_table[col] = df_table[col].round(4)

    # Optional: make feature strings easier to read
    if "feature_set" in df_table.columns:
        df_table["feature_set"] = df_table["feature_set"].str.replace(",", ", ")

    # Sort rows so models stay grouped under each feature set
    sort_cols = [c for c in ["feature_set", "model"] if c in df_table.columns]
    if sort_cols:
        df_table = df_table.sort_values(sort_cols).reset_index(drop=True)

    print("\n=== Evaluation Results ===")
    print(df_table.to_string(index=False))

    fig_height = max(3, 0.45 * len(df_table) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    
    ax.set_title(
        f"Evaluation Results",
        pad=10
    )

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center"
    )

     # ----- Dynamic column widths -----
    col_widths = []
    for col in df_table.columns:
        max_len = max(
            df_table[col].astype(str).apply(len).max(),
            len(col)
        )
        col_widths.append(max_len)

    total_width = sum(col_widths)
    col_widths = [w / total_width for w in col_widths]

    for i, width in enumerate(col_widths):
        for (row, col), cell in table.get_celld().items():
            if col == i:
                cell.set_width(width)

    # ----- Styling -----
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
def plot_roc_curve(model, X_test, y_test, model_name="Model", ax = None):
    """
    Plots ROC curve and returns AUC score.

    Parameters:
        model: trained sklearn model (must support predict_proba)
        X_test: test features
        y_test: true labels
        model_name: label for legend
        ax: optional matplotlib axis (for multi-model plots)

    Returns:
        auc_score (float)
    """

    # Get probabilities for positive class
    y_probs = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    # Plot
    if ax is None:
        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], linestyle='--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")

    return auc_score
    
