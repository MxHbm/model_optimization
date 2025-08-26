import os
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------- CONFIG -----------------
INPUT_DIR   = r"C:\Users\mahu123a\Documents\Data\RandomDataGeneration_Gendreau"
OUTPUT_DIR  = os.path.join(os.getcwd(),"FeatureSubsetResultsLR_JSON")
LABEL_COL   = "CP Status"
DROP_COLS   = ["filename", "Route"]           # columns to drop from features
MAX_FEATURES = 30                              # cap
MIN_FEATURES = 5
N_SPLITS     = 5                               # 5-fold CV
RANDOM_STATE = 42

# If your positive class is 1 (as in your code), nothing to change here.

# ----------------- MODEL & CV -----------------
# Balanced logistic regression; scaling generally helps LR.
base_estimator = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ))
])

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def binary_auc(y_true, y_score):
    # y_score can be 1D or 2D; take positive class if 2D
    if hasattr(y_score, "shape") and len(y_score.shape) == 2:
        y_score = y_score[:, 1]
    return roc_auc_score(y_true, y_score)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1":       make_scorer(f1_score),
    "roc_auc":  make_scorer(binary_auc, needs_proba=True),
    "mcc":      make_scorer(matthews_corrcoef)
}



# ----------------- UTIL -----------------
def safe_colnames(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnamed columns, strip spaces, etc. (optional but handy)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    return df

def evaluate_subset(X: pd.DataFrame, y: pd.Series, features: list[str]) -> dict:
    """Return mean & std for each metric over CV using only 'features'."""
    res = cross_validate(
        base_estimator,
        X[features],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    out = {}
    for key, arr in res.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        out[metric] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}
    return out

def select_features_k(X: pd.DataFrame, y: pd.Series, k: int, metric:str) -> list[str]:
    """
    Sequential forward selection to pick k features that maximize ROC AUC under CV.
    We select using ROC AUC because it’s robust to imbalance; you’ll still get
    Accuracy/F1/ROC AUC reported for the chosen subset.
    """
    if metric == "mcc":
        scoring_fn = make_scorer(matthews_corrcoef)
    else:
        scoring_fn = metric  # use sklearn built-in string scorers

    selector = SequentialFeatureSelector(
        base_estimator,
        n_features_to_select=k,
        direction="forward",
        scoring=scoring_fn,
        cv=cv,
        n_jobs=-1
    )
    selector.fit(X, y)
    support = selector.get_support(indices=True)
    return list(X.columns[support])

def summarize_best(results_by_k: list[dict]) -> dict:
    """
    From the list of result entries (each with k, scores, features),
    find the best (highest mean) per metric.
    """
    best = {}
    for metric in ["accuracy", "f1", "roc_auc"]:
        best_entry = max(
            results_by_k,
            key=lambda d: d["scores"][metric]["mean"]
        )
        best[metric] = {
            "k": best_entry["k"],
            "mean": best_entry["scores"][metric]["mean"],
            "std": best_entry["scores"][metric]["std"],
            "features": best_entry["features"]
        }
    return best

# ----------------- MAIN LOOP -----------------
def process_dataset(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = safe_colnames(df).dropna()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in {csv_path}")

    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL] + [c for c in DROP_COLS if c in df.columns])

    # defensive: ensure binary labels {0,1}
    unique_y = sorted(y.unique().tolist())
    if len(unique_y) != 2:
        raise ValueError(f"Expecting binary target in {csv_path}; got classes {unique_y}")

    n_total_features = X.shape[1]
    cap = min(MAX_FEATURES, n_total_features)

    results = []
    for k in range(MIN_FEATURES, cap + 1):
        features_k = select_features_k(X, y, k, "mcc")
        scores_k   = evaluate_subset(X, y, features_k)
        results.append({
            "k": k,
            "features": features_k,
            "scores": scores_k
        })

    summary = {
        "dataset": os.path.basename(csv_path),
        "n_samples": int(len(df)),
        "n_features_total": int(n_total_features),
        "label_column": LABEL_COL,
        "cv_folds": N_SPLITS,
        "feature_selection": {
            "method": "SequentialForwardSelection",
            "direction": "forward",
            "selection_scoring": "roc_auc",
            "max_features_considered": cap
        },
        "results_by_k": results,
        "best_by_metric": summarize_best(results)
    }
    return summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for foldername in os.listdir(INPUT_DIR):
        csv_files = sorted(glob.glob(str(Path(INPUT_DIR,foldername) / "*.csv")))
        if not csv_files:
            print(f"No CSV files found in: {INPUT_DIR}")
            continue

        for csv_path in csv_files:

            try:
                print(f"Processing: {csv_path}")
                summary = process_dataset(csv_path)

                out_name = os.path.splitext(os.path.basename(csv_path))[0] + "_feature_subsets.json"
                out_path = str(Path(OUTPUT_DIR) / out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"[ERROR] {csv_path}: {e}")

if __name__ == "__main__":
    main()
