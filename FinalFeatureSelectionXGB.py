# pip install xgboost
import os
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer, matthews_corrcoef
from xgboost import XGBClassifier

# ----------------- CONFIG -----------------
INPUT_DIR    = r"C:\Users\mahu123a\Documents\Data\RandomDataGeneration_Gendreau"
OUTPUT_DIR   = os.path.join(os.getcwd(), "FeatureSubsetResultsXGB_JSON")
LABEL_COL    = "CP Status"
DROP_COLS    = ["filename", "Route"]     # columns to drop from features
MAX_FEATURES = 35
MIN_FEATURES = 5
N_SPLITS     = 5
RANDOM_STATE = 42
N_JOBS       = 16                         # CV/SFS parallelism; keep XGB n_jobs=1

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "mcc": make_scorer(matthews_corrcoef)
}

# ----------------- UTIL -----------------
def safe_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    return df

def make_xgb(y: pd.Series) -> XGBClassifier:
    """Create a single-threaded XGBClassifier with imbalance handling."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",           # keeps AUC in training logs; scoring controlled by CV
        max_depth=15,
        n_estimators=300,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=8,                   # IMPORTANT: parallelize at CV/SFS level, not inside model
        verbosity=1
    )

def evaluate_subset(X: pd.DataFrame, y: pd.Series, features: list[str], estimator) -> dict:
    res = cross_validate(
        estimator,
        X[features],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=N_JOBS,
        return_train_score=False,
    )
    out = {}
    for key, arr in res.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        arr = np.asarray(arr, dtype=float)
        out[metric] = {
            "folds": [float(v) for v in arr],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        }
    return out

def select_features_k(X: pd.DataFrame, y: pd.Series, k: int, metric: str, estimator) -> list[str]:
    """
    Sequential forward selection to pick k features optimizing MCC (or other metric if desired).
    """
    scoring_fn = make_scorer(matthews_corrcoef) if metric == "mcc" else metric

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=k,
        direction="forward",
        scoring=scoring_fn,
        cv=cv,
        n_jobs=N_JOBS
    )
    selector.fit(X, y)
    support = selector.get_support(indices=True)
    return list(X.columns[support])

def summarize_best(results_by_k: list[dict]) -> dict:
    best = {}
    for metric in ["accuracy", "f1", "roc_auc", "mcc"]:
        best_entry = max(results_by_k, key=lambda d: d["scores"][metric]["mean"])
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

    # ensure binary labels {0,1}
    unique_y = sorted(y.unique().tolist())
    if len(unique_y) != 2:
        raise ValueError(f"Expecting binary target in {csv_path}; got classes {unique_y}")

    # model (built per-dataset to get correct scale_pos_weight)
    base_estimator = make_xgb(y)

    n_total_features = X.shape[1]
    cap = min(MAX_FEATURES, n_total_features)

    results = []
    for k in range(MIN_FEATURES, cap + 1):
        features_k = select_features_k(X, y, k, "mcc", base_estimator)
        scores_k   = evaluate_subset(X, y, features_k, base_estimator)
        results.append({
            "k": k,
            "features": features_k,
            "scores": scores_k
        })

    # optional summary stats (keep your existing keys)
    mean_average_vol  = float(round(df["Rel Volume"].mean(), 5))  if "Rel Volume"  in df.columns else None
    mean_average_mass = float(round(df["Rel Weight"].mean(), 5))  if "Rel Weight"  in df.columns else None
    pos_share = round(float((y == 1).mean()), 4)
    neg_share = round(1 - pos_share, 4)

    summary = {
        "dataset": os.path.basename(csv_path),
        "n_samples": int(len(df)),
        "model_type": "XGB",
        "Mean_Rel_Volume": mean_average_vol,
        "Mean_Rel_Weight": mean_average_mass,
        "neg_share": neg_share,
        "pos_share": pos_share,
        "n_features_total": int(n_total_features),
        "label_column": LABEL_COL,
        "cv_folds": N_SPLITS,
        "feature_selection": {
            "method": "SequentialForwardSelection",
            "direction": "forward",
            "selection_scoring": "mcc",
            "max_features_considered": cap
        },
        "results_by_k": results,
        "best_by_metric": summarize_best(results)
    }
    return summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for foldername in os.listdir(INPUT_DIR):
        csv_files = sorted(glob.glob(str(Path(INPUT_DIR, foldername) / "*.csv")))
        if not csv_files:
            print(f"No CSV files found in: {Path(INPUT_DIR, foldername)}")
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
