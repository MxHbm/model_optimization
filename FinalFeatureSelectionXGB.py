# pip install mlxtend threadpoolctl
import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier

# ----------------- Load & prep -----------------
basepath = r"C:\Users\mahu123a\Documents\Data\RandomDataGeneration_Gendreau"
for folder in os.listdir(basepath):
    if folder not in ["RandomData_4_30_30","RandomData_5_40_40","RandomData_3_20_20"]:
        typedata = folder.split("RandomData")[0]
        csv_name = folder + ".csv" 
        data = pd.read_csv(os.path.join(basepath,folder,csv_name))
        data.dropna(inplace=True)
        print(f"opening data: {os.path.join(basepath,folder,csv_name)}")

        drop_cols = ["filename", "Route"]
        labelcol  = "CP Status"

        y = data[labelcol].astype(int)
        X = data.drop(columns=drop_cols + [labelcol])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )


        # ----------------- Config -----------------
        OUTPUT_DIR = f"./fs_output/fs_outputs{typedata}"            # where to save JSON & plots
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        PROJECT_TAG = "xgb_fwd_float_5cv_auc"  # used in filenames
        MAX_THREADS = 5

        # Limit features searched up to this many (keeps SFS runtime sane)
        KMAX = min(30, X_train.shape[1])

        # ----------------- Base model (single-threaded; outer parallelism handles speed) -----------------
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = neg / max(pos, 1)

        base_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=10,
            n_estimators=300,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,            # keep model single-threaded
            verbosity=1,
            tree_method="hist"
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ----------------- Sequential Forward Selection (floating) -----------------
        with threadpool_limits(limits=MAX_THREADS):
            sfs = SFS(
                estimator=base_model,
                k_features=(1, KMAX),
                forward=True,
                floating=True,
                scoring="roc_auc",
                cv=cv,
                n_jobs=MAX_THREADS,
                verbose=2
            )
            sfs = sfs.fit(X_train, y_train)

        # Selected features (final step)
        selected_feats = list(sfs.k_feature_names_)
        print(f"Selected ({len(selected_feats)}): {selected_feats}")

        # ----------------- Save full metric dict as JSON -----------------
        metric_dict = sfs.get_metric_dict()
        # Convert all numpy types and arrays to native python types for JSON
        def make_jsonable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return [make_jsonable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: make_jsonable(v) for k,v in obj.items()}
            return obj

        metric_dict_json = make_jsonable(metric_dict)
        json_path = os.path.join(OUTPUT_DIR, f"{PROJECT_TAG}_sfs_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metric_dict_json, f, indent=2)
        print(f"Saved SFS metrics JSON -> {json_path}")

        # ----------------- Refit final model; 5-fold CV AUROC summary -----------------
        with threadpool_limits(limits=MAX_THREADS):
            base_model.fit(X_train[selected_feats], y_train)
            cv_scores = cross_val_score(base_model, X_train[selected_feats], y_train,
                                        scoring="roc_auc", cv=cv, n_jobs=MAX_THREADS)
        print(f"Final (selected) 5-fold CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ----------------- ROC curves on TEST for every SFS step -----------------
        # SFS steps are 1..len(metric_dict). For each, we get the feature names, fit on train, predict proba on test.
        roc_info = []  # list of dicts with 'k', 'features', 'auc', 'fpr', 'tpr'
        with threadpool_limits(limits=MAX_THREADS):
            for k in sorted(metric_dict.keys(), key=int):
                feat_names = list(metric_dict[k]['feature_names'])
                # Fit & predict
                model_k = XGBClassifier(**{**base_model.get_params()})  # clone with same params
                model_k.fit(X_train[feat_names], y_train)
                y_prob = model_k.predict_proba(X_test[feat_names])[:, 1]
                auc_k = roc_auc_score(y_test, y_prob)
                fpr_k, tpr_k, _ = roc_curve(y_test, y_prob)
                roc_info.append({
                    "k": int(k),
                    "features": feat_names,
                    "auc": float(auc_k),
                    "fpr": fpr_k.tolist(),
                    "tpr": tpr_k.tolist()
                })

        # Save all ROC curves & subsets to JSON
        roc_json_path = os.path.join(OUTPUT_DIR, f"{PROJECT_TAG}_test_roc_all_steps.json")
        with open(roc_json_path, "w") as f:
            json.dump(roc_info, f, indent=2)
        print(f"Saved all test ROC curves JSON -> {roc_json_path}")

        # ----------------- Combined ROC plot (all in one; best AUROC emphasized) -----------------
        # Find best by AUROC
        best_idx = int(np.argmax([r["auc"] for r in roc_info]))
        best = roc_info[best_idx]

        plt.figure(figsize=(8, 7))
        # Plot others first (thin & semi-transparent)
        for i, r in enumerate(roc_info):
            if i == best_idx: 
                continue
            plt.plot(r["fpr"], r["tpr"], linewidth=1, alpha=0.4, label=f'k={r["k"]} AUC={r["auc"]:.3f}')
        # Plot best on top (thick line)
        plt.plot(best["fpr"], best["tpr"], linewidth=3, label=f'BEST k={best["k"]} AUC={best["auc"]:.3f}')
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (Test) for All SFS Steps")
        plt.legend(loc="lower right", fontsize=8, ncol=1, frameon=True)
        combined_plot_path = os.path.join(OUTPUT_DIR, f"{PROJECT_TAG}_test_roc_all_steps_combined.png")
        plt.tight_layout()
        plt.savefig(combined_plot_path, dpi=200)
        plt.close()
        print(f"Saved combined ROC plot -> {combined_plot_path}")

        # ----------------- Individual ROC plots per subset -----------------
        for r in roc_info:
            plt.figure(figsize=(5, 4))
            plt.plot(r["fpr"], r["tpr"], linewidth=2, label=f'k={r["k"]} AUC={r["auc"]:.3f}')
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC (Test) | k={r['k']}")
            plt.legend(loc="lower right")
            out_path = os.path.join(OUTPUT_DIR, f"{PROJECT_TAG}_test_roc_k{r['k']:02d}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
        print(f"Saved individual ROC plots in -> {OUTPUT_DIR}")

        # ----------------- OPTIONAL: Subplot grid with BEST in the MIDDLE -----------------
        # If you'd like a grid of small multiples with the best curve centered:
        def subplot_grid_with_center_best(roc_info, grid_cols=3):
            n = len(roc_info)
            rows = math.ceil(n / grid_cols)
            # The "center" index for (rows x cols) grid:
            center_row = rows // 2
            center_col = grid_cols // 2
            # Sort so best goes to center position:
            order = sorted(range(n), key=lambda i: roc_info[i]["auc"])  # ascending
            best_i = int(np.argmax([r["auc"] for r in roc_info]))
            # Move the best index to the middle of the ordered list
            center_pos = center_row * grid_cols + center_col
            if center_pos < len(order):
                order.remove(best_i)
                order.insert(center_pos, best_i)
            else:
                # if grid has more cells than items, we still ensure best is last slot
                order.remove(best_i)
                order.append(best_i)

            # Plot
            fig, axes = plt.subplots(rows, grid_cols, figsize=(grid_cols*3, rows*3))
            axes = np.atleast_2d(axes)
            for ax_i in range(rows * grid_cols):
                r = roc_info[ order[ax_i] ] if ax_i < n else None
                ax = axes[ax_i // grid_cols, ax_i % grid_cols]
                ax.plot([0,1],[0,1], linestyle="--")
                if r is not None:
                    lw = 3 if order[ax_i] == best_i else 1
                    alpha = 1.0 if order[ax_i] == best_i else 0.6
                    ax.plot(r["fpr"], r["tpr"], linewidth=lw, alpha=alpha)
                    ax.set_title(f'k={r["k"]} | AUC={r["auc"]:.3f}', fontsize=9)
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                if (ax_i // grid_cols) == rows-1:
                    ax.set_xlabel("FPR")
                if (ax_i % grid_cols) == 0:
                    ax.set_ylabel("TPR")
            plt.tight_layout()
            grid_path = os.path.join(OUTPUT_DIR, f"{PROJECT_TAG}_test_roc_grid_best_center.png")
            plt.savefig(grid_path, dpi=200)
            plt.close()
            print(f"Saved grid ROC plot (best centered) -> {grid_path}")

        # Call it if you want the grid:
        subplot_grid_with_center_best(roc_info, grid_cols=3)
