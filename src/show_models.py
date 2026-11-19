"""
PathSnap: Inspect Trained Model Architectures and Metrics
Run via terminal: python src/show_models.py
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
COMPARISON_CSV = os.path.join(BASE_DIR, "results", "model_comparison.csv")
VITERBI_CSV = os.path.join(BASE_DIR, "results", "route_reconstruction", "viterbi_comparison_results.csv")

FEATURES = [
    'cand_distance_m', 'heading_diff_deg', 'cand_rank', 'dist_ratio_to_min',
    'dist_diff_to_min', 'gps_speed_kmh', 'gps_step_dist_m', 'time_diff_s',
    'road_length_m', 'gps_lon', 'gps_lat', 'road_heading', 'gps_heading'
]

def main():
    print("=" * 86)
    print("         PATHSNAP: TRAINED MACHINE LEARNING MODEL ARCHITECTURES & METRICS")
    print("=" * 86)

    model_files = [
        ("logistic_regression.joblib", "Logistic Regression (Linear)"),
        ("k_nearest_neighbors.joblib", "K-Nearest Neighbors (Instance-Based)"),
        ("decision_tree.joblib", "Decision Tree (CART)"),
        ("random_forest.joblib", "Random Forest (Bagged Ensemble)"),
        ("xgboost.joblib", "XGBoost (Baseline Gradient Boosted Trees)"),
        ("final_model.joblib", "Tuned XGBoost (Optuna Bayesian Optimized)")
    ]

    for fname, desc in model_files:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            continue

        model = joblib.load(path)
        print(f"\n[MODEL] {desc}")
        print(f"  Artifact File : models/{fname}")
        print(f"  Class Type    : {type(model).__name__}")
        print(f"  Input Features: {getattr(model, 'n_features_in_', 13)} dimensions")

        # Specific Architecture Details
        if fname == "logistic_regression.joblib":
            print(f"  Solver        : {model.solver} | C: {model.C}")
            print(f"  Intercept (b) : {model.intercept_[0]:.4f}")
            print("  Top Learned Weights (w):")
            coefs = sorted(zip(FEATURES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
            for f, c in coefs[:5]:
                print(f"    - {f:18}: {c:+.4f}")

        elif fname == "k_nearest_neighbors.joblib":
            print(f"  Neighbors (k) : {model.n_neighbors} | Metric: {model.metric} (p={model.p})")
            print(f"  Weights       : {model.weights} | Algorithm: {model.algorithm} | Leaf Size: {model.leaf_size}")

        elif fname == "decision_tree.joblib":
            print(f"  Criterion     : {model.criterion} | Max Depth: {model.tree_.max_depth}")
            print(f"  Total Nodes   : {model.tree_.node_count:,} | Leaves: {model.get_n_leaves():,}")

        elif fname == "random_forest.joblib":
            print(f"  Ensemble Size : {model.n_estimators} Trees | Max Depth: {model.max_depth}")
            print(f"  Criterion     : {model.criterion} | Max Features: {model.max_features}")
            print("  Top Feature Importances (Gini):")
            imps = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
            for f, imp in imps[:5]:
                print(f"    - {f:18}: {imp*100:5.2f}%")

        elif fname in ("xgboost.joblib", "final_model.joblib"):
            params = model.get_params()
            print(f"  Objective     : {params.get('objective')}")
            print(f"  Num Estimators: {params.get('n_estimators')} Trees | Max Depth: {params.get('max_depth')}")
            print(f"  Learning Rate : {params.get('learning_rate')} | Subsample: {params.get('subsample')}")
            print(f"  Colsample     : {params.get('colsample_bytree')} | L1: {params.get('reg_alpha')} | L2: {params.get('reg_lambda')}")
            print("  Top Feature Importances (Gain):")
            imps = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
            for f, imp in imps[:5]:
                print(f"    - {f:18}: {imp*100:5.2f}%")

    print("\n" + "=" * 86)
    print("              EMPIRICAL 5-MODEL BENCHMARK PERFORMANCE COMPARISON")
    print("=" * 86)
    if os.path.exists(COMPARISON_CSV):
        df_comp = pd.read_csv(COMPARISON_CSV)
        cols = ['Model', 'CV Macro F1', 'Student-Test Macro F1', 'Top-1 Accuracy', 'Route Continuity', 'Log Loss', 'Training Time (s)', 'Inference Time (s)']
        sub_df = df_comp[[c for c in cols if c in df_comp.columns]]
        print(sub_df.to_string(index=False))

    print("\n" + "=" * 86)
    print("        SEQUENTIAL ROUTE RECONSTRUCTION (HMM + LOG-SPACE VITERBI)")
    print("=" * 86)
    if os.path.exists(VITERBI_CSV):
        df_vit = pd.read_csv(VITERBI_CSV)
        cols_vit = ['trip_id', 'n_obs', 'clf_top1_acc', 'hmm_top1_acc', 'clf_continuity', 'hmm_continuity', 'hmm_route_overlap', 'exec_time_s']
        sub_vit = df_vit[[c for c in cols_vit if c in df_vit.columns]].copy()
        # Format percentages
        for col in ['clf_top1_acc', 'hmm_top1_acc', 'clf_continuity', 'hmm_continuity', 'hmm_route_overlap']:
            if col in sub_vit.columns:
                sub_vit[col] = sub_vit[col].apply(lambda x: f"{x*100:5.2f}%")
        if 'exec_time_s' in sub_vit.columns:
            sub_vit['exec_time_ms'] = (sub_vit['exec_time_s'] * 1000).apply(lambda x: f"{x:5.1f} ms")
            sub_vit.drop(columns=['exec_time_s'], inplace=True)
        print(sub_vit.to_string(index=False))

    print("\n" + "=" * 86)

if __name__ == "__main__":
    main()
