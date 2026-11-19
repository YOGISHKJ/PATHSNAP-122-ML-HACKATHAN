"""
PathSnap Bayesian Hyperparameter Optimization with Optuna
Performs Bayesian optimization on XGBoost using GroupKFold cross-validation
on the 80 training trips, refits the optimal model, and benchmarks against untuned model.
"""

import os
import sys

# Ensure src is in python path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from xgboost import XGBClassifier

from feature_engineering import FeaturePipeline
from train_models import evaluate_observation_level

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optimization(artifacts_dir='artifacts', models_dir='models', opt_dir='optimization', n_trials=15):
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading candidate datasets for Bayesian Optimization...")
    train_df = pd.read_parquet(os.path.join(artifacts_dir, 'train_candidates.parquet'))
    test_df = pd.read_parquet(os.path.join(artifacts_dir, 'test_candidates.parquet'))
    pipe = FeaturePipeline.load(os.path.join(artifacts_dir, 'feature_pipeline.joblib'))
    
    X_train = pipe.transform(train_df)
    y_train = train_df['label'].values
    groups_train = train_df['trip_id'].values
    
    X_test = pipe.transform(test_df)
    y_test = test_df['label'].values
    
    # 3-Fold GroupKFold CV during optimization on a representative 30-trip subset for high efficiency
    unique_trips = np.unique(groups_train)
    np.random.seed(42)
    search_trips = np.random.choice(unique_trips, size=30, replace=False)
    mask_search = np.isin(groups_train, search_trips)
    X_search = X_train[mask_search]
    y_search = y_train[mask_search]
    groups_search = groups_train[mask_search]
    
    print(f"Search space evaluated on {len(search_trips)} representative trips ({len(X_search):,} candidate samples)...")
    gkf = GroupKFold(n_splits=3)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 60, 160, step=20),
            'max_depth': trial.suggest_int('max_depth', 5, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 5.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }
        
        scores = []
        for tr_idx, val_idx in gkf.split(X_search, y_search, groups=groups_search):
            X_tr, y_tr = X_search[tr_idx], y_search[tr_idx]
            X_val, y_val = X_search[val_idx], y_search[val_idx]
            
            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            
        return float(np.mean(scores))
        
    print(f"Starting Bayesian Optimization study ({n_trials} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    for i in range(n_trials):
        study.optimize(objective, n_trials=1)
        trial = study.trials[-1]
        print(f"  Trial {i+1:02d}/{n_trials:02d} | CV Macro F1: {trial.value:.4f} | Best: {study.best_value:.4f} (Elapsed: {time.time()-t0:.1f}s)")
        
    total_opt_time = float(time.time() - t0)
    print(f"\nOptimization completed in {total_opt_time:.2f} seconds.")
    print(f"Best Trial CV Macro F1: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(opt_dir, 'optuna_study.csv')
    trials_df.to_csv(trials_csv, index=False)
    print(f"Trials history saved to {trials_csv}")
    
    # Save best parameters JSON
    best_params_path = os.path.join(opt_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_cv_macro_f1': study.best_value,
            'optimization_duration_seconds': total_opt_time,
            'n_trials': n_trials,
            'best_parameters': study.best_params
        }, f, indent=2)
    print(f"Best parameters saved to {best_params_path}")
    
    # -------------------------------------------------------------
    # Train Optimal Tuned Model on Full Training Partition
    # -------------------------------------------------------------
    print("\nRefitting Optimal Tuned Model on full training partition...")
    tuned_params = study.best_params.copy()
    tuned_params['random_state'] = 42
    tuned_params['n_jobs'] = -1
    tuned_params['eval_metric'] = 'logloss'
    tuned_params['tree_method'] = 'hist'
    
    tuned_clf = XGBClassifier(**tuned_params)
    t_tr = time.time()
    tuned_clf.fit(X_train, y_train)
    tuned_train_time = float(time.time() - t_tr)
    
    # Evaluate Tuned Model on Student-Testing Partition
    t_inf = time.time()
    y_test_pred_tuned = tuned_clf.predict(X_test)
    tuned_inf_time = float(time.time() - t_inf)
    y_test_probs_tuned = tuned_clf.predict_proba(X_test)[:, 1]
    
    tuned_test_acc = float(accuracy_score(y_test, y_test_pred_tuned))
    tuned_test_f1 = float(f1_score(y_test, y_test_pred_tuned, average='macro', zero_division=0))
    tuned_log_loss = float(log_loss(y_test, y_test_probs_tuned))
    tuned_top1_acc, tuned_continuity, _ = evaluate_observation_level(test_df, y_test_probs_tuned)
    
    # Save Final Tuned Model
    final_model_path = os.path.join(models_dir, 'final_model.joblib')
    joblib.dump(tuned_clf, final_model_path)
    print(f"Final optimal model saved to {final_model_path}")
    
    # -------------------------------------------------------------
    # Compare Untuned vs Tuned Model
    # -------------------------------------------------------------
    # Load untuned XGBoost results from model_comparison.csv
    comp_df = pd.read_csv('results/model_comparison.csv')
    untuned_row = comp_df[comp_df['Model'] == 'XGBoost'].iloc[0]
    
    comparison = pd.DataFrame([
        {
            'Version': 'Untuned XGBoost',
            'CV Macro F1': untuned_row['CV Macro F1'],
            'Student-Test Macro F1': untuned_row['Student-Test Macro F1'],
            'Student-Test Accuracy': untuned_row['Student-Test Accuracy'],
            'Top-1 Accuracy': untuned_row['Top-1 Accuracy'],
            'Route Continuity': untuned_row['Route Continuity'],
            'Log Loss': untuned_row['Log Loss'],
            'Inference Time (s)': untuned_row['Inference Time (s)']
        },
        {
            'Version': 'Tuned XGBoost (Bayesian Opt)',
            'CV Macro F1': study.best_value,
            'Student-Test Macro F1': tuned_test_f1,
            'Student-Test Accuracy': tuned_test_acc,
            'Top-1 Accuracy': tuned_top1_acc,
            'Route Continuity': tuned_continuity,
            'Log Loss': tuned_log_loss,
            'Inference Time (s)': tuned_inf_time
        }
    ])
    
    comp_path = os.path.join(opt_dir, 'tuned_vs_untuned_comparison.csv')
    comparison.to_csv(comp_path, index=False)
    print("\n==================================================")
    print("Tuned vs Untuned Model Comparison:")
    print("==================================================")
    print(comparison.to_string(index=False))
    
    # -------------------------------------------------------------
    # Generate Optimization Plots
    # -------------------------------------------------------------
    print("\nGenerating optimization plots...")
    
    # 1. Optimization History Plot
    plt.figure(figsize=(8, 4.5))
    trial_numbers = [t.number + 1 for t in study.trials]
    trial_values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(trial_values)
    
    plt.scatter(trial_numbers, trial_values, color='#4c72b0', alpha=0.7, label='Trial CV Macro F1')
    plt.plot(trial_numbers, best_so_far, color='#c44e52', linewidth=2.2, label='Best Objective Value')
    plt.title('Optuna Bayesian Optimization History (GroupKFold CV F1)', fontsize=12, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=11)
    plt.ylabel('CV Macro F1-Score', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(opt_dir, 'optimization_history.png'), dpi=300)
    plt.close()
    
    # 2. Tuned vs Untuned Bar Plot
    metrics = ['CV Macro F1', 'Student-Test Macro F1', 'Top-1 Accuracy']
    untuned_vals = [untuned_row['CV Macro F1'], untuned_row['Student-Test Macro F1'], untuned_row['Top-1 Accuracy']]
    tuned_vals = [study.best_value, tuned_test_f1, tuned_top1_acc]
    
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, untuned_vals, width, label='Untuned XGBoost', color='#999999')
    plt.bar(x + width/2, tuned_vals, width, label='Tuned XGBoost', color='#2ca02c')
    plt.xticks(x, metrics, fontsize=10)
    plt.ylabel('Score', fontsize=11)
    plt.ylim(0.7, 0.95)
    plt.title('Untuned vs Bayesian Tuned XGBoost Performance', fontsize=12, fontweight='bold')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(opt_dir, 'tuned_vs_untuned_comparison.png'), dpi=300)
    plt.close()
    print("Optimization artifacts and plots successfully saved.")

if __name__ == '__main__':
    run_optimization()
