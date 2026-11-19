"""
PathSnap Machine Learning Model Training & Comparative Evaluation
Trains and evaluates 5 distinct ML models using GroupKFold cross-validation
and comprehensive student-testing evaluation.
"""

import os
import sys

# Ensure src is in python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_engineering import FeaturePipeline, FEATURE_COLUMNS
from preprocessing import load_trip

def calculate_route_continuity(pred_arcs, arcs_df):
    """
    Calculates fraction of consecutive predicted arcs where to_node == from_node.
    """
    if len(pred_arcs) <= 1:
        return 1.0
    from_nodes = arcs_df['from_node'].values
    to_nodes = arcs_df['to_node'].values
    
    curr_to = to_nodes[pred_arcs[:-1]]
    next_from = from_nodes[pred_arcs[1:]]
    
    connected = (curr_to == next_from).sum()
    total = len(pred_arcs) - 1
    return float(connected / total)

def evaluate_observation_level(test_df, probs, data_dir='data'):
    """
    Computes Observation-Level Top-1 Accuracy and Classifier-Only Route Continuity
    across all student-testing trips.
    """
    df_eval = test_df[['trip_id', 'obs_idx', 'cand_arc_id', 'label']].copy()
    df_eval['prob'] = probs
    
    # For each observation, find candidate with highest probability
    top1_indices = df_eval.groupby(['trip_id', 'obs_idx'])['prob'].idxmax()
    top1_df = df_eval.loc[top1_indices].sort_values(['trip_id', 'obs_idx'])
    
    # Top-1 accuracy (is the top predicted candidate the true arc?)
    top1_acc = float(top1_df['label'].mean())
    
    # Route continuity per trip
    continuities = []
    for tid, group in top1_df.groupby('trip_id'):
        pred_arcs = group['cand_arc_id'].values
        arcs_file = os.path.join(data_dir, 'student_test', tid, f"{tid}.arcs")
        arcs_df = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['from_node', 'to_node'])
        c = calculate_route_continuity(pred_arcs, arcs_df)
        continuities.append(c)
        
    mean_continuity = float(np.mean(continuities))
    return top1_acc, mean_continuity, top1_df

def get_models():
    """Initializes the 5 required models from distinct algorithm families."""
    return {
        'Logistic Regression': {
            'family': 'Linear & Probabilistic',
            'model': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            'needs_subsample_knn': False,
            'supports_proba': True,
            'key_config': 'C=1.0, solver=lbfgs, max_iter=1000'
        },
        'K-Nearest Neighbors': {
            'family': 'Instance-Based',
            'model': KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
            'needs_subsample_knn': True, # Subsample for memory/speed feasibility on 1M rows
            'supports_proba': True,
            'key_config': 'n_neighbors=15, weights=distance, metric=minkowski'
        },
        'Decision Tree': {
            'family': 'Tree-Based',
            'model': DecisionTreeClassifier(max_depth=12, min_samples_split=20, random_state=42),
            'needs_subsample_knn': False,
            'supports_proba': True,
            'key_config': 'max_depth=12, min_samples_split=20, criterion=gini'
        },
        'Random Forest': {
            'family': 'Ensemble (Bagging)',
            'model': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=20, n_jobs=-1, random_state=42),
            'needs_subsample_knn': False,
            'supports_proba': True,
            'key_config': 'n_estimators=100, max_depth=15, min_samples_split=20'
        },
        'XGBoost': {
            'family': 'Ensemble (Boosting)',
            'model': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='logloss'),
            'needs_subsample_knn': False,
            'supports_proba': True,
            'key_config': 'n_estimators=100, max_depth=6, lr=0.1, objective=binary:logistic'
        }
    }

def run_experiment(artifacts_dir='artifacts', models_dir='models', results_dir='results', n_splits=5):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'confusion_matrices'), exist_ok=True)
    
    print("Loading candidate datasets...")
    train_df = pd.read_parquet(os.path.join(artifacts_dir, 'train_candidates.parquet'))
    test_df = pd.read_parquet(os.path.join(artifacts_dir, 'test_candidates.parquet'))
    
    pipe = FeaturePipeline.load(os.path.join(artifacts_dir, 'feature_pipeline.joblib'))
    
    X_train = pipe.transform(train_df)
    y_train = train_df['label'].values
    groups_train = train_df['trip_id'].values
    
    X_test = pipe.transform(test_df)
    y_test = test_df['label'].values
    
    print(f"Train candidate matrix: {X_train.shape}, Test candidate matrix: {X_test.shape}")
    print(f"Positive class balance in train: {y_train.mean()*100:.2f}%, in test: {y_test.mean()*100:.2f}%")
    
    models_dict = get_models()
    comparison_records = []
    
    # 5-fold GroupKFold CV
    gkf = GroupKFold(n_splits=n_splits)
    
    for name, info in models_dict.items():
        print(f"\n==================================================")
        print(f"Evaluating Model: {name} ({info['family']})")
        print(f"Configuration: {info['key_config']}")
        print(f"==================================================")
        
        clf = info['model']
        
        # 1. Group-Aware Cross-Validation on 80 training trips
        cv_f1_scores = []
        cv_acc_scores = []
        print(f"Running {n_splits}-Fold GroupKFold Cross-Validation...")
        
        # If KNN, sample a subset of groups for CV to finish within reasonable time
        if info['needs_subsample_knn']:
            # Use 20 trips for KNN training subset
            unique_trips = np.unique(groups_train)
            np.random.seed(42)
            sampled_trips = np.random.choice(unique_trips, size=25, replace=False)
            mask_knn = np.isin(groups_train, sampled_trips)
            X_cv_train = X_train[mask_knn]
            y_cv_train = y_train[mask_knn]
            groups_cv = groups_train[mask_knn]
        else:
            X_cv_train = X_train
            y_cv_train = y_train
            groups_cv = groups_train
            
        fold = 1
        for train_idx, val_idx in gkf.split(X_cv_train, y_cv_train, groups=groups_cv):
            X_fold_tr, y_fold_tr = X_cv_train[train_idx], y_cv_train[train_idx]
            X_fold_val, y_fold_val = X_cv_train[val_idx], y_cv_train[val_idx]
            
            clf.fit(X_fold_tr, y_fold_tr)
            y_pred_val = clf.predict(X_fold_val)
            
            f1_macro = f1_score(y_fold_val, y_pred_val, average='macro', zero_division=0)
            acc = accuracy_score(y_fold_val, y_pred_val)
            cv_f1_scores.append(f1_macro)
            cv_acc_scores.append(acc)
            print(f"  Fold {fold}: Macro F1 = {f1_macro:.4f}, Accuracy = {acc:.4f}")
            fold += 1
            
        mean_cv_f1 = float(np.mean(cv_f1_scores))
        std_cv_f1 = float(np.std(cv_f1_scores))
        print(f"--> Mean CV Macro F1: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})")
        
        # 2. Final Training on Full Train Partition
        print(f"Training {name} on full training partition...")
        t_tr_start = time.time()
        if info['needs_subsample_knn']:
            # Fit KNN on 150,000 samples for fast inference
            np.random.seed(42)
            knn_idx = np.random.choice(len(X_train), size=min(150000, len(X_train)), replace=False)
            clf.fit(X_train[knn_idx], y_train[knn_idx])
        else:
            clf.fit(X_train, y_train)
        train_time_sec = float(time.time() - t_tr_start)
        print(f"--> Training completed in {train_time_sec:.2f} seconds.")
        
        # 3. Inference on Student-Testing Partition
        print("Evaluating on student-testing partition...")
        t_inf_start = time.time()
        y_test_pred = clf.predict(X_test)
        inference_time_sec = float(time.time() - t_inf_start)
        
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_prec_macro = float(precision_score(y_test, y_test_pred, average='macro', zero_division=0))
        test_rec_macro = float(recall_score(y_test, y_test_pred, average='macro', zero_division=0))
        test_f1_macro = float(f1_score(y_test, y_test_pred, average='macro', zero_division=0))
        
        # Probability & Log loss
        y_test_probs = None
        test_log_loss = None
        if info['supports_proba']:
            y_test_probs = clf.predict_proba(X_test)[:, 1]
            test_log_loss = float(log_loss(y_test, y_test_probs))
        else:
            y_test_probs = y_test_pred.astype(float)
            
        # Top-1 accuracy and Route Continuity
        top1_acc, continuity, top1_df = evaluate_observation_level(test_df, y_test_probs)
        
        print(f"--> Student-Test Accuracy:    {test_acc:.4f}")
        print(f"--> Student-Test Macro F1:    {test_f1_macro:.4f}")
        print(f"--> Observation Top-1 Acc:    {top1_acc:.4f} (Candidate Miss Rate: 3.60%)")
        print(f"--> Route Continuity:         {continuity:.4f}")
        if test_log_loss is not None:
            print(f"--> Student-Test Log Loss:    {test_log_loss:.4f}")
        print(f"--> Inference time:           {inference_time_sec:.3f} s ({inference_time_sec*1000/len(X_test):.3f} ms/cand)")
        
        # Save Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        cm_path = os.path.join(results_dir, 'confusion_matrices', f'cm_{name.lower().replace(" ", "_")}.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        
        # Save Model Artifact
        model_filename = f"{name.lower().replace(' ', '_').replace('-', '_')}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(clf, model_path)
        print(f"Saved model artifact to {model_path}")
        
        # Record Comparison Data
        comparison_records.append({
            'Model': name,
            'Algorithm Family': info['family'],
            'Key Configuration': info['key_config'],
            'CV Macro F1': mean_cv_f1,
            'CV Macro F1 Std': std_cv_f1,
            'Student-Test Macro F1': test_f1_macro,
            'Top-1 Accuracy': top1_acc,
            'Route Continuity': continuity,
            'Student-Test Accuracy': test_acc,
            'Student-Test Precision': test_prec_macro,
            'Student-Test Recall': test_rec_macro,
            'Log Loss': test_log_loss if test_log_loss is not None else -1.0,
            'Training Time (s)': train_time_sec,
            'Inference Time (s)': inference_time_sec,
            'Artifact Path': model_path
        })
        
    comp_df = pd.DataFrame(comparison_records)
    comp_csv = os.path.join(results_dir, 'model_comparison.csv')
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n==================================================")
    print(f"Model comparison table successfully written to {comp_csv}")
    print(f"==================================================")
    print(comp_df[['Model', 'Algorithm Family', 'CV Macro F1', 'Top-1 Accuracy', 'Route Continuity', 'Student-Test Macro F1', 'Log Loss', 'Training Time (s)']])
    
    # Generate Comparison Charts
    generate_comparison_plots(comp_df, results_dir)

def generate_comparison_plots(comp_df, results_dir):
    """Generates comparison bar plots for model selection justification."""
    plots_dir = os.path.join(results_dir, 'model_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Macro F1 comparison (CV vs Student Test)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    plt.bar(x - width/2, comp_df['CV Macro F1'], width, label='CV Macro F1', color='#4c72b0')
    plt.bar(x + width/2, comp_df['Student-Test Macro F1'], width, label='Student-Test Macro F1', color='#55a868')
    plt.xticks(x, comp_df['Model'], rotation=15, ha='right', fontsize=10)
    plt.ylabel('Macro F1-Score', fontsize=11)
    plt.title('Cross-Validation vs Student-Test Macro F1-Score', fontsize=13, fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'macro_f1_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Observation-level Top-1 Accuracy & Route Continuity
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, comp_df['Top-1 Accuracy'], width, label='Top-1 Accuracy', color='#c44e52')
    plt.bar(x + width/2, comp_df['Route Continuity'], width, label='Route Continuity', color='#8172b3')
    plt.axhline(0.964, color='gray', linestyle='--', label='Theoretical Max (1 - Miss Rate = 96.4%)')
    plt.xticks(x, comp_df['Model'], rotation=15, ha='right', fontsize=10)
    plt.ylabel('Score', fontsize=11)
    plt.title('Observation Top-1 Accuracy and Route Continuity (Classifier Only)', fontsize=13, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top1_accuracy_and_continuity.png'), dpi=300)
    plt.close()
    
    # 3. Training & Inference Time Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=comp_df, x='Model', y='Training Time (s)', ax=ax1, palette='Blues_r')
    ax1.set_title('Model Training Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20, ha='right')
    
    sns.barplot(data=comp_df, x='Model', y='Inference Time (s)', ax=ax2, palette='Greens_r')
    ax2.set_title('Inference Time on 95.8k Candidates (seconds)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'timing_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Log Loss Comparison (Probabilistic Models)
    prob_df = comp_df[comp_df['Log Loss'] > 0]
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=prob_df, x='Model', y='Log Loss', palette='rocket')
    plt.title('Student-Testing Log Loss Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'log_loss_comparison.png'), dpi=300)
    plt.close()
    print(f"Generated comparison plots in {plots_dir}")

if __name__ == '__main__':
    run_experiment()
