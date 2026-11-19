"""
PathSnap Dataset Generation & Feature Extraction
Builds candidate-level datasets for all 80 training trips and 10 student-testing trips.
Ensures zero data leakage: feature pipeline fitted only on training partition.
"""

import os
import sys

# Ensure src is in python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import json
import time
import pandas as pd
import numpy as np

from preprocessing import load_trip, align_track_to_reference_route
from candidate_generation import CandidateGenerator
from feature_engineering import FeaturePipeline, FEATURE_COLUMNS

def build_datasets(data_dir='data', artifacts_dir='artifacts', top_k=5):
    os.makedirs(artifacts_dir, exist_ok=True)
    generator = CandidateGenerator(top_k=top_k, search_pool=30)
    
    # 1. Process Training Partition
    train_trips = sorted([d for d in os.listdir(os.path.join(data_dir, 'train')) if os.path.isdir(os.path.join(data_dir, 'train', d))])
    print(f"Processing {len(train_trips)} training trips...")
    
    t0 = time.time()
    train_dfs = []
    train_miss_rates = []
    
    for idx, tid in enumerate(train_trips):
        trip_data = load_trip(tid, data_dir, partition='train')
        true_assigned = align_track_to_reference_route(trip_data)
        cand_df, miss_rate = generator.generate_candidates_for_trip(trip_data, true_assigned)
        train_dfs.append(cand_df)
        train_miss_rates.append(miss_rate)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(train_trips):
            print(f"  Processed {idx + 1}/{len(train_trips)} training trips (Elapsed: {time.time()-t0:.1f}s)")
            
    all_train_df = pd.concat(train_dfs, ignore_index=True)
    mean_train_miss_rate = float(np.mean(train_miss_rates))
    print(f"Training dataset ready: {len(all_train_df):,} candidate rows. Avg candidate miss rate: {mean_train_miss_rate*100:.2f}%")
    
    # 2. Process Student Testing Partition
    test_trips = sorted([d for d in os.listdir(os.path.join(data_dir, 'student_test')) if os.path.isdir(os.path.join(data_dir, 'student_test', d))])
    print(f"\nProcessing {len(test_trips)} student testing trips...")
    
    t1 = time.time()
    test_dfs = []
    test_miss_rates = []
    
    for idx, tid in enumerate(test_trips):
        trip_data = load_trip(tid, data_dir, partition='student_test')
        true_assigned = align_track_to_reference_route(trip_data)
        cand_df, miss_rate = generator.generate_candidates_for_trip(trip_data, true_assigned)
        test_dfs.append(cand_df)
        test_miss_rates.append(miss_rate)
        print(f"  Trip {tid}: {len(trip_data.track)} obs -> {len(cand_df)} cands. Miss rate: {miss_rate*100:.2f}%")
        
    all_test_df = pd.concat(test_dfs, ignore_index=True)
    mean_test_miss_rate = float(np.mean(test_miss_rates))
    print(f"Student testing dataset ready: {len(all_test_df):,} candidate rows. Avg candidate miss rate: {mean_test_miss_rate*100:.2f}%")
    
    # 3. Fit Feature Pipeline (Scaler) strictly on training partition
    print("\nFitting feature pipeline strictly on training data...")
    feature_pipe = FeaturePipeline()
    feature_pipe.fit(all_train_df)
    
    pipe_path = os.path.join(artifacts_dir, 'feature_pipeline.joblib')
    feature_pipe.save(pipe_path)
    print(f"Feature pipeline saved to {pipe_path}")
    
    # Save candidate datasets
    train_parquet_path = os.path.join(artifacts_dir, 'train_candidates.parquet')
    test_parquet_path = os.path.join(artifacts_dir, 'test_candidates.parquet')
    
    all_train_df.to_parquet(train_parquet_path, index=False, compression='snappy')
    all_test_df.to_parquet(test_parquet_path, index=False, compression='snappy')
    print(f"Train candidates saved to {train_parquet_path} ({os.path.getsize(train_parquet_path)/(1024**2):.2f} MB)")
    print(f"Test candidates saved to {test_parquet_path} ({os.path.getsize(test_parquet_path)/(1024**2):.2f} MB)")
    
    # Save metadata configuration
    config = {
        'top_k': top_k,
        'search_pool': 30,
        'feature_columns': FEATURE_COLUMNS,
        'n_train_trips': len(train_trips),
        'n_train_candidates': len(all_train_df),
        'train_miss_rate': mean_train_miss_rate,
        'n_test_trips': len(test_trips),
        'n_test_candidates': len(all_test_df),
        'test_miss_rate': mean_test_miss_rate
    }
    config_path = os.path.join(artifacts_dir, 'preprocessing_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Preprocessing configuration saved to {config_path}")

if __name__ == '__main__':
    build_datasets()
