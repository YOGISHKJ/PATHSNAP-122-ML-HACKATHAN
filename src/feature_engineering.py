"""
PathSnap Feature Engineering Pipeline
Extracts, transforms, normalizes, and packages candidate-level features
for ML model training, cross-validation, and deployment inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'cand_distance_m',
    'heading_diff_deg',
    'cand_rank',
    'dist_ratio_to_min',
    'dist_diff_to_min',
    'gps_speed_kmh',
    'gps_step_dist_m',
    'time_diff_s',
    'road_length_m',
    'gps_lon',
    'gps_lat',
    'road_heading',
    'gps_heading'
]

class FeaturePipeline:
    def __init__(self, scaler_path=None):
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.scaler_path = scaler_path
        self.is_fitted = False

    def fit(self, df):
        """Fits scaler only on training partition feature matrix."""
        X = df[self.feature_columns].values
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df):
        """Transforms feature matrix using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform.")
        X = df[self.feature_columns].values
        return self.scaler.transform(X)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self, filepath):
        """Saves scaler artifact and configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """Loads fitted scaler artifact and configuration."""
        data = joblib.load(filepath)
        pipe = cls(scaler_path=filepath)
        pipe.scaler = data['scaler']
        pipe.feature_columns = data['feature_columns']
        pipe.is_fitted = data['is_fitted']
        return pipe
