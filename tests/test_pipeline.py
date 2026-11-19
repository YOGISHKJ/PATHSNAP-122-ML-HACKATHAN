"""
PathSnap Automated Test Suite
Unit and Integration Tests for Dataset Ingestion, Preprocessing,
Candidate Generation, Feature Extraction, Model Loading, HMM/Viterbi,
and FastAPI Endpoints Across All 10 Student-Testing Trips.
"""

import os
import sys

# Ensure root, src, and route_reconstruction are in path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
route_dir = os.path.join(root_dir, 'route_reconstruction')
for d in [root_dir, src_dir, route_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import unittest
import numpy as np
import pandas as pd
import joblib

from preprocessing import load_trip, haversine_distance, calculate_bearing, align_track_to_reference_route
from candidate_generation import CandidateGenerator
from feature_engineering import FeaturePipeline
from hmm_viterbi import HMMViterbiDecoder
from backend.main import health_check, get_available_trips, predict_route, PredictRequest

class TestPathSnap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(root_dir, 'data')
        cls.models_dir = os.path.join(root_dir, 'models')
        cls.artifacts_dir = os.path.join(root_dir, 'artifacts')
        
        # Load artifacts
        cls.pipe = FeaturePipeline.load(os.path.join(cls.artifacts_dir, 'feature_pipeline.joblib'))
        cls.clf = joblib.load(os.path.join(cls.models_dir, 'final_model.joblib'))
        cls.generator = CandidateGenerator(top_k=5, search_pool=30)
        cls.decoder = HMMViterbiDecoder(
            classifier=cls.clf,
            feature_pipeline=cls.pipe,
            sigma_z=10.0,
            clf_weight=0.6,
            same_arc_logp=-0.1,
            direct_connected_logp=-0.3,
            disconnected_penalty=-10.0
        )
        cls.student_trips = sorted([d for d in os.listdir(os.path.join(cls.data_dir, 'student_test')) if os.path.isdir(os.path.join(cls.data_dir, 'student_test', d))])

    def test_01_dataset_parsing(self):
        """Test parsing of raw files for student_test trip 00000000."""
        trip = load_trip('00000000', self.data_dir, 'student_test')
        self.assertEqual(len(trip.track), 2503)
        self.assertEqual(len(trip.nodes), 8542)
        self.assertEqual(len(trip.arcs), 18196)
        self.assertIsNotNone(trip.route)
        self.assertEqual(len(trip.route), 179)
        print("  [PASS] Dataset parsing test passed.")

    def test_02_preprocessing_geodesics(self):
        """Test haversine geodesic calculation and projection sanity."""
        # 1 deg lat at equator is ~111,195 m
        d = haversine_distance(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(d, 111195.0, delta=500.0)
        
        bearing = calculate_bearing(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(bearing, 90.0, delta=1.0)
        print("  [PASS] Geodesic & bearing calculations passed.")

    def test_03_candidate_generation(self):
        """Test candidate generation and spatial indexing."""
        trip = load_trip('00000000', self.data_dir, 'student_test')
        cands, miss_rate = self.generator.generate_candidates_for_trip(trip, true_assigned_arcs=None)
        # N * top_k
        self.assertEqual(len(cands), 2503 * 5)
        self.assertIn('cand_distance_m', cands.columns)
        self.assertIn('heading_diff_deg', cands.columns)
        print("  [PASS] Candidate generation passed.")

    def test_04_feature_pipeline(self):
        """Test feature pipeline transformation on candidate table."""
        trip = load_trip('00000000', self.data_dir, 'student_test')
        cands, _ = self.generator.generate_candidates_for_trip(trip, true_assigned_arcs=None)
        X = self.pipe.transform(cands)
        self.assertEqual(X.shape, (len(cands), len(self.pipe.feature_columns)))
        self.assertFalse(np.isnan(X).any())
        print("  [PASS] Feature extraction & transformation passed.")

    def test_05_api_health_endpoint(self):
        """Test FastAPI /health route."""
        res = health_check()
        self.assertEqual(res.get('status'), 'healthy')
        self.assertTrue(res.get('model_loaded'))
        print("  [PASS] API /health passed.")

    def test_06_api_trips_endpoint(self):
        """Test FastAPI /trips route."""
        res = get_available_trips(partition='student_test')
        self.assertEqual(res.get('total_trips'), 10)
        self.assertEqual(len(res.get('trip_ids')), 10)
        print("  [PASS] API /trips passed.")

    def test_07_all_ten_student_testing_trips(self):
        """Compulsory requirement: test ALL 10 student testing trips through end-to-end pipeline."""
        self.assertEqual(len(self.student_trips), 10)
        print(f"\nTesting end-to-end prediction across all {len(self.student_trips)} student-testing trips:")
        
        for tid in self.student_trips:
            req = PredictRequest(trip_id=tid, partition='student_test')
            res = predict_route(req)
            self.assertEqual(res.status, 'success')
            self.assertEqual(res.trip_id, tid)
            self.assertGreater(res.total_predicted_arcs, 0)
            self.assertIsNotNone(res.evaluation_metrics)
            
            m = res.evaluation_metrics
            self.assertGreater(m.top1_accuracy_percent, 55.0)
            self.assertGreater(m.route_continuity_percent, 95.0)
            print(f"    Trip {tid}: Obs={res.trajectory_stats.n_observations:4d} | Top-1={m.top1_accuracy_percent:5.1f}% | Continuity={m.route_continuity_percent:5.1f}% | Overlap={m.reference_overlap_percent:5.1f}% | Latency={res.execution_time_ms:6.1f}ms")

if __name__ == '__main__':
    unittest.main()
