"""
Automated Test Suite for PathSnap FastAPI Backend & Prediction Pipeline
Tests server startup, health endpoint, trips metadata endpoint,
and full map-matching prediction across all 10 student-testing trips.
"""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import subprocess
import requests
import unittest

class TestPathSnapPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start backend server
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{root_dir};{os.path.join(root_dir, 'src')};{os.path.join(root_dir, 'route_reconstruction')}"
        cls.server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=root_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait up to 10s for server to start
        cls.base_url = "http://127.0.0.1:8000"
        started = False
        for _ in range(20):
            try:
                r = requests.get(f"{cls.base_url}/health", timeout=1)
                if r.status_code == 200:
                    started = True
                    break
            except Exception:
                time.sleep(0.5)
        if not started:
            cls.server_proc.terminate()
            raise RuntimeError("FastAPI server failed to start on 127.0.0.1:8000")

    @classmethod
    def tearDownClass(cls):
        cls.server_proc.terminate()
        cls.server_proc.wait()

    def test_01_health_endpoint(self):
        resp = requests.get(f"{self.base_url}/health", timeout=3)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        print(f"Health test passed: {data}")

    def test_02_trips_endpoint(self):
        resp = requests.get(f"{self.base_url}/trips?partition=student_test", timeout=3)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        trips = data.get("trip_ids", [])
        self.assertEqual(len(trips), 10)
        self.assertIn("00000000", trips)
        self.assertIn("00000007", trips)
        print(f"Trips test passed: Found {len(trips)} evaluation trips")

    def test_03_predict_all_ten_student_testing_trips(self):
        """Compulsory requirement: ensure all 10 student-testing trips pass through prediction."""
        trips = ['00000000', '00000007', '00000009', '00000040', '00000052', '00000056', '00000065', '00000076', '00000077', '00000091']
        
        for tid in trips:
            t0 = time.time()
            resp = requests.post(
                f"{self.base_url}/predict",
                json={"trip_id": tid, "partition": "student_test"},
                timeout=15
            )
            elapsed_ms = (time.time() - t0) * 1000.0
            self.assertEqual(resp.status_code, 200, f"Trip {tid} failed with status {resp.status_code}: {resp.text}")
            data = resp.json()
            
            # Validations
            self.assertEqual(data.get("status"), "success")
            self.assertEqual(data.get("trip_id"), tid)
            predicted_arcs = data.get("predicted_route_arcs", [])
            self.assertGreater(len(predicted_arcs), 0)
            
            metrics = data.get("evaluation_metrics", {})
            top1 = metrics.get("top1_accuracy_percent", 0.0)
            continuity = metrics.get("route_continuity_percent", 0.0)
            overlap = metrics.get("reference_overlap_percent", 0.0)
            
            self.assertGreater(top1, 50.0, f"Trip {tid} Top-1 Accuracy unusually low: {top1}%")
            self.assertGreater(continuity, 90.0, f"Trip {tid} Route Continuity unusually low: {continuity}%")
            
            print(f"  [PASS] Trip {tid}: N={data['trajectory_stats']['n_observations']} | Top-1={top1:.1f}% | Continuity={continuity:.1f}% | Overlap={overlap:.1f}% | Latency={elapsed_ms:.1f}ms")

if __name__ == '__main__':
    unittest.main()
