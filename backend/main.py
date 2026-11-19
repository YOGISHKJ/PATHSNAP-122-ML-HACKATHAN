"""
PathSnap FastAPI Backend Service
Provides high-performance RESTful map-matching and route-reconstruction inference.
Strict separation of concerns: prediction engine has ZERO access to ground truth;
server-side evaluation occurs independently post-inference.
"""

import os
import sys

# Ensure src and route_reconstruction are in python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
route_dir = os.path.join(root_dir, 'route_reconstruction')
for d in [src_dir, route_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import load_trip, align_track_to_reference_route
from candidate_generation import CandidateGenerator
from feature_engineering import FeaturePipeline
from hmm_viterbi import HMMViterbiDecoder

# Initialize FastAPI application
app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction API for Hackathon Evaluation",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts cache
DATA_DIR = os.path.join(root_dir, 'data')
MODELS_DIR = os.path.join(root_dir, 'models')
ARTIFACTS_DIR = os.path.join(root_dir, 'artifacts')

model_cache = {}

def get_engine():
    """Lazy-loads and caches model artifacts."""
    if 'decoder' not in model_cache:
        model_path = os.path.join(MODELS_DIR, 'final_model.joblib')
        pipe_path = os.path.join(ARTIFACTS_DIR, 'feature_pipeline.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(pipe_path):
            raise RuntimeError("Model artifacts not found. Please train models first.")
            
        clf = joblib.load(model_path)
        pipe = FeaturePipeline.load(pipe_path)
        
        decoder = HMMViterbiDecoder(
            classifier=clf,
            feature_pipeline=pipe,
            sigma_z=10.0,
            clf_weight=0.6,
            same_arc_logp=-0.1,
            direct_connected_logp=-0.3,
            disconnected_penalty=-10.0
        )
        generator = CandidateGenerator(top_k=5, search_pool=30)
        
        model_cache['decoder'] = decoder
        model_cache['generator'] = generator
        model_cache['clf_name'] = "Tuned XGBoost (Bayesian Opt) + HMM/Viterbi"
        
    return model_cache['decoder'], model_cache['generator']

class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Unique 8-digit trip identifier, e.g. '00000000'")
    partition: Optional[str] = Field("student_test", description="'student_test' or 'train'")

class TrajectoryStats(BaseModel):
    n_observations: int
    duration_s: float
    trajectory_length_km: float
    mean_speed_kmh: float

class EvaluationMetrics(BaseModel):
    top1_accuracy_percent: float
    route_continuity_percent: float
    reference_overlap_percent: float
    candidate_miss_rate_percent: float

class PredictResponse(BaseModel):
    trip_id: str
    partition: str
    status: str
    model_name: str
    execution_time_ms: float
    trajectory_stats: TrajectoryStats
    predicted_route_arcs: List[int]
    total_predicted_arcs: int
    unique_predicted_arcs: int
    evaluation_metrics: Optional[EvaluationMetrics] = None

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint verifying API and model status."""
    try:
        _, _ = get_engine()
        model_loaded = True
        model_name = model_cache.get('clf_name', 'Unknown')
    except Exception as e:
        model_loaded = False
        model_name = str(e)
        
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "PathSnap Map-Matching API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "model_name": model_name
    }

@app.get("/trips", tags=["Metadata"])
def get_available_trips(partition: str = Query("student_test", pattern="^(student_test|train)$")):
    """Lists available evaluation trip IDs for the selected partition."""
    part_dir = os.path.join(DATA_DIR, partition)
    if not os.path.exists(part_dir):
        raise HTTPException(status_code=404, detail=f"Partition '{partition}' not found.")
        
    trips = sorted([d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))])
    return {
        "partition": partition,
        "total_trips": len(trips),
        "trip_ids": trips
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_route(req: PredictRequest):
    """
    Main Map-Matching Endpoint.
    1. Loads GPS observations from raw .track, .nodes, .arcs.
    2. Runs candidate generation, feature extraction, classifier inference, and Viterbi decoding.
    3. Ground truth is NEVER accessed by prediction logic.
    4. Computes server-side evaluation metrics if reference route exists.
    """
    t_start = time.time()
    decoder, generator = get_engine()
    
    # 1. Load Trip Data (Zero access to ground truth route for inference)
    try:
        trip_data = load_trip(req.trip_id, data_dir=DATA_DIR, partition=req.partition)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Trip ID '{req.trip_id}' not found in partition '{req.partition}'.")
        
    # 2. Candidate Generation
    cand_df, miss_rate = generator.generate_candidates_for_trip(trip_data, true_assigned_arcs=None)
    
    # 3. Model Inference & Viterbi Decoding
    predicted_arcs, clf_only_arcs, _ = decoder.decode_trip(trip_data, cand_df)
    
    total_exec_time_ms = (time.time() - t_start) * 1000.0
    
    # Trajectory statistics
    traj_stats = TrajectoryStats(
        n_observations=len(trip_data.track),
        duration_s=float(trip_data.track['time'].iloc[-1] - trip_data.track['time'].iloc[0]) if len(trip_data.track) > 1 else 0.0,
        trajectory_length_km=float(trip_data.track['step_dist'].sum() / 1000.0),
        mean_speed_kmh=float(trip_data.track['speed_kmh'].mean())
    )
    
    # 4. Server-Side Evaluation (Decoupled from prediction logic)
    eval_metrics = None
    if trip_data.route is not None:
        true_assigned = align_track_to_reference_route(trip_data)
        valid_mask = true_assigned != -1
        
        top1_acc = float((np.array(predicted_arcs)[valid_mask] == true_assigned[valid_mask]).mean()) * 100.0 if valid_mask.sum() > 0 else 0.0
        
        # Route Continuity
        from_nodes = trip_data.arcs['from_node'].values
        to_nodes = trip_data.arcs['to_node'].values
        seq = np.array(predicted_arcs)
        if len(seq) > 1:
            curr_to = to_nodes[seq[:-1]]
            next_from = from_nodes[seq[1:]]
            connected = ((seq[:-1] == seq[1:]) | (curr_to == next_from)).sum()
            continuity_pct = float(connected / (len(seq) - 1)) * 100.0
        else:
            continuity_pct = 100.0
            
        ref_route_set = set(trip_data.route['arc_id'].values)
        overlap_pct = float(len(set(predicted_arcs) & ref_route_set) / max(len(ref_route_set), 1)) * 100.0
        
        eval_metrics = EvaluationMetrics(
            top1_accuracy_percent=round(top1_acc, 2),
            route_continuity_percent=round(continuity_pct, 2),
            reference_overlap_percent=round(overlap_pct, 2),
            candidate_miss_rate_percent=round(miss_rate * 100.0, 2)
        )
        
    return PredictResponse(
        trip_id=req.trip_id,
        partition=req.partition,
        status="success",
        model_name=model_cache.get('clf_name', 'Tuned XGBoost + HMM/Viterbi'),
        execution_time_ms=round(total_exec_time_ms, 2),
        trajectory_stats=traj_stats,
        predicted_route_arcs=predicted_arcs,
        total_predicted_arcs=len(predicted_arcs),
        unique_predicted_arcs=len(set(predicted_arcs)),
        evaluation_metrics=eval_metrics
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
