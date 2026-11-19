"""
PathSnap Hidden Markov Model (HMM) & Viterbi Route Reconstruction
Implements graph-aware sequential decoding using classifier emission probabilities,
topological road network transitions, log-domain Viterbi dynamic programming,
and rigorous comparative evaluation against classifier-only predictions.
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
import numpy as np
import pandas as pd
import networkx as nx

from feature_engineering import FeaturePipeline
from candidate_generation import CandidateGenerator
from preprocessing import load_trip, align_track_to_reference_route, haversine_distance

class HMMViterbiDecoder:
    """
    Decodes the most probable road-segment sequence across noisy GPS observations
    using a Hidden Markov Model with log-space Viterbi dynamic programming.
    """
    def __init__(
        self,
        classifier=None,
        feature_pipeline=None,
        sigma_z=10.0,
        beta=15.0,
        clf_weight=0.6,
        same_arc_logp=-0.2,
        direct_connected_logp=-0.5,
        disconnected_penalty=-12.0,
        gap_threshold_s=5.0
    ):
        self.clf = classifier
        self.pipe = feature_pipeline
        self.sigma_z = sigma_z
        self.beta = beta
        self.clf_weight = clf_weight
        self.same_arc_logp = same_arc_logp
        self.direct_connected_logp = direct_connected_logp
        self.disconnected_penalty = disconnected_penalty
        self.gap_threshold_s = gap_threshold_s

    def compute_emissions(self, cand_df):
        """
        Computes log emission probabilities combining calibrated classifier
        prediction probabilities with Gaussian orthogonal spatial distances.
        """
        N_cands = len(cand_df)
        if N_cands == 0:
            return np.array([])
            
        # Classifier probability
        X = self.pipe.transform(cand_df)
        p_clf = self.clf.predict_proba(X)[:, 1]
        p_clf = np.clip(p_clf, 1e-6, 1.0 - 1e-6)
        log_clf = np.log(p_clf)
        
        # Spatial distance emission: Gaussian N(0, sigma_z)
        dists = cand_df['cand_distance_m'].values
        log_spatial = -0.5 * (dists / self.sigma_z)**2 - np.log(self.sigma_z * np.sqrt(2 * np.pi))
        
        # Blended log emission probability
        log_emissions = self.clf_weight * log_clf + (1.0 - self.clf_weight) * log_spatial
        return log_emissions

    def decode_trip(self, trip_data, cand_df):
        """
        Runs Viterbi dynamic programming in log-space for a given trip.
        Returns:
            predicted_arcs: list of int arc_ids (length T)
            classifier_only_arcs: list of int arc_ids (length T)
            execution_time_s: float
        """
        t0 = time.time()
        if len(cand_df) == 0:
            return [], [], 0.0
            
        log_emissions_all = self.compute_emissions(cand_df)
        
        # Add emissions to cand_df before grouping
        cand_df = cand_df.copy()
        cand_df['log_emission'] = log_emissions_all
        cand_df['clf_prob'] = np.exp(cand_df['log_emission'])
        
        # Classifier-only baseline (argmax of probability per observation)
        clf_only_indices = cand_df.groupby('obs_idx')['log_emission'].idxmax()
        clf_only_arcs = cand_df.loc[clf_only_indices, 'cand_arc_id'].values.tolist()
        
        obs_groups = [group for _, group in cand_df.groupby('obs_idx')]
        T = len(obs_groups)
        if T == 0:
            return [], [], 0.0
        
        # Prepare road network topological adjacency
        from_nodes = trip_data.arcs['from_node'].values
        to_nodes = trip_data.arcs['to_node'].values
        
        # Track intervals
        dts = trip_data.track['dt'].values
        step_dists = trip_data.track['step_dist'].values
        
        # Initialize Trellis in log space
        first_group = obs_groups[0]
        K_0 = len(first_group)
        V_prev = first_group['log_emission'].values.copy()
        prev_arcs = first_group['cand_arc_id'].values
        
        # Backpointers: list of arrays of shape (K_t,)
        backpointers = []
        
        for t in range(1, T):
            curr_group = obs_groups[t]
            K_t = len(curr_group)
            curr_arcs = curr_group['cand_arc_id'].values
            curr_emissions = curr_group['log_emission'].values
            
            dt = dts[t]
            is_gap = dt > self.gap_threshold_s
            gps_dist = step_dists[t]
            
            V_curr = np.full(K_t, -np.inf)
            bp_curr = np.zeros(K_t, dtype=int)
            
            # Transition matrix from prev_arcs (K_{t-1}) to curr_arcs (K_t)
            for j in range(K_t):
                curr_arc = curr_arcs[j]
                curr_from = from_nodes[curr_arc]
                
                best_score = -np.inf
                best_prev_idx = 0
                
                for i in range(len(prev_arcs)):
                    prev_arc = prev_arcs[i]
                    prev_to = to_nodes[prev_arc]
                    
                    # Compute log transition probability
                    if prev_arc == curr_arc:
                        # Vehicle remained on the same road segment
                        log_trans = self.same_arc_logp
                    elif prev_to == curr_from:
                        # Directly connected road segment (continuous topological step)
                        log_trans = self.direct_connected_logp
                    else:
                        # Disconnected in single step
                        if is_gap:
                            # Large GPS gap: vehicle likely traversed intermediate unobserved segments
                            log_trans = -3.0
                        else:
                            log_trans = self.disconnected_penalty
                            
                    score = V_prev[i] + log_trans
                    if score > best_score:
                        best_score = score
                        best_prev_idx = i
                        
                V_curr[j] = best_score + curr_emissions[j]
                bp_curr[j] = best_prev_idx
                
            backpointers.append(bp_curr)
            V_prev = V_curr
            prev_arcs = curr_arcs
            
        # Backtracking
        best_last_idx = int(np.argmax(V_prev))
        best_path_indices = [best_last_idx]
        
        for t in range(T - 2, -1, -1):
            prev_idx = backpointers[t][best_path_indices[-1]]
            best_path_indices.append(prev_idx)
            
        best_path_indices.reverse()
        
        # Extract arc IDs
        predicted_arcs = []
        for t in range(T):
            group = obs_groups[t]
            arc_id = int(group['cand_arc_id'].iloc[best_path_indices[t]])
            predicted_arcs.append(arc_id)
            
        exec_time = float(time.time() - t0)
        return predicted_arcs, clf_only_arcs, exec_time

def evaluate_route_reconstruction(
    data_dir='data',
    artifacts_dir='artifacts',
    models_dir='models',
    results_dir='results',
    partition='student_test'
):
    """
    Evaluates and rigorously compares Classifier-Only vs Classifier+HMM+Viterbi
    across all student-testing trips.
    """
    os.makedirs(results_dir, exist_ok=True)
    out_dir = os.path.join(results_dir, 'route_reconstruction')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading optimal model and candidate generator...")
    model_path = os.path.join(models_dir, 'final_model.joblib')
    clf = joblib.load(model_path)
    pipe = FeaturePipeline.load(os.path.join(artifacts_dir, 'feature_pipeline.joblib'))
    
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
    
    trips = sorted([d for d in os.listdir(os.path.join(data_dir, partition)) if os.path.isdir(os.path.join(data_dir, partition, d))])
    print(f"Evaluating {len(trips)} {partition} trips with HMM / Viterbi...")
    
    results = []
    
    for tid in trips:
        trip_data = load_trip(tid, data_dir, partition=partition)
        true_assigned = align_track_to_reference_route(trip_data)
        cand_df, miss_rate = generator.generate_candidates_for_trip(trip_data, true_assigned)
        
        # Decode trip
        viterbi_arcs, clf_only_arcs, exec_time = decoder.decode_trip(trip_data, cand_df)
        
        # Top-1 accuracy against assigned ground-truth arcs
        valid_mask = true_assigned != -1
        N_valid = valid_mask.sum()
        
        clf_top1 = float((np.array(clf_only_arcs)[valid_mask] == true_assigned[valid_mask]).mean()) if N_valid > 0 else 0.0
        hmm_top1 = float((np.array(viterbi_arcs)[valid_mask] == true_assigned[valid_mask]).mean()) if N_valid > 0 else 0.0
        
        # Route Continuity
        from_nodes = trip_data.arcs['from_node'].values
        to_nodes = trip_data.arcs['to_node'].values
        
        def continuity(arcs_seq):
            if len(arcs_seq) <= 1:
                return 1.0
            seq = np.array(arcs_seq)
            curr_to = to_nodes[seq[:-1]]
            next_from = from_nodes[seq[1:]]
            # Connected if either same arc OR to_node == from_node
            connected = ((seq[:-1] == seq[1:]) | (curr_to == next_from)).sum()
            return float(connected / (len(seq) - 1))
            
        clf_continuity = continuity(clf_only_arcs)
        hmm_continuity = continuity(viterbi_arcs)
        
        # Reference Route overlap (sequence level)
        ref_route_set = set(trip_data.route['arc_id'].values) if trip_data.route is not None else set()
        clf_overlap = len(set(clf_only_arcs) & ref_route_set) / max(len(ref_route_set), 1)
        hmm_overlap = len(set(viterbi_arcs) & ref_route_set) / max(len(ref_route_set), 1)
        
        print(f"Trip {tid} (N={len(trip_data.track)}, Arcs={len(ref_route_set)}):")
        print(f"  Top-1 Accuracy:     Classifier = {clf_top1*100:.2f}% | HMM+Viterbi = {hmm_top1*100:.2f}% (Delta: {(hmm_top1-clf_top1)*100:+.2f}%)")
        print(f"  Route Continuity:   Classifier = {clf_continuity*100:.2f}% | HMM+Viterbi = {hmm_continuity*100:.2f}% (Delta: {(hmm_continuity-clf_continuity)*100:+.2f}%)")
        print(f"  Reference Overlap:  Classifier = {clf_overlap*100:.2f}% | HMM+Viterbi = {hmm_overlap*100:.2f}% (Delta: {(hmm_overlap-clf_overlap)*100:+.2f}%)")
        print(f"  Execution Time:     {exec_time*1000:.1f} ms\n")
        
        results.append({
            'trip_id': tid,
            'n_obs': len(trip_data.track),
            'route_len': len(ref_route_set),
            'miss_rate': miss_rate,
            'clf_top1_acc': clf_top1,
            'hmm_top1_acc': hmm_top1,
            'top1_improvement': hmm_top1 - clf_top1,
            'clf_continuity': clf_continuity,
            'hmm_continuity': hmm_continuity,
            'continuity_improvement': hmm_continuity - clf_continuity,
            'clf_route_overlap': clf_overlap,
            'hmm_route_overlap': hmm_overlap,
            'exec_time_s': exec_time
        })
        
    res_df = pd.DataFrame(results)
    res_csv = os.path.join(out_dir, 'viterbi_comparison_results.csv')
    res_df.to_csv(res_csv, index=False)
    print(f"Viterbi evaluation results saved to {res_csv}")
    
    # Summary
    print("==================================================")
    print("OVERALL ROUTE RECONSTRUCTION BENCHMARK SUMMARY")
    print("==================================================")
    print(f"Mean Top-1 Accuracy:   Classifier = {res_df['clf_top1_acc'].mean()*100:.2f}% | HMM+Viterbi = {res_df['hmm_top1_acc'].mean()*100:.2f}%")
    print(f"Mean Route Continuity: Classifier = {res_df['clf_continuity'].mean()*100:.2f}% | HMM+Viterbi = {res_df['hmm_continuity'].mean()*100:.2f}%")
    print(f"Mean Route Overlap:    Classifier = {res_df['clf_route_overlap'].mean()*100:.2f}% | HMM+Viterbi = {res_df['hmm_route_overlap'].mean()*100:.2f}%")
    print(f"Mean Trip Exec Time:   {res_df['exec_time_s'].mean()*1000:.1f} ms")
    
    # Plot Comparison
    generate_viterbi_plots(res_df, out_dir)
    return res_df

def generate_viterbi_plots(res_df, out_dir):
    """Generates comparison charts between Classifier-Only and HMM+Viterbi."""
    import matplotlib.pyplot as plt
    
    # 1. Route Continuity by Trip
    plt.figure(figsize=(10, 5))
    x = np.arange(len(res_df))
    width = 0.35
    plt.bar(x - width/2, res_df['clf_continuity']*100, width, label='Classifier-Only', color='#c44e52')
    plt.bar(x + width/2, res_df['hmm_continuity']*100, width, label='HMM + Viterbi', color='#2ca02c')
    plt.xticks(x, res_df['trip_id'], rotation=25, fontsize=10)
    plt.ylabel('Route Continuity (%)', fontsize=11)
    plt.title('Topological Route Continuity: Classifier-Only vs HMM + Viterbi Decoding', fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'route_continuity_improvement.png'), dpi=300)
    plt.close()
    
    # 2. Top-1 Accuracy by Trip
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, res_df['clf_top1_acc']*100, width, label='Classifier-Only', color='#4c72b0')
    plt.bar(x + width/2, res_df['hmm_top1_acc']*100, width, label='HMM + Viterbi', color='#55a868')
    plt.xticks(x, res_df['trip_id'], rotation=25, fontsize=10)
    plt.ylabel('Observation Top-1 Accuracy (%)', fontsize=11)
    plt.title('Observation Top-1 Accuracy: Classifier-Only vs HMM + Viterbi', fontsize=12, fontweight='bold')
    plt.ylim(60, 100)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'top1_accuracy_comparison.png'), dpi=300)
    plt.close()
    print("Viterbi comparative plots successfully generated.")

if __name__ == '__main__':
    evaluate_route_reconstruction()
