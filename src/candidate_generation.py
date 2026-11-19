"""
PathSnap Candidate Generation Pipeline
Implements spatial indexing with KDTree, geometric perpendicular distance
to directed road segments, angular heading differences, and candidate set formation.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class CandidateGenerator:
    """
    Generates spatial road candidates for GPS observations using KDTree indexing
    on road segment midpoints and exact vector projection onto line segments.
    """
    def __init__(self, top_k=5, search_pool=30):
        self.top_k = top_k
        self.search_pool = search_pool

    def generate_candidates_for_trip(self, trip_data, true_assigned_arcs=None):
        """
        Generates candidate sets for each GPS point in trip_data.
        Returns:
            candidates_df: DataFrame of candidate-level rows with features
            miss_rate: float, fraction of observations where true arc was not in top-k
        """
        N = len(trip_data.track)
        n_arcs = len(trip_data.arcs)
        
        # Build KDTree on arc midpoints
        midpoints = np.column_stack([trip_data.arc_mid_x, trip_data.arc_mid_y])
        tree = cKDTree(midpoints)
        
        # GPS coordinates
        gps_x = trip_data.track['x'].values
        gps_y = trip_data.track['y'].values
        gps_headings = trip_data.track['heading'].values
        gps_speeds = trip_data.track['speed_kmh'].values
        gps_dts = trip_data.track['dt'].values
        gps_dists = trip_data.track['step_dist'].values
        gps_lons = trip_data.track['lon'].values
        gps_lats = trip_data.track['lat'].values
        gps_times = trip_data.track['time'].values
        
        # Query nearest search_pool arcs for all points at once (vectorized KDTree query!)
        k_query = min(self.search_pool, n_arcs)
        _, candidate_indices = tree.query(np.column_stack([gps_x, gps_y]), k=k_query)
        if candidate_indices.ndim == 1:
            candidate_indices = candidate_indices[:, np.newaxis]
            
        candidate_rows = []
        hits = 0
        total_eval = 0
        
        for i in range(N):
            px, py = gps_x[i], gps_y[i]
            cand_arc_ids = candidate_indices[i]
            
            # Segment endpoints for these candidates
            ax = trip_data.arc_ax[cand_arc_ids]
            ay = trip_data.arc_ay[cand_arc_ids]
            bx = trip_data.arc_bx[cand_arc_ids]
            by = trip_data.arc_by[cand_arc_ids]
            
            dx = bx - ax
            dy = by - ay
            l2 = dx * dx + dy * dy
            
            # Vector projection factor t clamped to [0, 1]
            ap_x = px - ax
            ap_y = py - ay
            proj = (ap_x * dx + ap_y * dy)
            t = np.where(l2 > 0, np.clip(proj / np.maximum(l2, 1e-6), 0.0, 1.0), 0.0)
            
            # Closest point on segment
            closest_x = ax + t * dx
            closest_y = ay + t * dy
            
            # Orthogonal distance in meters
            dists = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
            
            # Sort candidate arcs by orthogonal distance
            sort_order = np.argsort(dists)
            top_indices = sort_order[:self.top_k]
            
            min_dist = dists[top_indices[0]]
            
            true_arc = true_assigned_arcs[i] if true_assigned_arcs is not None else -1
            top_arc_ids = cand_arc_ids[top_indices]
            
            if true_arc != -1:
                total_eval += 1
                if true_arc in top_arc_ids:
                    hits += 1
                    
            for rank_0, s_idx in enumerate(top_indices):
                arc_id = int(cand_arc_ids[s_idx])
                dist_m = float(dists[s_idx])
                road_hdg = float(trip_data.arc_heading[arc_id])
                gps_hdg = float(gps_headings[i])
                
                # Heading difference in [0, 180]
                hdg_diff = abs(gps_hdg - road_hdg) % 360.0
                hdg_diff = min(hdg_diff, 360.0 - hdg_diff)
                
                road_len = float(trip_data.arc_len[arc_id])
                
                # Relative distance metrics
                dist_ratio = dist_m / max(min_dist, 1.0)
                dist_diff = dist_m - min_dist
                
                # Target label (1 if matching assigned true arc, 0 otherwise)
                label = 1 if (true_arc != -1 and arc_id == true_arc) else 0
                
                candidate_rows.append({
                    'trip_id': trip_data.trip_id,
                    'obs_idx': i,
                    'cand_rank': rank_0 + 1,
                    'cand_arc_id': arc_id,
                    'cand_distance_m': dist_m,
                    'road_length_m': road_len,
                    'gps_speed_kmh': gps_speeds[i],
                    'gps_step_dist_m': gps_dists[i],
                    'time_diff_s': gps_dts[i],
                    'gps_heading': gps_hdg,
                    'road_heading': road_hdg,
                    'heading_diff_deg': hdg_diff,
                    'dist_ratio_to_min': dist_ratio,
                    'dist_diff_to_min': dist_diff,
                    'gps_lon': gps_lons[i],
                    'gps_lat': gps_lats[i],
                    'time': gps_times[i],
                    'label': label
                })
                
        candidates_df = pd.DataFrame(candidate_rows)
        miss_rate = 1.0 - (hits / max(total_eval, 1)) if total_eval > 0 else 0.0
        return candidates_df, miss_rate
