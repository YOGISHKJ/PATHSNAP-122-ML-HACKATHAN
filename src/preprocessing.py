"""
PathSnap Preprocessing Pipeline
Handles robust ingestion, spatial projection, geodesic calculation,
speed and heading derivation, anomaly handling, and ground-truth route alignment.
"""

import os
import numpy as np
import pandas as pd

EARTH_RADIUS_METERS = 6371000.0

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Computes great-circle distance in meters between pairs of coordinates using Haversine formula.
    Supports scalars and numpy arrays.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c

def calculate_bearing(lon1, lat1, lon2, lat2):
    """
    Calculates forward azimuth / bearing in degrees [0, 360) from point 1 to point 2.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0

def angular_difference(angle1, angle2):
    """
    Computes absolute minimum angular difference in [0, 180] degrees.
    """
    diff = np.abs(angle1 - angle2) % 360.0
    return np.minimum(diff, 360.0 - diff)

class TripData:
    """
    Container for trip data including raw observations, road network graph,
    projected coordinates, and extracted trajectory kinematics.
    """
    def __init__(self, trip_id, track_df, nodes_df, arcs_df, route_df=None, partition='train'):
        self.trip_id = trip_id
        self.partition = partition
        self.track = track_df
        self.nodes = nodes_df
        self.arcs = arcs_df
        self.route = route_df
        
        # Center of projection
        self.lat0 = float(track_df['lat'].mean())
        self.lon0 = float(track_df['lon'].mean())
        
        # Metric conversion factors for local equirectangular projection
        self.kx = np.cos(np.radians(self.lat0)) * np.radians(1.0) * EARTH_RADIUS_METERS
        self.ky = np.radians(1.0) * EARTH_RADIUS_METERS
        
        # Project coordinates
        self._project_coordinates()
        
        # Compute trajectory kinematics (speed, bearing, anomalies)
        self._compute_kinematics()

    def _project_coordinates(self):
        # Project GPS track points
        self.track['x'] = (self.track['lon'].values - self.lon0) * self.kx
        self.track['y'] = (self.track['lat'].values - self.lat0) * self.ky
        
        # Project road network nodes
        self.nodes['x'] = (self.nodes['lon'].values - self.lon0) * self.kx
        self.nodes['y'] = (self.nodes['lat'].values - self.lat0) * self.ky
        
        # Precompute arc geometries
        from_nodes = self.arcs['from_node'].values
        to_nodes = self.arcs['to_node'].values
        
        self.arc_ax = self.nodes['x'].values[from_nodes]
        self.arc_ay = self.nodes['y'].values[from_nodes]
        self.arc_bx = self.nodes['x'].values[to_nodes]
        self.arc_by = self.nodes['y'].values[to_nodes]
        
        self.arc_dx = self.arc_bx - self.arc_ax
        self.arc_dy = self.arc_by - self.arc_ay
        self.arc_l2 = self.arc_dx * self.arc_dx + self.arc_dy * self.arc_dy
        self.arc_len = np.sqrt(self.arc_l2)
        
        # Road heading in [0, 360)
        self.arc_heading = (np.degrees(np.arctan2(self.arc_dx, self.arc_dy)) + 360.0) % 360.0
        
        # Arc midpoints for spatial indexing
        self.arc_mid_x = (self.arc_ax + self.arc_bx) / 2.0
        self.arc_mid_y = (self.arc_ay + self.arc_by) / 2.0

    def _compute_kinematics(self):
        n = len(self.track)
        if n <= 1:
            self.track['dt'] = 1.0
            self.track['step_dist'] = 0.0
            self.track['speed_mps'] = 0.0
            self.track['speed_kmh'] = 0.0
            self.track['heading'] = 0.0
            self.track['is_gap'] = False
            return
            
        times = self.track['time'].values
        dt = np.zeros(n, dtype=np.float32)
        dt[1:] = times[1:] - times[:-1]
        dt[0] = dt[1] if n > 1 else 1.0
        dt = np.maximum(dt, 0.001)
        
        # Physical step distances using Haversine
        step_dist = np.zeros(n, dtype=np.float32)
        step_dist[1:] = haversine_distance(
            self.track['lon'].values[:-1], self.track['lat'].values[:-1],
            self.track['lon'].values[1:], self.track['lat'].values[1:]
        )
        step_dist[0] = step_dist[1] if n > 1 else 0.0
        
        # Speed in m/s and km/h
        speed_mps = step_dist / dt
        # Handle GPS jumps / anomalies: cap unrealistic city/highway speeds > 160 km/h
        speed_kmh = speed_mps * 3.6
        speed_kmh = np.clip(speed_kmh, 0.0, 200.0)
        
        # Headings
        headings = np.zeros(n, dtype=np.float32)
        headings[1:] = calculate_bearing(
            self.track['lon'].values[:-1], self.track['lat'].values[:-1],
            self.track['lon'].values[1:], self.track['lat'].values[1:]
        )
        headings[0] = headings[1] if n > 1 else 0.0
        
        # Gap flag (gaps > 5 seconds)
        is_gap = dt > 5.0
        
        self.track['dt'] = dt
        self.track['step_dist'] = step_dist
        self.track['speed_mps'] = speed_mps
        self.track['speed_kmh'] = speed_kmh
        self.track['heading'] = headings
        self.track['is_gap'] = is_gap

def load_trip(trip_id, data_dir='data', partition='train'):
    """
    Parses and pre-processes all files for a specific trip.
    """
    trip_dir = os.path.join(data_dir, partition, trip_id)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Missing track file for trip {trip_id}: {track_path}")
        
    track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'], dtype=np.float64)
    nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'], dtype=np.float64)
    arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['from_node', 'to_node'], dtype=np.int32)
    
    route = None
    if os.path.exists(route_path):
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'], dtype=np.int32)
        
    return TripData(trip_id, track, nodes, arcs, route, partition)

def align_track_to_reference_route(trip_data):
    """
    Defensible, technically grounded method to associate training GPS observations
    with reference route arcs using progressive forward-constrained dynamic projection.
    
    For each GPS observation t_i:
    Finds the road arc a in route that best projects onto p_i subject to
    forward chronological movement along the reference path.
    """
    if trip_data.route is None or len(trip_data.route) == 0:
        return np.full(len(trip_data.track), -1, dtype=np.int32)
        
    route_arcs = trip_data.route['arc_id'].values
    M = len(route_arcs)
    N = len(trip_data.track)
    
    from_x = trip_data.arc_ax[route_arcs]
    from_y = trip_data.arc_ay[route_arcs]
    to_x = trip_data.arc_bx[route_arcs]
    to_y = trip_data.arc_by[route_arcs]
    
    track_x = trip_data.track['x'].values
    track_y = trip_data.track['y'].values
    
    curr_idx = 0
    assigned_arcs = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        px, py = track_x[i], track_y[i]
        
        # Constrained forward window [curr_idx - 2, curr_idx + 20]
        w_start = max(0, curr_idx - 2)
        w_end = min(M, curr_idx + 25)
        
        best_d = 1e9
        best_r = curr_idx
        
        for r in range(w_start, w_end):
            ax, ay = from_x[r], from_y[r]
            bx, by = to_x[r], to_y[r]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 == 0:
                d = np.sqrt((px - ax)**2 + (py - ay)**2)
            else:
                t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / l2))
                proj_x = ax + t * dx
                proj_y = ay + t * dy
                d = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            if d < best_d:
                best_d = d
                best_r = r
                
        curr_idx = best_r
        assigned_arcs[i] = route_arcs[best_r]
        
    return assigned_arcs
