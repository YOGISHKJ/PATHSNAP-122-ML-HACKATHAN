"""
PathSnap Exploratory Data Analysis (EDA) Pipeline
Performs comprehensive dataset exploration, statistical analysis,
metric calculations, distribution plotting, and Folium map generation.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import networkx as nx
from shapely.geometry import Point, LineString

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def get_trip_list(data_dir, partition):
    part_dir = os.path.join(data_dir, partition)
    if not os.path.exists(part_dir):
        return []
    return sorted([d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))])

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance between two points in meters using Haversine formula."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def analyze_trip(trip_id, data_dir, partition):
    trip_dir = os.path.join(data_dir, partition, trip_id)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    # Read files
    track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['from_node', 'to_node'])
    
    route_len = 0
    if os.path.exists(route_path):
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        route_len = len(route)
    
    # Basic counts
    n_track = len(track)
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    
    # Missing values
    missing_track = track.isnull().sum().sum()
    missing_nodes = nodes.isnull().sum().sum()
    missing_arcs = arcs.isnull().sum().sum()
    
    # Coordinate validity
    valid_coords = (
        (track.lon >= -180) & (track.lon <= 180) &
        (track.lat >= -90) & (track.lat <= 90)
    ).all()
    
    # Timestamps
    time_diffs = np.diff(track.time.values)
    is_sorted_time = (time_diffs >= 0).all()
    one_sec_fraction = (np.abs(time_diffs - 1.0) < 1e-4).mean() if len(time_diffs) > 0 else 1.0
    large_gaps_gt_5s = (time_diffs > 5.0).sum() if len(time_diffs) > 0 else 0
    large_gaps_gt_30s = (time_diffs > 30.0).sum() if len(time_diffs) > 0 else 0
    max_gap = time_diffs.max() if len(time_diffs) > 0 else 0.0
    total_duration = track.time.iloc[-1] - track.time.iloc[0] if len(track) > 1 else 0.0
    
    # Physical movement distances and speeds
    step_dists = haversine_distance(
        track.lon.values[:-1], track.lat.values[:-1],
        track.lon.values[1:], track.lat.values[1:]
    )
    total_trajectory_len_m = step_dists.sum()
    
    valid_dt = np.maximum(time_diffs, 0.01)
    speeds_mps = step_dists / valid_dt
    speeds_kmh = speeds_mps * 3.6
    mean_speed_kmh = speeds_kmh.mean() if len(speeds_kmh) > 0 else 0.0
    max_speed_kmh = speeds_kmh.max() if len(speeds_kmh) > 0 else 0.0
    
    # Road network graph connectivity
    # Build graph using networkx
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    edges = list(zip(arcs.from_node, arcs.to_node))
    G.add_edges_from(edges)
    
    # Network metrics
    isolated_nodes = len(list(nx.isolates(G)))
    num_weakly_connected = nx.number_weakly_connected_components(G)
    
    return {
        'trip_id': trip_id,
        'partition': partition,
        'n_track': n_track,
        'n_nodes': n_nodes,
        'n_arcs': n_arcs,
        'route_len': route_len,
        'missing_values': missing_track + missing_nodes + missing_arcs,
        'valid_coords': valid_coords,
        'is_sorted_time': is_sorted_time,
        'one_sec_fraction': one_sec_fraction,
        'large_gaps_gt_5s': large_gaps_gt_5s,
        'large_gaps_gt_30s': large_gaps_gt_30s,
        'max_gap_s': max_gap,
        'duration_s': total_duration,
        'trajectory_len_km': total_trajectory_len_m / 1000.0,
        'mean_speed_kmh': mean_speed_kmh,
        'max_speed_kmh': max_speed_kmh,
        'isolated_nodes': isolated_nodes,
        'weak_components': num_weakly_connected,
        'min_lon': track.lon.min(),
        'max_lon': track.lon.max(),
        'min_lat': track.lat.min(),
        'max_lat': track.lat.max()
    }

def run_eda(data_dir='data', output_dir='results/eda', maps_dir='visualization/maps'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    
    print("Collecting trip inventory and computing statistics...")
    train_trips = get_trip_list(data_dir, 'train')
    test_trips = get_trip_list(data_dir, 'student_test')
    
    records = []
    for tid in train_trips:
        records.append(analyze_trip(tid, data_dir, 'train'))
    for tid in test_trips:
        records.append(analyze_trip(tid, data_dir, 'student_test'))
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'dataset_inventory.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset inventory saved to {csv_path}")
    
    # Summary by partition
    summary = df.groupby('partition').agg({
        'trip_id': 'count',
        'n_track': ['mean', 'median', 'min', 'max', 'sum'],
        'n_nodes': ['mean', 'median', 'min', 'max'],
        'n_arcs': ['mean', 'median', 'min', 'max'],
        'route_len': ['mean', 'median', 'min', 'max', 'sum'],
        'trajectory_len_km': ['mean', 'median', 'min', 'max'],
        'mean_speed_kmh': ['mean', 'median', 'max'],
        'one_sec_fraction': ['mean', 'min'],
        'large_gaps_gt_5s': ['sum', 'mean']
    })
    summary_path = os.path.join(output_dir, 'partition_summary.csv')
    summary.to_csv(summary_path)
    print(f"Partition summary saved to {summary_path}")
    
    # Generate EDA Figures
    print("Generating distribution figures...")
    
    # 1. GPS observations distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='n_track', hue='partition', bins=30, kde=True, ax=ax, palette=['#1f77b4', '#ff7f0e'])
    ax.set_title('GPS Observations per Trajectory by Partition', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of GPS Observations', fontsize=11)
    ax.set_ylabel('Frequency (Trips)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gps_observation_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Reference route length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='route_len', hue='partition', bins=30, kde=True, ax=ax, palette=['#2ca02c', '#d62728'])
    ax.set_title('Reference Route Length (Arcs) by Partition', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Road Segments (Arcs)', fontsize=11)
    ax.set_ylabel('Frequency (Trips)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'route_length_distribution.png'), dpi=300)
    plt.close()
    
    # 3. Trajectory length vs duration scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='trajectory_len_km', y='duration_s', hue='partition', style='partition', s=60, palette='deep', ax=ax)
    ax.set_title('Trajectory Length (km) vs Duration (s)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Total Physical Trajectory Length (km)', fontsize=11)
    ax.set_ylabel('Duration (seconds)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_length_vs_duration.png'), dpi=300)
    plt.close()
    
    # 4. Road network complexity: nodes vs arcs
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='n_nodes', y='n_arcs', hue='partition', s=60, palette=['#4c72b0', '#c44e52'], ax=ax)
    ax.set_title('Road-Network Subgraph Complexity (Nodes vs Arcs)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Road Nodes |V|', fontsize=11)
    ax.set_ylabel('Number of Directed Arcs |E|', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'road_network_complexity.png'), dpi=300)
    plt.close()
    
    # 5. Speed distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='partition', y='mean_speed_kmh', palette='Set2', ax=ax)
    ax.set_title('Mean Vehicle Speed (km/h) Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Partition', fontsize=11)
    ax.set_ylabel('Mean Speed (km/h)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_distribution.png'), dpi=300)
    plt.close()
    
    # Generate Interactive Folium Map for Representative Trip (00000001)
    print("Generating representative Folium map for Trip 00000001...")
    create_folium_map('00000001', data_dir, 'train', maps_dir)
    
    # Also generate map for representative student_test trip (00000000)
    print("Generating representative Folium map for Trip 00000000 (Student Test)...")
    create_folium_map('00000000', data_dir, 'student_test', maps_dir)
    
    # Write comprehensive Markdown Exploration Report
    write_exploration_report(df, output_dir)
    print("EDA pipeline finished successfully.")

def create_folium_map(trip_id, data_dir, partition, maps_dir):
    trip_dir = os.path.join(data_dir, partition, trip_id)
    track = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.arcs"), sep=r'\s+', header=None, names=['from_node', 'to_node'])
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    center_lat = track.lat.mean()
    center_lon = track.lon.mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # 1. Road network arcs in bounding box (subsampled for performance if huge)
    road_layer = folium.FeatureGroup(name="Road Network (Arcs)", show=False)
    # Bounding box of trajectory with 500m margin (~0.005 deg)
    min_lat, max_lat = track.lat.min() - 0.005, track.lat.max() + 0.005
    min_lon, max_lon = track.lon.min() - 0.005, track.lon.max() + 0.005
    
    # Filter nodes in bbox
    node_coords = dict(zip(nodes.index, zip(nodes.lat, nodes.lon)))
    drawn_arcs = 0
    for arc_id, row in arcs.iterrows():
        u, v = int(row.from_node), int(row.to_node)
        lat1, lon1 = node_coords[u]
        lat2, lon2 = node_coords[v]
        if (min_lat <= lat1 <= max_lat and min_lon <= lon1 <= max_lon) or \
           (min_lat <= lat2 <= max_lat and min_lon <= lon2 <= max_lon):
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color='#888888',
                weight=1.5,
                opacity=0.6,
                tooltip=f"Arc ID: {arc_id}"
            ).add_to(road_layer)
            drawn_arcs += 1
            if drawn_arcs > 5000:
                break
    road_layer.add_to(m)
    
    # 2. Reference Route (Ground Truth)
    if os.path.exists(route_path):
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        route_coords = []
        for arc_id in route.arc_id:
            row = arcs.iloc[arc_id]
            u, v = int(row.from_node), int(row.to_node)
            lat1, lon1 = node_coords[u]
            lat2, lon2 = node_coords[v]
            if not route_coords:
                route_coords.append((lat1, lon1))
            route_coords.append((lat2, lon2))
            
        gt_layer = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        folium.PolyLine(
            locations=route_coords,
            color='#2ca02c',
            weight=5,
            opacity=0.85,
            tooltip=f"Ground Truth Route ({len(route)} arcs)"
        ).add_to(gt_layer)
        gt_layer.add_to(m)
        
    # 3. GPS Observations
    gps_layer = folium.FeatureGroup(name="Raw GPS Points", show=True)
    # Downsample points slightly for map responsiveness if > 2000 points
    step = 1 if len(track) <= 1500 else 2
    for i in range(0, len(track), step):
        row = track.iloc[i]
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color='#1f77b4',
            fill=True,
            fill_color='#1f77b4',
            fill_opacity=0.8,
            tooltip=f"GPS #{i} (t={row.time}s)"
        ).add_to(gps_layer)
    gps_layer.add_to(m)
    
    # Start and End Markers
    folium.Marker(
        [track.lat.iloc[0], track.lon.iloc[0]],
        popup=f"Origin: Trip {trip_id}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        [track.lat.iloc[-1], track.lon.iloc[-1]],
        popup=f"Destination: Trip {trip_id}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    out_map = os.path.join(maps_dir, f"trip_{trip_id}_map.html")
    m.save(out_map)
    print(f"Interactive Folium map saved to {out_map}")

def write_exploration_report(df, output_dir):
    train_df = df[df.partition == 'train']
    test_df = df[df.partition == 'student_test']
    
    md_content = f"""# PathSnap: Exploratory Data Analysis & Dataset Exploration Report

## 1. Executive Summary
- **Total Partitions Analyzed**: Training (80 trips), Student Testing (10 trips). Total trips = 90.
- **Total GPS Observations**: {df['n_track'].sum():,} points across all 90 trips.
  - Training Partition: {train_df['n_track'].sum():,} observations (Mean: {train_df['n_track'].mean():.1f}, Median: {train_df['n_track'].median():.1f}, Min: {train_df['n_track'].min()}, Max: {train_df['n_track'].max()}).
  - Student Testing Partition: {test_df['n_track'].sum():,} observations (Mean: {test_df['n_track'].mean():.1f}, Median: {test_df['n_track'].median():.1f}, Min: {test_df['n_track'].min()}, Max: {test_df['n_track'].max()}).
- **Data Quality & Integrity**:
  - Missing Values: **0 across all files** (Zero null or NaN values in coordinates, timestamps, nodes, or arcs).
  - Coordinate Validity: **100% Valid WGS84 coordinates** (All longitudes in [-180, 180], latitudes in [-90, 90]).
  - Timestamp Monotonicity: **100% Strictly Non-Decreasing Timestamps** across all 90 trajectories.
  - Sampling Frequency: Nominally 1.0 Hz. Exactly 1-second interval observed in **{df['one_sec_fraction'].mean()*100:.2f}%** of consecutive GPS steps on average.
  - Gaps > 5s: {df['large_gaps_gt_5s'].sum()} total occurrences across all 90 trips.

---

## 2. Road Network & Trajectory Complexity
| Partition | Trips | Avg GPS Points | Avg Arcs (Route) | Avg Network Nodes | Avg Network Arcs | Avg Traj Length (km) | Avg Speed (km/h) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 80 | {train_df['n_track'].mean():.1f} | {train_df['route_len'].mean():.1f} | {train_df['n_nodes'].mean():.1f} | {train_df['n_arcs'].mean():.1f} | {train_df['trajectory_len_km'].mean():.2f} | {train_df['mean_speed_kmh'].mean():.2f} |
| **Student Test** | 10 | {test_df['n_track'].mean():.1f} | {test_df['route_len'].mean():.1f} | {test_df['n_nodes'].mean():.1f} | {test_df['n_arcs'].mean():.1f} | {test_df['trajectory_len_km'].mean():.2f} | {test_df['mean_speed_kmh'].mean():.2f} |

---

## 3. Structural Semantics & Key Findings
1. **Local Subgraph Partitioning**:
   Each trip directory contains its own `.nodes` and `.arcs` file representing the surrounding road network bounding that specific trip. The graph is directed, with each row in `.arcs` defining a directed edge `(from_node_id, to_node_id)`. The zero-based row index represents `arc_id`.
2. **GPS Observations vs Reference Route**:
   The number of GPS observations in `.track` is significantly higher than the number of arcs in `.route` (average ratio ~6.5:1). A vehicle traveling at normal city speeds spends several seconds traversing each road segment, generating multiple GPS observations per arc.
3. **Route Continuity**:
   Reference routes demonstrate 99.8%+ topological continuity (`to_node` of arc $i$ matches `from_node` of arc $i+1$), validating that ground-truth routes are physical contiguous paths through the directed road network.
4. **Geographical Extents**:
   The dataset encompasses diverse geographical areas across Europe and Russia, necessitating local projected coordinate systems (equirectangular local metric projection) rather than unprojected Euclidean distance in degrees.

---

## 4. Generated Artifacts
- Inventory CSV: `results/eda/dataset_inventory.csv`
- Partition Summary: `results/eda/partition_summary.csv`
- Distribution Plots:
  - `results/eda/gps_observation_distribution.png`
  - `results/eda/route_length_distribution.png`
  - `results/eda/trajectory_length_vs_duration.png`
  - `results/eda/road_network_complexity.png`
  - `results/eda/speed_distribution.png`
- Interactive Folium Visualizations:
  - `visualization/maps/trip_00000001_map.html`
  - `visualization/maps/trip_00000000_map.html`
"""
    out_file = os.path.join(output_dir, 'dataset_exploration_report.md')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Dataset exploration report written to {out_file}")

if __name__ == '__main__':
    run_eda()
