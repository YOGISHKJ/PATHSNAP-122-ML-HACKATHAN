# PathSnap: Exploratory Data Analysis & Dataset Exploration Report

## 1. Executive Summary
- **Total Partitions Analyzed**: Training (80 trips), Student Testing (10 trips). Total trips = 90.
- **Total GPS Observations**: 227,577 points across all 90 trips.
  - Training Partition: 208,417 observations (Mean: 2605.2, Median: 1529.0, Min: 361, Max: 19313).
  - Student Testing Partition: 19,160 observations (Mean: 1916.0, Median: 1420.0, Min: 262, Max: 3958).
- **Data Quality & Integrity**:
  - Missing Values: **0 across all files** (Zero null or NaN values in coordinates, timestamps, nodes, or arcs).
  - Coordinate Validity: **100% Valid WGS84 coordinates** (All longitudes in [-180, 180], latitudes in [-90, 90]).
  - Timestamp Monotonicity: **100% Strictly Non-Decreasing Timestamps** across all 90 trajectories.
  - Sampling Frequency: Nominally 1.0 Hz. Exactly 1-second interval observed in **94.32%** of consecutive GPS steps on average.
  - Gaps > 5s: 1260 total occurrences across all 90 trips.

---

## 2. Road Network & Trajectory Complexity
| Partition | Trips | Avg GPS Points | Avg Arcs (Route) | Avg Network Nodes | Avg Network Arcs | Avg Traj Length (km) | Avg Speed (km/h) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 80 | 2605.2 | 391.7 | 24168.6 | 48461.3 | 37.23 | 54.65 |
| **Student Test** | 10 | 1916.0 | 347.7 | 9068.9 | 18504.1 | 29.70 | 55.80 |

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
