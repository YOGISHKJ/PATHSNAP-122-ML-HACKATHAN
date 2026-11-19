# Team PT-122 — Yogish
## Contributions: Exploratory Data Analysis & Route Reconstruction (Report Sections 2.1, 2.2, 4, 5)

### What this part covers:
- `results/eda/` — Dataset inventory, partition summaries, distribution plots:
  - `dataset_inventory.csv` (Detailed inventory of all 90 trips)
  - `partition_summary.csv` (Aggregated statistics for train and student_test partitions)
  - `dataset_exploration_report.md` (Exploratory data analysis report)
  - `gps_observation_distribution.png` (Distribution of GPS points per trip)
  - `speed_distribution.png` (Estimated kinematic speed profile)
  - `route_length_distribution.png` (Reference route length distribution)
  - `road_network_complexity.png` (Road graph node and arc counts)
  - `trajectory_length_vs_duration.png` (Spatial trajectory length vs time duration)
- `results/route_reconstruction/` — Sequential HMM + Log-space Viterbi decoding results:
  - `viterbi_comparison_results.csv` (Per-trip continuity and accuracy gains across all 10 student-test trips)
  - `route_continuity_improvement.png` (Route continuity jump from 87.5% to 99.0%)
  - `top1_accuracy_comparison.png` (Observation Top-1 accuracy improvement from 80.1% to 83.7%)
- `src/eda.py` — Exploratory Data Analysis script
- `route_reconstruction/hmm_viterbi.py` — HMM + Viterbi engine

### Reproduce:
```bash
python src/eda.py
python route_reconstruction/hmm_viterbi.py
```
