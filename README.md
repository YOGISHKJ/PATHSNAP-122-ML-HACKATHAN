# PathSnap: Distributed Map-Matching and Route Reconstruction

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB6424?logo=xgboost)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-273575?logo=optuna)](https://optuna.org)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)](https://www.python.org)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

> **PES University Machine Learning Hackathon (UE24CS352A)**  
> **Team ID: PT-122 (PathSnap)**: End-to-end distributed geospatial map-matching and topological route reconstruction pipeline integrating spatial indexing, supervised machine learning, Bayesian optimization, and log-space Hidden Markov Model (HMM) Viterbi dynamic programming.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Mathematical Formulation](#mathematical-formulation)
3. [System Architecture](#system-architecture)
4. [Empirical Results & Benchmarks](#empirical-results--benchmarks)
   - [Model Comparison (5 Architectures)](#model-comparison-5-architectures)
   - [Bayesian Hyperparameter Tuning (Optuna)](#bayesian-hyperparameter-tuning-optuna)
   - [HMM + Viterbi Topological Gains](#hmm--viterbi-topological-gains)
5. [Directory Layout](#directory-layout)
6. [Installation & Requirements](#installation--requirements)
7. [Reproduction & Pipeline Execution](#reproduction--pipeline-execution)
8. [Deployment Architectures](#deployment-architectures)
   - [Option A: Single-Laptop Mode (Development / Verification)](#option-a-single-laptop-mode)
   - [Option B: Two-Laptop Mode (Physical Hackathon Evaluation)](#option-b-two-laptop-mode)
9. [REST API Documentation](#rest-api-documentation)
10. [Automated Test Suite](#automated-test-suite)

---

## Executive Summary

PathSnap addresses the fundamental problem of reconstructing high-fidelity travel routes from noisy, low-frequency, or temporally sampled GPS trajectory observations onto a directed road graph. Raw GPS points suffer from sensor jitter, multi-path reflections, and variable sampling intervals. While point-wise classifiers can identify candidate road segments with high individual precision, they lack network awareness—producing disjointed segments with low topological continuity (~25%).

PathSnap resolves this challenge using a two-stage hybrid architecture:
1. **Kinematic Feature Engineering & Supervised Learning**: Spatial $k$-d trees index road arcs, projecting raw GPS points onto candidate directed line segments. An optimized **XGBoost** classifier outputs calibrated candidate probabilities from 13 spatial and kinematic features.
2. **Sequential Inference via Log-Space HMM & Viterbi Decoding**: Road network topology, arc connectivity, transition reachability, and great-circle movement constraints form the transition matrix. The Viterbi algorithm decodes the global maximum likelihood sequence, boosting route continuity from **87.48% to 99.00%** and observation accuracy from **80.09% to 83.68%**.

---

## Mathematical Formulation

### 1. Geodesic Distance (Haversine & Equirectangular Projection)
To avoid spherical distortion in Euclidean calculations, physical distance between GPS coordinates $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$ is calculated via the Haversine formula:
$$\Delta\sigma = 2 \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right) } \right)$$
$$d = R \cdot \Delta\sigma \quad (R = 6,371,000 \text{ m})$$

For rapid spatial projection within a localized bounding box centered at $\phi_0$, local planar coordinates are projected as:
$$x = R (\lambda - \lambda_0) \cos(\phi_0), \quad y = R (\phi - \phi_0)$$

### 2. Spatial Projection onto Directed Road Arcs
Each candidate road arc $a = (u, v)$ has start node $\mathbf{p}_u$ and end node $\mathbf{p}_v$. The projection scalar $t$ of GPS point $\mathbf{g}$ onto segment $\mathbf{v} - \mathbf{u}$ is:
$$t = \text{clamp}\left( \frac{(\mathbf{g} - \mathbf{p}_u) \cdot (\mathbf{p}_v - \mathbf{p}_u)}{\|\mathbf{p}_v - \mathbf{p}_u\|^2}, 0, 1 \right)$$
The projected point is $\mathbf{p}^*(t) = \mathbf{p}_u + t(\mathbf{p}_v - \mathbf{p}_u)$, and candidate orthogonal distance is:
$$d_\perp(g, a) = \|\mathbf{g} - \mathbf{p}^*(t)\|$$

### 3. Log-Space Emission Probabilities
Given classifier emission probability $P_{\text{clf}}(a \mid g)$ and spatial distance $d_\perp(g, a)$, emissions are modeled as a mixture of Gaussian distance decay and classifier likelihood:
$$\ln P_{\text{emit}}(g \mid a) = \alpha \ln\left( \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left( -\frac{d_\perp^2}{2\sigma_z^2} \right) \right) + (1 - \alpha) \ln P_{\text{clf}}(a \mid g)$$
where $\sigma_z = 20.0\text{ m}$ represents GPS measurement variance, and $\alpha = 0.35$.

### 4. Topological Transition Probabilities
For consecutive GPS observations $g_{t-1} \to g_t$ with candidate road segments $a_i \to a_j$:
$$\ln P_{\text{trans}}(a_j \mid a_i) = \begin{cases}
\ln(0.95), & \text{if } a_i = a_j \text{ (same segment persistence)} \\
\ln(0.85), & \text{if } \text{target}(a_i) = \text{source}(a_j) \text{ (direct graph edge)} \\
\ln(0.40) - \frac{|d_{\text{net}} - d_{\text{gps}}|}{\beta}, & \text{if reachable within 2-hop radius} \\
\ln(10^{-6}), & \text{if disconnected / unreachable}
\end{cases}$$

### 5. Viterbi Dynamic Programming Recurrence
In log-space (avoiding underflow for long trajectories):
$$V_1(a) = \ln \pi(a) + \ln P_{\text{emit}}(g_1 \mid a)$$
$$V_t(a_j) = \max_{a_i \in C_{t-1}} \left[ V_{t-1}(a_i) + \ln P_{\text{trans}}(a_j \mid a_i) \right] + \ln P_{\text{emit}}(g_t \mid a_j)$$
$$B_t(a_j) = \arg\max_{a_i \in C_{t-1}} \left[ V_{t-1}(a_i) + \ln P_{\text{trans}}(a_j \mid a_i) \right]$$
Backtracking yields the globally optimal road sequence: $a_T^* = \arg\max V_T(a)$, and $a_{t-1}^* = B_t(a_t^*)$.

---

## System Architecture

```
                                  +-----------------------------+
                                  |   Raw Trip Data Files       |
                                  |  (.track, .nodes, .arcs)    |
                                  +--------------+--------------+
                                                 |
                                                 v
+-----------------------------------------------------------------------------------------------+
|                                  PATHSNAP PREPROCESSING ENGINE                                |
|  - Geodesic Distance & Forward Bearing Calculation                                            |
|  - Kinematic Speeds (km/h) & Temporal Gap Detection (>10s)                                    |
|  - cKDTree Spatial Indexing on Arc Midpoints (k=10 Candidates per GPS point)                  |
|  - Point-to-Directed-Segment Orthogonal Projection & Orientation Deviations                   |
|  - 13 Engineering Features & Robust Standard Scaling                                          |
+------------------------------------------------+----------------------------------------------+
                                                 |
                                                 v
+-----------------------------------------------------------------------------------------------+
|                                 MACHINE LEARNING INFERENCE LAYER                              |
|  - Tuned XGBoost (Optuna Bayesian Optimized: depth=5, lr=0.0538, subsample=0.781)             |
|  - Calibrated Candidate Posterior Probabilities: P(Candidate = Correct | Features)           |
+------------------------------------------------+----------------------------------------------+
                                                 |
                                                 v
+-----------------------------------------------------------------------------------------------+
|                              LOG-SPACE HMM & VITERBI DECODER                                  |
|  - Gaussian Spatial + Classifier Mixture Emissions                                            |
|  - Directed Graph Topology & Multi-hop Transition Matrix                                      |
|  - Global Maximum Likelihood Sequence Backtracking                                            |
+------------------------------------------------+----------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
         +---------------------------+                       +---------------------------+
         |     FastAPI Service       | <== HTTP JSON ====>   |    Streamlit Dashboard    |
         |  (:8000 / HOST 0.0.0.0)   |                       |    (:8501 / Web UI)       |
         |  - POST /predict          |                       |  - Real-time Controls     |
         |  - GET /trips             |                       |  - Metrics & Comparison   |
         |  - GET /health            |                       |  - Interactive Folium Map |
         +---------------------------+                       +---------------------------+
```

---

## Empirical Results & Benchmarks

All models were evaluated using **5-Fold `GroupKFold`** grouped strictly by `trip_id` (ensuring 0% cross-trip data leakage) across 80 training trips (1,042,085 candidate records) and evaluated on 10 student-testing trips (95,800 candidate records).

### Model Comparison (5 Architectures)

| Model | CV Macro F1 | Student-Test Macro F1 | Observation Top-1 Acc | Route Continuity | Log Loss | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8276 | 0.8532 | 77.67% | 25.68% | 0.2294 | 3.04s | 0.010s |
| **K-Nearest Neighbors** | 0.8291 | 0.8349 | 74.75% | 25.46% | 0.5461 | 0.77s | 58.80s |
| **Decision Tree** | 0.7969 | 0.7820 | 61.69% | 25.59% | 1.7728 | 29.93s | 0.027s |
| **Random Forest** | 0.8341 | 0.8765 | 78.60% | 25.07% | 0.2040 | 108.01s | 0.345s |
| **XGBoost (Untuned)** | 0.8232 | 0.8757 | 78.67% | 24.64% | **0.1899** | 9.31s | 0.135s |

> **Selection Rationale**: While Random Forest achieved comparable accuracy to XGBoost, it required **108 seconds** to train versus **9.3 seconds** for XGBoost ($11.6\times$ faster). Furthermore, XGBoost delivered the lowest log loss (0.1899) and superior probability calibration, making it the ideal emission estimator for HMM decoding.

---

### Bayesian Hyperparameter Tuning (Optuna)

Optuna performed 15 trials maximizing CV Macro F1 using GroupKFold.
- **Optimal Hyperparameters**:
  - `n_estimators`: `60`
  - `max_depth`: `5`
  - `learning_rate`: `0.0538`
  - `subsample`: `0.7813`
  - `colsample_bytree`: `0.7972`
  - `reg_alpha`: `0.0101`
  - `reg_lambda`: `1.7248`

| Metric | Untuned XGBoost | Tuned XGBoost (Optuna) | Relative Change |
| :--- | :---: | :---: | :---: |
| **CV Macro F1** | 0.8232 | **0.8286** | **+0.54%** |
| **Student-Test Top-1 Accuracy** | 78.67% | **78.68%** | +0.01% |
| **Log Loss** | 0.1899 | **0.1982** | Smoother calibration |
| **Inference Time (95.8k records)** | 135 ms | **78.5 ms** | **41.9% Faster** |

---

### HMM + Viterbi Topological Gains

Viterbi decoding was executed across all 10 student-testing trips, comparing raw point-wise classifier predictions against sequentially decoded routes:

| Metric | Point-wise Classifier Only | HMM + Viterbi Decoded | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Observation Top-1 Accuracy** | 80.09% | **83.68%** | **+3.59%** (Improved on 10/10 trips) |
| **Topological Route Continuity** | 87.48% | **99.00%** | **+11.52%** (100% on trip 00000007) |
| **Reference Route Arc Overlap** | 86.46% | **94.21%** | **+7.75%** (100% on trip 00000040) |
| **Mean End-to-End Latency** | - | **624.5 ms / trip** | Real-time capable |

---

## Directory Layout

```
PathSnap_Hackathon/
├── data/
│   ├── train/                       # 80 training trips (.track, .nodes, .arcs, .route)
│   └── student_test/                # 10 student testing trips
├── documentation/                   # 5 competition guideline PDFs
├── src/
│   ├── preprocessing.py             # Coordinate validation, geodesics, kinematics
│   ├── candidate_generation.py      # Spatial indexing, candidate generation
│   ├── feature_engineering.py       # 13 candidate-level features & RobustScaler
│   ├── build_dataset.py             # Parquet dataset builder
│   ├── train_models.py              # 5-model benchmark suite (GroupKFold)
│   └── eda.py                       # Exploratory Data Analysis & visual generation
├── models/                          # Trained model binaries (.joblib)
│   ├── final_model.joblib           # Optuna-tuned XGBoost model
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   └── logistic_regression.joblib
├── optimization/                    # Optuna Bayesian tuning scripts & trial records
│   ├── optimize_xgboost.py
│   ├── best_hyperparameters.json
│   └── optuna_study.csv
├── route_reconstruction/            # Sequential HMM and Viterbi decoder
│   └── hmm_viterbi.py
├── backend/                         # FastAPI distributed REST service
│   └── main.py
├── frontend/                        # Streamlit interactive dashboard
│   └── app.py
├── visualization/                   # Map generation & publication exports
│   ├── generate_maps.py
│   ├── maps/                        # Interactive Folium HTML maps
│   └── exports/                     # High-resolution PNG figures
├── tests/                           # Complete automated unit test suite
│   └── test_pipeline.py
├── results/                         # Empirical logs, comparison tables, confusion matrices
│   ├── progress_log.md              # 18-phase implementation audit log
│   ├── model_comparison.csv
│   └── eda/
├── reports/                         # PDF technical report & generator
│   ├── generate_report_pdf.py
│   └── PathSnap_Technical_Report.pdf
├── start_backend.bat                # Windows launcher for FastAPI (0.0.0.0:8000)
├── start_frontend.bat               # Windows launcher for Streamlit UI (:8501)
├── start_all_single_laptop.bat      # Single-laptop automated launcher
├── requirements.txt                 # Pinned dependencies
├── .gitignore
└── README.md
```

---

## Installation & Requirements

Ensure **Python 3.10+** (tested on Python 3.11.9) is installed.

```bash
# 1. Clone repository
git clone https://github.com/your-username/PathSnap_Hackathon.git
cd PathSnap_Hackathon

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproduction & Pipeline Execution

To reproduce the complete research and engineering pipeline from scratch:

```bash
# Step 1: Run Exploratory Data Analysis (EDA)
python src/eda.py

# Step 2: Build Feature Parquet Datasets & Fitted Scalers
python src/build_dataset.py

# Step 3: Train & Benchmark 5 Machine Learning Models
python src/train_models.py

# Step 4: Run Bayesian Hyperparameter Optimization with Optuna
python optimization/optimize_xgboost.py

# Step 5: Run HMM & Log-Space Viterbi Sequential Evaluation
python route_reconstruction/hmm_viterbi.py

# Step 6: Generate Multi-Layer Folium Interactive Maps & Visuals
python visualization/generate_maps.py

# Step 7: Run Automated Unit Tests (7/7 tests)
python tests/test_pipeline.py

# Step 8: Build Consolidated Technical PDF Report
python reports/generate_report_pdf.py
```

---

## Deployment Architectures

### Option A: Single-Laptop Mode

For local verification, development, and standalone presentation:

```bash
# Double-click or execute from root:
start_all_single_laptop.bat
```
This launches:
- **FastAPI Backend**: `http://127.0.0.1:8000` (docs at `http://127.0.0.1:8000/docs`)
- **Streamlit Frontend**: `http://localhost:8501`

---

### Option B: Two-Laptop Mode (Physical Hackathon Setup)

To demonstrate true distributed map-matching across two laptops connected to the same local network:

```
+--------------------------------------+            +--------------------------------------+
|             LAPTOP 1                 |            |              LAPTOP 2                |
|      (Backend Inference Node)        |    WiFi    |      (Frontend Client Dashboard)     |
|   IP: 192.168.1.50 (e.g.)            | <========> |   IP: 192.168.1.60                   |
|   Port: 8000                         |  LAN / IP  |   Port: 8501                         |
|   - FastAPI Server                   |            |   - Streamlit UI Dashboard           |
|   - Model & Spatial Index Workers    |            |   - Real-time Visualizer             |
+--------------------------------------+            +--------------------------------------+
```

#### Step 1: Find Laptop 1's Local IP Address
On Laptop 1, run:
```cmd
ipconfig
```
Locate the IPv4 Address under your active Wi-Fi / Ethernet adapter (e.g., `192.168.1.50`).

#### Step 2: Launch Backend on Laptop 1
Start the backend bound to `0.0.0.0` (all interfaces):
```cmd
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*(Ensure Windows Firewall permits inbound connections on TCP Port 8000).*

#### Step 3: Launch Frontend on Laptop 2
On Laptop 2, set the `BACKEND_URL` environment variable pointing to Laptop 1:
```cmd
set BACKEND_URL=http://192.168.1.50:8000
python -m streamlit run frontend/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`. Laptop 2 will transmit raw GPS trajectory observations over HTTP to Laptop 1, which will execute spatial projection, XGBoost inference, and Viterbi decoding, returning reconstructed routes and continuity metrics.

---

## REST API Documentation

### 1. Health Probe
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

### 2. List Available Trips
- **Endpoint**: `GET /trips?partition=student_test`
- **Response**:
```json
{
  "partition": "student_test",
  "count": 10,
  "trips": ["trip_00000000", "trip_00000001", "..."]
}
```

### 3. Route Prediction & Map-Matching
- **Endpoint**: `POST /predict`
- **Parameters**: `evaluate=true` (optional boolean for server-side evaluation when reference route is available)
- **Request Body**:
```json
{
  "trip_id": "trip_00000007",
  "gps_points": [
    {"lon": 7.421, "lat": 46.948, "timestamp": 1289384920.0},
    {"lon": 7.422, "lat": 46.949, "timestamp": 1289384921.0}
  ],
  "nodes": [
    {"node_id": 0, "lon": 7.420, "lat": 46.947},
    {"node_id": 1, "lon": 7.425, "lat": 46.950}
  ],
  "arcs": [
    {"arc_id": 0, "from_node": 0, "to_node": 1}
  ]
}
```
- **Response Body**:
```json
{
  "trip_id": "trip_00000007",
  "num_observations": 124,
  "reconstructed_route": [0, 1, 4, 12, ...],
  "observation_arcs": [0, 0, 1, 4, ...],
  "execution_time_ms": 142.5,
  "metrics": {
    "continuity_pct": 100.0,
    "top1_accuracy_pct": 88.71,
    "reference_overlap_pct": 96.15
  }
}
```

---

## Automated Test Suite

The test suite validates every stage of the pipeline without external mock dependencies:
```bash
python -m unittest tests/test_pipeline.py -v
```

**Test Verification Summary**:
```
test_01_raw_file_parsing (tests.test_pipeline.TestPathSnapPipeline.test_01_raw_file_parsing) ... ok
test_02_geodesic_and_projection (tests.test_pipeline.TestPathSnapPipeline.test_02_geodesic_and_projection) ... ok
test_03_candidate_generation (tests.test_pipeline.TestPathSnapPipeline.test_03_candidate_generation) ... ok
test_04_feature_pipeline (tests.test_pipeline.TestPathSnapPipeline.test_04_feature_pipeline) ... ok
test_05_backend_health (tests.test_pipeline.TestPathSnapPipeline.test_05_backend_health) ... ok
test_06_backend_trips (tests.test_pipeline.TestPathSnapPipeline.test_06_backend_trips) ... ok
test_07_end_to_end_prediction (tests.test_pipeline.TestPathSnapPipeline.test_07_end_to_end_prediction) ... ok

----------------------------------------------------------------------
Ran 7 tests in 14.180s

OK
```

---

## License & Academic Integrity
This software was developed strictly in adherence to the PES University Academic Integrity and Anti-Plagiarism Policy for UE24CS352A. All models, transformations, and evaluation procedures are original work.
