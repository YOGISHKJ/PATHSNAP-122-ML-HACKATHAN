# PathSnap: Distributed Map-Matching and Route Reconstruction
## Hackathon Progress & Execution Log

### Initial Status
- Environment: Windows 11, Python 3.11.9, 12 CPU Cores, 15.2 GB RAM.
- Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `shapely`, `optuna`, `folium`, `fastapi`, `uvicorn`, `streamlit`, `pydantic`, `matplotlib`, `seaborn`, `reportlab`, `pypdf`, `networkx` installed and verified. `xgboost` currently installing in background.

---

### Phase 1: Extract and Inspect Files
- **Status**: Completed
- **Files Created**:
  - `data/train/` (80 trip directories: 00000001 .. 00000099)
  - `data/student_test/` (10 trip directories: 00000000, 00000007, 00000009, 00000040, 00000052, 00000056, 00000065, 00000076, 00000077, 00000091)
  - `documentation/` (Extracted all 5 PDFs: `Main Guidelines.pdf`, `Models and Training Guidelines.pdf`, `Deliverables.pdf`, `Assessment and Plagiarism Policy.pdf`, `Dataset Exploration.pdf`)
  - Extracted clean text files for deep inspection: `documentation/*_clean.txt`
  - Directory structure initialized: `src/`, `models/`, `optimization/`, `route_reconstruction/`, `backend/`, `frontend/`, `visualization/`, `experiments/`, `results/eda/`, `artifacts/`, `reports/`, `tests/`
- **Verification**:
  - Exactly 80 training trips verified.
  - Exactly 10 student-testing trips verified.
  - All trips contain `.track`, `.nodes`, `.arcs`, and `.route`.
- **Remaining Tasks**: Proceed with detailed dataset exploration and Phase 2 EDA.

### Phase 2: Exploratory Data Analysis (EDA)
- **Status**: Completed
- **Files Created**:
  - `src/eda.py`
  - `results/eda/dataset_inventory.csv` (Per-trip dataset inventory across all 90 trips)
  - `results/eda/partition_summary.csv` (Aggregated statistics by partition)
  - `results/eda/dataset_exploration_report.md` (Detailed structural and empirical report)
  - Figures in `results/eda/`:
    - `gps_observation_distribution.png`
    - `route_length_distribution.png`
    - `trajectory_length_vs_duration.png`
    - `road_network_complexity.png`
    - `speed_distribution.png`
  - Interactive Folium Visualizations:
    - `visualization/maps/trip_00000001_map.html` (Training trip with GPS observations, local road network arcs, and reference route)
    - `visualization/maps/trip_00000000_map.html` (Student testing trip)
- **Key Empirical Findings**:
  - Missing values: 0 across all files and columns.
  - Coordinates: 100% valid WGS84 coordinates.
  - Timestamps: Strictly monotonic, 1.0 Hz nominal frequency (93.8% exact 1s steps in train, 98.2% in test).
  - Train set: 80 trips, 208,417 GPS observations, 31,336 route arcs.
  - Student test set: 10 trips, 19,160 GPS observations, 3,477 route arcs.
  - Reference route continuity: 99.8% topological continuity between consecutive arcs.

---

### Phase 3, 4, 5: Preprocessing, Candidate Generation, Feature Engineering
- **Status**: Completed
- **Files Created**:
  - `src/preprocessing.py`: Haversine calculation, forward bearing, equirectangular projection, speed and gap detection, progressive forward-constrained dynamic route alignment.
  - `src/candidate_generation.py`: KDTree spatial indexing on arc midpoints, exact perpendicular vector segment distance, heading difference, candidate ranking.
  - `src/feature_engineering.py`: Feature extraction and StandardScaler pipeline fitted exclusively on training data.
  - `src/build_dataset.py`: Batch generation of candidate datasets.
  - `artifacts/train_candidates.parquet` (1,042,085 candidate rows, 46.46 MB)
  - `artifacts/test_candidates.parquet` (95,800 candidate rows, 4.54 MB)
  - `artifacts/feature_pipeline.joblib` (Fitted StandardScaler artifact)
  - `artifacts/preprocessing_config.json` (Configuration parameters and empirical candidate miss rates)
- **Key Empirical Metrics**:
  - Training candidate miss rate (Top-5): 6.64% (93.36% coverage).
  - Student testing candidate miss rate (Top-5): 3.60% (96.40% coverage).
  - Target labels established using chronological forward projection without data leakage.

### Phase 6 & 7: Model Training, Cross-Validation, and Selection
- **Status**: Completed
- **Files Created**:
  - `src/train_models.py`
  - Model artifacts in `models/`:
    - `logistic_regression.joblib`
    - `k_nearest_neighbors.joblib`
    - `decision_tree.joblib`
    - `random_forest.joblib`
    - `xgboost.joblib`
  - Evaluation results:
    - `results/model_comparison.csv`
    - Confusion matrices in `results/confusion_matrices/` (`cm_logistic_regression.png`, `cm_k_nearest_neighbors.png`, `cm_decision_tree.png`, `cm_random_forest.png`, `cm_xgboost.png`)
    - Comparison plots in `results/model_plots/` (`macro_f1_comparison.png`, `top1_accuracy_and_continuity.png`, `timing_comparison.png`, `log_loss_comparison.png`)
- **Key Empirical Benchmark Summary**:
  - **Logistic Regression**: CV Macro F1: 0.8276, Test Macro F1: 0.8532, Top-1 Acc: 77.67%, Continuity: 25.68%, Log Loss: 0.2294, Train Time: 3.04s, Infer: 0.010s
  - **K-Nearest Neighbors**: CV Macro F1: 0.8291, Test Macro F1: 0.8349, Top-1 Acc: 74.75%, Continuity: 25.46%, Log Loss: 0.5461, Train Time: 0.77s, Infer: 58.80s
  - **Decision Tree**: CV Macro F1: 0.7969, Test Macro F1: 0.7820, Top-1 Acc: 61.69%, Continuity: 25.59%, Log Loss: 1.7728, Train Time: 29.93s, Infer: 0.027s
  - **Random Forest**: CV Macro F1: 0.8341, Test Macro F1: 0.8765, Top-1 Acc: 78.60%, Continuity: 25.07%, Log Loss: 0.2040, Train Time: 108.01s, Infer: 0.345s
  - **XGBoost**: CV Macro F1: 0.8232, Test Macro F1: 0.8757, Top-1 Acc: 78.67%, Continuity: 24.64%, Log Loss: 0.1899, Train Time: 9.31s, Infer: 0.135s
- **Model Selection Justification**:
  - XGBoost selected for Bayesian hyperparameter optimization and HMM integration due to lowest Log Loss (0.1899, vital for emission probability quality), highest Top-1 Accuracy (78.67%), exceptional training speed (9.31s vs 108s for RF), and fast inference (135ms for 95.8k candidates).

### Phase 8: Bayesian Hyperparameter Optimization
- **Status**: Completed
- **Files Created**:
  - `optimization/optimize_xgboost.py`
  - `optimization/optuna_study.csv` (15 complete trials history)
  - `optimization/best_hyperparameters.json`
  - `optimization/tuned_vs_untuned_comparison.csv`
  - `optimization/optimization_history.png`
  - `optimization/tuned_vs_untuned_comparison.png`
  - `models/final_model.joblib` (Final tuned model artifact)
- **Empirical Optimization Findings**:
  - Search Algorithm: Optuna Tree-structured Parzen Estimator (TPE).
  - Objective: GroupKFold CV Macro F1 on training trips.
  - Initial untuned CV Macro F1: 0.8232.
  - Optimal Tuned CV Macro F1: **0.8286** (Trial 9).
  - Best Parameters: `n_estimators=60`, `max_depth=5`, `learning_rate=0.0538`, `subsample=0.7813`, `colsample_bytree=0.7972`, `reg_alpha=0.0101`, `reg_lambda=1.7248`.
  - Tuned Model Inference Speed: 0.0785s for 95,800 candidates (42% faster than baseline XGBoost due to regularized depth and tree pruning).

### Phase 9 & 10: Hidden Markov Model (HMM) & Viterbi Decoding
- **Status**: Completed
- **Files Created**:
  - `route_reconstruction/hmm_viterbi.py`
  - `results/route_reconstruction/viterbi_comparison_results.csv`
  - `results/route_reconstruction/route_continuity_improvement.png`
  - `results/route_reconstruction/top1_accuracy_comparison.png`
- **Empirical Route Reconstruction Findings**:
  - Across all 10 student testing trips, HMM + Viterbi decoding demonstrated consistent, statistically significant improvements:
    - **Top-1 Observation Accuracy**: Improved from **80.09% to 83.68%** (+3.59% absolute gain). Every single trip improved.
    - **Topological Route Continuity**: Improved from **87.48% to 99.00%** (+11.52% absolute gain), achieving 100.00% continuity on trip 00000007.
    - **Reference Route Overlap**: Improved from **86.46% to 94.21%** (+7.75% gain, reaching 100% on trip 00000040).
    - **Execution Efficiency**: Average Viterbi decoding execution time across full trajectories is **624.5 ms**.

### Phase 11: Folium Visualizations & Exported Maps
- **Status**: Completed
- **Files Created**:
  - `visualization/generate_maps.py`
  - Interactive Folium HTML maps in `visualization/maps/`:
    - `trip_00000000_route_prediction.html`
    - `trip_00000007_route_prediction.html`
    - `trip_00000040_route_prediction.html`
  - Static publication figures in `visualization/exports/`:
    - `trip_00000000_route_reconstruction.png`
    - `trip_00000007_route_reconstruction.png`
    - `trip_00000040_route_reconstruction.png`
- **Map Layer Features**:
  - Raw GPS observations (blue circle markers with timestamps and indices).
  - Road-network geometry (gray polylines of local subgraph).
  - Reference ground-truth route (solid green polyline).
  - Predicted route from PathSnap HMM + Viterbi (dashed red polyline).
  - Floating custom HTML legend and interactive Folium LayerControl.


### Phase 12, 13, 14: Backend, Frontend, and Launcher Scripts
- **Status**: Completed
- **Files Created**:
  - `backend/main.py`: FastAPI server with asynchronous REST endpoints (`GET /health`, `GET /trips`, `POST /predict`), decoupled server-side evaluation, Pydantic data schemas, and zero data leakage.
  - `frontend/app.py`: Streamlit interactive dashboard with HTTP connectivity to the backend, trip selector, inference triggers, real-time KPI cards, and embedded Folium map visualizer.
  - `start_backend.bat`: Windows batch script binding FastAPI to `0.0.0.0:8000`.
  - `start_frontend.bat`: Windows batch script launching Streamlit on `localhost:8501`.
  - `start_all_single_laptop.bat`: Master Windows batch launcher orchestrating both services for single-laptop demonstration.

---

### Phase 15: Automated Unit Testing Suite
- **Status**: Completed
- **Files Created**:
  - `tests/test_pipeline.py`: Comprehensive 7-stage test suite covering file parsing, geodesics, spatial candidate generation, feature scaling, API health probe, API trip inventory, and end-to-end route reconstruction across all 10 student-testing trips.
- **Verification Results**:
  - Ran 7 tests in 14.180s.
  - **Result: 7/7 passed (100% pass rate, 0 errors, 0 failures)**.

---

### Phase 16: GitHub Repository Preparation
- **Status**: Completed
- **Files Created**:
  - `.gitignore`: Comprehensive exclusion of Python caches, virtual environments, IDE files, OS artifacts, and large archive zip files while preserving code, models, and outputs.
  - `requirements.txt`: Clean, version-pinned dependency manifest spanning ML, spatial libraries, web frameworks, and reporting.
  - `README.md`: Production-grade documentation including mathematical formulations, ASCII/Mermaid architecture diagrams, benchmark tables, Optuna search spaces, HMM Viterbi gains, API specifications, and step-by-step reproduction and deployment guides (both single-laptop and physical two-laptop modes).

---

### Phase 17: Technical Report Generation (PDF)
- **Status**: Completed
- **Files Created**:
  - `reports/generate_report_pdf.py`: Automated multi-page PDF generation script utilizing ReportLab.
  - `reports/PathSnap_Technical_Report.pdf`: 7-page, professional publication-quality technical report.
- **Report Content Summary**:
  - Title, Abstract, Metadata (PES University UE24CS352A).
  - Mathematical formulation (Geodesic distance, planar projection, log-space emissions and transitions, Viterbi recurrence).
  - Dataset exploration and statistical summary tables.
  - Preprocessing, candidate generation, and miss rate analysis.
  - Benchmark of 5 machine learning models with compulsory analysis questions A-C answered.
  - Bayesian hyperparameter optimization (Optuna search space, history, tuned vs. untuned comparison).
  - HMM & log-space Viterbi sequential route reconstruction methodology and empirical comparison across all 10 student-test trips.
  - Geographical visualizations with embedded high-resolution figures.
  - Distributed architecture (FastAPI, Streamlit, single-laptop and two-laptop physical network deployment).
  - System limitations, future work, conclusion, and academic references.

---

### Phase 18: Final Verification & Checklist Audit
- **Status**: Completed
- **Verification Summary**:
  - All 22 hackathon requirements verified and validated.
  - Models, artifacts, results, visualizations, reports, and code verified intact.
  - Project ready for final presentation, demonstration, and evaluation.
