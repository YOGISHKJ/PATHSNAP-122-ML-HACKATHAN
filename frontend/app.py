"""
PathSnap Streamlit Frontend
Application: PathSnap – Intelligent GPS Map Matching
Communicates strictly via HTTP requests with the FastAPI backend service.
"""

import os
import sys

# Ensure root and src are in path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import folium

# Set page configuration
st.set_page_config(
    page_title="PathSnap – Intelligent GPS Map Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Backend URL configuration (defaults to local, overridable via environment variable)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/100/route.png", width=80)
st.sidebar.title("PathSnap Controls")
st.sidebar.markdown("**Team ID: PT-122** | PES University ML Hackathon")

# Backend Health Check
@st.cache_data(ttl=5)
def check_backend_health():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"detail": f"Status {resp.status_code}"}
    except Exception as e:
        return False, {"detail": str(e)}

is_healthy, health_info = check_backend_health()

if is_healthy:
    st.sidebar.success(f"Backend Connected ({BACKEND_URL})")
    st.sidebar.caption(f"Engine: {health_info.get('model_name', 'Optimal XGBoost+HMM')}")
else:
    st.sidebar.error(f"Backend Offline ({BACKEND_URL})")
    st.sidebar.caption(f"Error: {health_info.get('detail', 'Connection refused')}")
    st.sidebar.info("Please start the backend service on port 8000:\n`python backend/main.py`")

# Partition Selection
partition = st.sidebar.selectbox(
    "Dataset Partition",
    options=["student_test", "train"],
    index=0,
    help="Select evaluation partition. Student Test contains the 10 official reserved evaluation trips."
)

# Fetch Available Trips
@st.cache_data(ttl=10)
def fetch_trips(part):
    try:
        resp = requests.get(f"{BACKEND_URL}/trips?partition={part}", timeout=4)
        if resp.status_code == 200:
            return resp.json().get("trip_ids", [])
    except Exception:
        pass
    # Local fallback if backend warming up
    part_dir = os.path.join(root_dir, 'data', part)
    if os.path.exists(part_dir):
        return sorted([d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))])
    return []

trips = fetch_trips(partition)
selected_trip = st.sidebar.selectbox("Select Trip ID", options=trips if trips else ["00000000"])

# Model architecture info in sidebar
with st.sidebar.expander("System Architecture Details", expanded=False):
    st.markdown("""
    - **Classifier**: Bayesian-Tuned XGBoost (Optuna TPE)
    - **Features**: 13 Candidate-level kinematic & spatial features
    - **Decoder**: Log-Space Viterbi Dynamic Programming
    - **Transition Model**: Graph Topology Adjacency + Gap Handling
    - **Spatial Index**: Fast 2D Vector cKDTree Projection
    """)

# Main Page Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-header">PathSnap – Intelligent GPS Map Matching</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Distributed Vehicle Trajectory Reconstruction & Road-Network Matching Engine</div>', unsafe_allow_html=True)
with col_status:
    if is_healthy:
        st.markdown('<div style="text-align: right; padding-top: 15px;"><span class="badge-success">● API ONLINE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: right; padding-top: 15px;"><span style="background-color:#fee2e2; color:#b91c1c; padding:4px 10px; border-radius:12px; font-weight:600;">● API OFFLINE</span></div>', unsafe_allow_html=True)

st.markdown("---")

# Main Action Button
col_btn, col_info = st.columns([1, 4])
with col_btn:
    predict_clicked = st.button("🚀 Predict Route", type="primary", use_container_width=True, disabled=not is_healthy)
with col_info:
    st.caption(f"Selected Trip: **{selected_trip}** | Partition: **{partition}** | Architecture: **FastAPI + Streamlit HTTP Decoupled Client**")

if predict_clicked:
    with st.spinner(f"Requesting map-matching inference from backend API for Trip {selected_trip}..."):
        t0 = time.time()
        payload = {"trip_id": selected_trip, "partition": partition}
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=20)
            client_elapsed_ms = (time.time() - t0) * 1000.0
            
            if resp.status_code == 200:
                data = resp.json()
                st.session_state['pred_data'] = data
                st.session_state['client_time_ms'] = client_elapsed_ms
                st.session_state['trip_id'] = selected_trip
                st.success(f"Route successfully predicted in {data.get('execution_time_ms', 0):.1f} ms (API Roundtrip: {client_elapsed_ms:.1f} ms)!")
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API: {e}")

# Render Results Dashboard if data present in session state
if 'pred_data' in st.session_state and st.session_state.get('trip_id') == selected_trip:
    data = st.session_state['pred_data']
    traj = data.get('trajectory_stats', {})
    metrics = data.get('evaluation_metrics')
    
    st.markdown("### 📊 Trajectory & Model Performance Metrics")
    
    # Row 1: Trajectory Kinematics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("GPS Observations", f"{traj.get('n_observations', 0):,}")
    with c2:
        st.metric("Trajectory Length", f"{traj.get('trajectory_length_km', 0):.2f} km")
    with c3:
        st.metric("Average Speed", f"{traj.get('mean_speed_kmh', 0):.1f} km/h")
    with c4:
        st.metric("Duration", f"{traj.get('duration_s', 0):.0f} sec")
        
    st.markdown("")
    
    # Row 2: Evaluation Metrics (when ground truth permitted)
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Top-1 Accuracy", f"{metrics.get('top1_accuracy_percent', 0):.2f}%", help="Percentage of GPS observations correctly matched to true road segment")
        with m2:
            st.metric("Route Continuity", f"{metrics.get('route_continuity_percent', 0):.2f}%", help="Percentage of consecutive predicted arcs connected in road graph")
        with m3:
            st.metric("Reference Route Overlap", f"{metrics.get('reference_overlap_percent', 0):.2f}%", help="Fraction of reference route road segments recovered")
        with m4:
            st.metric("Inference Engine Time", f"{data.get('execution_time_ms', 0):.1f} ms", help="Server-side end-to-end inference execution latency")
            
    st.markdown("---")
    
    # Section 3: Interactive Map Display
    st.markdown("### 🗺️ Interactive Folium Route Map")
    st.caption("Layer Control allows toggling Raw GPS Observations, Road Network Arcs, Reference Ground Truth, and Predicted Path.")
    
    # Check if pre-rendered Folium HTML map exists
    map_file = os.path.join(root_dir, 'visualization', 'maps', f"trip_{selected_trip}_route_prediction.html")
    if os.path.exists(map_file):
        with open(map_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=650, scrolling=True)
    else:
        # Fallback to general eda map or notification
        eda_map = os.path.join(root_dir, 'visualization', 'maps', f"trip_{selected_trip}_map.html")
        if os.path.exists(eda_map):
            with open(eda_map, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=650, scrolling=True)
        else:
            st.info("Generating real-time interactive map view...")
            # We can run visualization script for this trip
            from visualization.generate_maps import generate_trip_visualization
            generate_trip_visualization(selected_trip, partition=partition)
            if os.path.exists(map_file):
                with open(map_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=650, scrolling=True)
                
    # Section 4: Predicted Road Arc Sequence
    with st.expander("📋 View Raw Predicted Road Arc Sequence (IDs)", expanded=False):
        arcs = data.get('predicted_route_arcs', [])
        st.write(f"Total Sequence Length: **{len(arcs)}** arcs | Unique Segments: **{data.get('unique_predicted_arcs', 0)}**")
        st.code(str(arcs[:100]) + ("..." if len(arcs) > 100 else ""))
else:
    st.info("👉 Select a trip from the sidebar and click **'🚀 Predict Route'** to run end-to-end map-matching and route reconstruction.")
    
    # Display preview image if available
    preview_img = os.path.join(root_dir, 'visualization', 'exports', 'trip_00000000_route_reconstruction.png')
    if os.path.exists(preview_img):
        st.image(preview_img, caption="Representative PathSnap Map Matching Visualization (Trip 00000000)", use_container_width=True)
