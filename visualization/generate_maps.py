"""
PathSnap Multi-Layer Map & Route Visualization Pipeline
Generates comprehensive interactive Folium maps and publication-quality static
route comparison figures for the technical report and Streamlit frontend.
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

import folium
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_trip, align_track_to_reference_route
from candidate_generation import CandidateGenerator
from feature_engineering import FeaturePipeline
from hmm_viterbi import HMMViterbiDecoder

def generate_trip_visualization(trip_id, data_dir='data', partition='student_test', output_dir='visualization/maps', img_dir='visualization/exports'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    trip_data = load_trip(trip_id, data_dir, partition=partition)
    
    # Load optimal model and pipeline
    model_path = os.path.join('models', 'final_model.joblib')
    pipe_path = os.path.join('artifacts', 'feature_pipeline.joblib')
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
    
    true_assigned = align_track_to_reference_route(trip_data)
    cand_df, miss_rate = generator.generate_candidates_for_trip(trip_data, true_assigned)
    predicted_arcs, clf_only_arcs, _ = decoder.decode_trip(trip_data, cand_df)
    
    # Coordinates lookup
    node_coords = dict(zip(trip_data.nodes.index, zip(trip_data.nodes.lat, trip_data.nodes.lon)))
    
    # Bounding Box
    center_lat = float(trip_data.track.lat.mean())
    center_lon = float(trip_data.track.lon.mean())
    
    # -------------------------------------------------------------
    # 1. Interactive Folium Map
    # -------------------------------------------------------------
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
    
    # Layer A: Road network (arcs within 0.008 deg of trajectory)
    min_lat, max_lat = trip_data.track.lat.min() - 0.008, trip_data.track.lat.max() + 0.008
    min_lon, max_lon = trip_data.track.lon.min() - 0.008, trip_data.track.lon.max() + 0.008
    
    road_layer = folium.FeatureGroup(name="Road Network (Arcs)", show=False)
    drawn = 0
    for arc_id, row in trip_data.arcs.iterrows():
        u, v = int(row.from_node), int(row.to_node)
        lat1, lon1 = node_coords[u]
        lat2, lon2 = node_coords[v]
        if (min_lat <= lat1 <= max_lat and min_lon <= lon1 <= max_lon) or \
           (min_lat <= lat2 <= max_lat and min_lon <= lon2 <= max_lon):
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='#aaaaaa', weight=1.5, opacity=0.5).add_to(road_layer)
            drawn += 1
            if drawn > 6000:
                break
    road_layer.add_to(m)
    
    # Layer B: Reference Route (Ground Truth)
    if trip_data.route is not None:
        ref_layer = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        ref_coords = []
        for arc_id in trip_data.route['arc_id']:
            u = trip_data.arcs.iloc[arc_id].from_node
            v = trip_data.arcs.iloc[arc_id].to_node
            lat1, lon1 = node_coords[u]
            lat2, lon2 = node_coords[v]
            if not ref_coords:
                ref_coords.append((lat1, lon1))
            ref_coords.append((lat2, lon2))
        folium.PolyLine(ref_coords, color='#2ca02c', weight=6, opacity=0.8, tooltip=f"Reference Route ({len(trip_data.route)} arcs)").add_to(ref_layer)
        ref_layer.add_to(m)
        
    # Layer C: Predicted Route (HMM + Viterbi)
    pred_layer = folium.FeatureGroup(name="Predicted Route (PathSnap HMM+Viterbi)", show=True)
    pred_coords = []
    for arc_id in predicted_arcs:
        u = trip_data.arcs.iloc[arc_id].from_node
        v = trip_data.arcs.iloc[arc_id].to_node
        lat1, lon1 = node_coords[u]
        lat2, lon2 = node_coords[v]
        if not pred_coords:
            pred_coords.append((lat1, lon1))
        pred_coords.append((lat2, lon2))
    folium.PolyLine(pred_coords, color='#d62728', weight=3.5, opacity=0.9, dash_array='5, 5', tooltip="Predicted Route (HMM+Viterbi)").add_to(pred_layer)
    pred_layer.add_to(m)
    
    # Layer D: Original GPS Observations
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    step = 1 if len(trip_data.track) < 1500 else 2
    for i in range(0, len(trip_data.track), step):
        row = trip_data.track.iloc[i]
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color='#1f77b4',
            fill=True,
            fill_color='#1f77b4',
            fill_opacity=0.8,
            tooltip=f"GPS Point #{i} (t={row.time}s)"
        ).add_to(gps_layer)
    gps_layer.add_to(m)
    
    # Start / End markers
    folium.Marker([trip_data.track.lat.iloc[0], trip_data.track.lon.iloc[0]], popup=f"Trip {trip_id} Start", icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker([trip_data.track.lat.iloc[-1], trip_data.track.lon.iloc[-1]], popup=f"Trip {trip_id} End", icon=folium.Icon(color='red', icon='stop')).add_to(m)
    
    # Add floating HTML legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 230px; height: 130px; 
                background-color: white; border:2px solid #888; z-index:9999; 
                font-size:12px; font-family: Arial; padding: 10px; border-radius: 6px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b>PathSnap Map Legend</b><br>
        <span style="color:#1f77b4; font-size:16px;">●</span> Raw GPS Observations<br>
        <span style="color:#2ca02c; font-size:18px;">━</span> Reference Route (Ground Truth)<br>
        <span style="color:#d62728; font-size:18px;">┅</span> Predicted Route (HMM+Viterbi)<br>
        <span style="color:#aaaaaa; font-size:14px;">━</span> Road Network Arcs
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    
    html_path = os.path.join(output_dir, f"trip_{trip_id}_route_prediction.html")
    m.save(html_path)
    print(f"Interactive Folium map saved to {html_path}")
    
    # -------------------------------------------------------------
    # 2. Static Publication-Quality Plot for Technical Report PDF
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot GPS points
    ax.scatter(trip_data.track.lon, trip_data.track.lat, c='#1f77b4', s=12, alpha=0.6, label='Raw GPS Points', zorder=2)
    
    # Plot Ground Truth Route
    if trip_data.route is not None:
        ref_lons = [c[1] for c in ref_coords]
        ref_lats = [c[0] for c in ref_coords]
        ax.plot(ref_lons, ref_lats, color='#2ca02c', linewidth=3.5, alpha=0.75, label='Reference Route (Ground Truth)', zorder=3)
        
    # Plot Predicted Route
    pred_lons = [c[1] for c in pred_coords]
    pred_lats = [c[0] for c in pred_coords]
    ax.plot(pred_lons, pred_lats, color='#d62728', linestyle='--', linewidth=2.0, alpha=0.9, label='Predicted Route (HMM+Viterbi)', zorder=4)
    
    ax.scatter([trip_data.track.lon.iloc[0]], [trip_data.track.lat.iloc[0]], c='green', marker='^', s=120, label='Trip Origin', zorder=5)
    ax.scatter([trip_data.track.lon.iloc[-1]], [trip_data.track.lat.iloc[-1]], c='red', marker='v', s=120, label='Trip Destination', zorder=5)
    
    ax.set_title(f'PathSnap Route Reconstruction - Trip {trip_id}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (WGS84 Degrees)', fontsize=11)
    ax.set_ylabel('Latitude (WGS84 Degrees)', fontsize=11)
    ax.legend(frameon=True, loc='best', fontsize=10)
    plt.tight_layout()
    
    png_path = os.path.join(img_dir, f"trip_{trip_id}_route_reconstruction.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Publication-quality route map saved to {png_path}")

def generate_all_visualizations():
    print("Generating Folium maps and publication figures for representative evaluation trips...")
    sample_trips = ['00000000', '00000007', '00000040']
    for tid in sample_trips:
        generate_trip_visualization(tid, partition='student_test')
    print("Map visualization generation complete.")

if __name__ == '__main__':
    generate_all_visualizations()
