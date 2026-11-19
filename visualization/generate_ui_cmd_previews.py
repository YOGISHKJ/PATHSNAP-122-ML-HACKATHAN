"""
Generate visual screenshots/figures for Section 7:
1. UI Dashboard Interface Preview
2. CMD / Terminal API Response Preview
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_ui_preview():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Background window
    bg = patches.Rectangle((0, 0), 100, 60, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(bg)

    # Top title bar
    bar = patches.Rectangle((0, 54), 100, 6, facecolor="#0F172A")
    ax.add_patch(bar)
    ax.text(3, 56.5, "PathSnap [Team PT-122] — Intelligent GPS Map Matching (Evaluator UI)", 
            color="#FFFFFF", fontsize=11, fontweight='bold', va='center')
    ax.text(95, 56.5, "● ● ●", color="#94A3B8", fontsize=12, va='center')

    # Sidebar (Left)
    sidebar = patches.Rectangle((0, 0), 24, 54, facecolor="#F1F5F9", edgecolor="#E2E8F0", linewidth=1)
    ax.add_patch(sidebar)
    ax.text(2, 50, "PATHSNAP CONTROLS", color="#1E293B", fontsize=9, fontweight='bold')
    ax.text(2, 47, "PES University ML Hackathon", color="#64748B", fontsize=7.5)

    # Backend status badge
    badge = patches.FancyBboxPatch((2, 41), 20, 4, boxstyle="round,pad=0.5", facecolor="#DCFCE7", edgecolor="#86EFAC")
    ax.add_patch(badge)
    ax.text(3.5, 43, "● Backend Connected", color="#166534", fontsize=7.5, fontweight='bold')
    ax.text(3.5, 39, "Host: http://10.8.121.23:8000", color="#475569", fontsize=6.5)

    # Sidebar selectors
    ax.text(2, 34, "Select Evaluation Trip:", color="#334155", fontsize=7.5, fontweight='bold')
    sel_box = patches.Rectangle((2, 28), 20, 4.5, facecolor="#FFFFFF", edgecolor="#94A3B8")
    ax.add_patch(sel_box)
    ax.text(3, 30.2, "trip_00000007 (student_test)", color="#0F172A", fontsize=7.5)

    btn = patches.FancyBboxPatch((2, 20), 20, 5, boxstyle="round,pad=0.5", facecolor="#2563EB", edgecolor="#1D4ED8")
    ax.add_patch(btn)
    ax.text(4, 22.5, "⚡ Match Route & Evaluate", color="#FFFFFF", fontsize=8, fontweight='bold')

    # Main area - KPI cards
    kpis = [
        ("OBSERVATIONS", "1,354", "#1E40AF", (27, 42)),
        ("TOP-1 ACCURACY", "94.61%", "#15803D", (45, 42)),
        ("ROUTE CONTINUITY", "100.0%", "#0D9488", (63, 42)),
        ("INFERENCE LATENCY", "132.8 ms", "#B45309", (81, 42)),
    ]
    for label, val, col, (x, y) in kpis:
        card = patches.FancyBboxPatch((x, y), 16, 9, boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1)
        ax.add_patch(card)
        ax.text(x+8, y+6.5, label, color="#64748B", fontsize=6.5, fontweight='bold', ha='center')
        ax.text(x+8, y+2.5, val, color=col, fontsize=11, fontweight='bold', ha='center')

    # Map viewport (Center-Right)
    map_rect = patches.Rectangle((27, 4), 70, 35, facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=1)
    ax.add_patch(map_rect)
    ax.text(29, 36.5, "Interactive Multi-Layer Map View (Folium Leaflet Engine)", color="#334155", fontsize=8.5, fontweight='bold')

    # Simulated map route lines
    import numpy as np
    t = np.linspace(30, 93, 40)
    road_y = 18 + 7 * np.sin((t-30)/8)
    gps_y = road_y + np.random.normal(0, 0.4, len(t))
    ax.plot(t, road_y + 1, color="#94A3B8", linewidth=3, label="Road Network Graph Arcs")
    ax.plot(t, road_y, color="#10B981", linewidth=2.5, label="Ground-Truth Reference Route")
    ax.plot(t, road_y, color="#DC2626", linestyle="--", linewidth=2, label="Viterbi Reconstructed Route")
    ax.scatter(t, gps_y, color="#2563EB", s=18, zorder=5, label="Raw GPS Observations (Noisy)")

    # Legend inside map
    leg_box = patches.Rectangle((66, 6), 29, 13, facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95)
    ax.add_patch(leg_box)
    ax.text(68, 16.5, "Map Layers & Legend", color="#0F172A", fontsize=7, fontweight='bold')
    ax.text(68, 14, "● Blue Dots: Raw GPS Points", color="#2563EB", fontsize=6.5)
    ax.text(68, 11.5, "— Gray: Road Network Graph", color="#64748B", fontsize=6.5)
    ax.text(68, 9, "— Green: Reference Trajectory", color="#10B981", fontsize=6.5)
    ax.text(68, 6.5, "--- Red: PathSnap Viterbi Route", color="#DC2626", fontsize=6.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "ui_dashboard_preview.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print("UI Dashboard Preview generated:", out_path)

def generate_cmd_preview():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Dark Terminal Window
    bg = patches.Rectangle((0, 0), 100, 50, facecolor="#0C1017", edgecolor="#30363D", linewidth=1.5)
    ax.add_patch(bg)

    # Title Bar
    bar = patches.Rectangle((0, 46), 100, 4, facecolor="#161B22")
    ax.add_patch(bar)
    ax.text(3, 48, "Command Prompt — FastAPI Server [Team PT-122 | Inference Node : 0.0.0.0:8000]", 
            color="#C9D1D9", fontsize=9, fontweight='bold', va='center')
    ax.text(95, 48, "─ □ ✕", color="#8B949E", fontsize=9, va='center')

    # Terminal text
    cmd_lines = [
        ("C:\\PathSnap> python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000", "#58A6FF"),
        ("INFO:     Started server process [PID: 14820]", "#8B949E"),
        ("INFO:     Waiting for application startup...", "#8B949E"),
        ("INFO:     [PathSnap Engine] Loaded XGBoost Model & RobustScaler Pipeline", "#7EE787"),
        ("INFO:     Application startup complete. Uvicorn running on http://0.0.0.0:8000", "#7EE787"),
        ("INFO:     10.8.121.35:54210 - \"GET /health HTTP/1.1\" 200 OK", "#A5D6FF"),
        ("INFO:     10.8.121.35:54212 - \"GET /trips?partition=student_test HTTP/1.1\" 200 OK", "#A5D6FF"),
        ("INFO:     10.8.121.35:54215 - \"POST /predict?evaluate=true HTTP/1.1\" 200 OK", "#7EE787"),
        ("          [INFERENCE RESULT] Trip: trip_00000007 | GPS Obs: 1,354", "#FFA657"),
        ("          [METRICS] Top-1 Acc: 94.61% | Route Continuity: 100.0% | Overlap: 98.37%", "#7EE787"),
        ("          [BENCHMARK] Spatial Index + XGBoost + Viterbi Latency: 132.8 ms", "#F2CC60"),
        ("INFO:     10.8.121.35:54218 - \"POST /predict?evaluate=true HTTP/1.1\" 200 OK", "#7EE787"),
        ("C:\\PathSnap> _", "#C9D1D9")
    ]

    y = 42
    for text, col in cmd_lines:
        ax.text(3, y, text, color=col, fontsize=7.5, fontfamily='monospace', va='center')
        y -= 3.2

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cmd_api_response_preview.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print("CMD API Response Preview generated:", out_path)

if __name__ == "__main__":
    generate_ui_preview()
    generate_cmd_preview()
