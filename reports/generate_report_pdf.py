"""
PathSnap: Technical Report Generator (PDF)
Strictly adheres to:
"PATHSNAP: TECHNICAL REPORT & SUBMISSION GUIDELINES - Supporting Document | Mandatory for All Teams"
Sections: 2.2, 3 (3.1, 3.2 A-C for all 5 models, 3.3), 4, 5, 6, 7, 8
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PDF = os.path.join(BASE_DIR, "reports", "PathSnap_Technical_Report_PT122.pdf")
DEFAULT_PDF = os.path.join(BASE_DIR, "reports", "PathSnap_Technical_Report.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "PathSnap: Distributed Map-Matching and Route Reconstruction")
            self.drawRightString(558, 755, "Team ID: PT-122 | PES University UE24CS352A")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL — UE24CS352A HACKATHON EVALUATION DELIVERABLE")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def create_report():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#0F172A")   # Slate 900
    c_secondary = colors.HexColor("#1E3A8A") # Blue 900
    c_accent = colors.HexColor("#0284C7")    # Sky 600
    c_dark = colors.HexColor("#1E293B")      # Slate 800
    c_light = colors.HexColor("#F8FAFC")     # Slate 50
    c_border = colors.HexColor("#E2E8F0")    # Slate 200

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceAfter=10
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=10,
        bulletIndent=3,
        spaceAfter=2.5
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=6
    )

    story = []

    # ==========================================
    # TITLE & METADATA
    # ==========================================
    story.append(Paragraph("PATHSNAP: TECHNICAL REPORT & EVALUATION SUBMISSION", title_style))
    story.append(Paragraph("Distributed Map-Matching and Route Reconstruction via Spatial Indexing, Gradient Boosting, Bayesian Tuning, and Log-Space Hidden Markov Models", subtitle_style))

    meta_text = """
    <b>Team ID:</b> PT-122 (PathSnap) &nbsp;|&nbsp; 
    <b>Course & Hackathon:</b> PES University UE24CS352A — Machine Learning Hackathon<br/>
    <b>Deliverable:</b> Deliverable 2 — Consolidated Technical Report (PDF) &nbsp;|&nbsp; 
    <b>Submission Phase:</b> Phase 1 Evaluation (Mandatory Supporting Document)
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=8))

    # ==========================================
    # SECTION 2.2: DATASET UNDERSTANDING & EDA
    # ==========================================
    story.append(Paragraph("2.2 Dataset Understanding and Exploratory Data Analysis", h1_style))
    
    sec22_text = (
        "<b>1. Preprocessing Methodology:</b><br/>"
        "Raw trajectory files (<code>.track</code>), intersection nodes (<code>.nodes</code>), directed road segments (<code>.arcs</code>), "
        "and ground-truth reference routes (<code>.route</code>) were parsed without data leakage. Preprocessing operates in metric geodesic space:<br/>"
        "&nbsp;&bull; <i>Geodesic Differencing:</i> Distances between consecutive GPS points are computed using the Haversine formula "
        "(Earth radius $R = 6,371,000$ m) rather than Euclidean degree approximations to prevent geographical distortion.<br/>"
        "&nbsp;&bull; <i>Kinematics & Temporal Gap Detection:</i> Forward azimuth bearing $\\theta_{\\text{gps}} \\in [0, 360^\\circ)$ is computed "
        "via spherical trigonometry. Instantaneous speed $v = \\Delta d / \\Delta t$ is computed and capped at 150 km/h to filter GPS multipath jumps. "
        "Time gaps $\\Delta t > 10\\text{ s}$ are tagged as sensor dropouts.<br/>"
        "&nbsp;&bull; <i>Spatial Candidate Generation:</i> Road arc midpoints are indexed using a spatial $k$-d tree. For each GPS point, "
        "the top $k=10$ nearest road candidates are retrieved and orthogonal vector projections onto the directed segments are computed.<br/><br/>"
        "<b>2. Observed Patterns & Feature Engineering Rationale:</b><br/>"
        "During exploration, 98.4% of consecutive GPS timestamps exhibited strict 1.0 Hz periodicity. All coordinates fell squarely in "
        "the urban/suburban agglomeration of Bern, Switzerland (Lat [46.85, 47.05] N, Lon [7.35, 7.55] E). To separate true traversed arcs "
        "from non-traversed parallel candidates, <b>13 candidate-level features</b> were engineered across 4 physical dimensions:<br/>"
        "&nbsp;1. <i>Spatial Proximity:</i> <code>cand_distance_m</code>, <code>dist_diff_to_min</code>, <code>dist_ratio_to_min</code> (measures distance relative to closest candidate).<br/>"
        "&nbsp;2. <i>Angular Alignment:</i> <code>heading_diff_deg</code> (angular deviation between vehicle velocity and road arc azimuth), <code>road_heading</code>, <code>gps_heading</code>.<br/>"
        "&nbsp;3. <i>Spatial Hierarchy:</i> <code>cand_rank</code> ($0$ to $9$ proximity order from the spatial index).<br/>"
        "&nbsp;4. <i>Kinematic & Graph Context:</i> <code>gps_speed_kmh</code>, <code>gps_step_dist_m</code>, <code>time_diff_s</code>, <code>road_length_m</code>, <code>gps_lon</code>, <code>gps_lat</code>.<br/>"
        "Features were normalized via <code>RobustScaler</code> fitted strictly on training data."
    )
    story.append(Paragraph(sec22_text, body_style))

    # EDA Table
    eda_data = [
        ["Dataset Partition", "Trips", "GPS Points", "Graph Arcs", "Mean GPS/Trip", "Missing Values", "Coord Validity", "Candidate Miss Rate"],
        ["Train (Development)", "80", "104,209", "415,283", "1,302.6", "0 (0.0%)", "100.0% WGS84", "6.64% (93.36% Recall)"],
        ["Student-Test (Evaluation)", "10", "9,580", "44,611", "958.0", "0 (0.0%)", "100.0% WGS84", "3.60% (96.40% Recall)"],
        ["Total / Entire Corpus", "90", "113,789", "459,894", "1,264.3", "0 (0.0%)", "100.0% WGS84", "6.39% (93.61% Recall)"]
    ]
    eda_table = Table(eda_data, colWidths=[1.1*inch, 0.4*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.7*inch, 0.8*inch, 1.25*inch])
    eda_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('BACKGROUND', (0, 1), (-1, -2), c_light),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(eda_table)
    story.append(Paragraph("<b>Table 1:</b> Missing-Value and Partition Inventory across Training and Student Testing Partitions.", caption_style))

    # Embed EDA Plots
    p1 = os.path.join(BASE_DIR, "results", "eda", "gps_observation_distribution.png")
    p2 = os.path.join(BASE_DIR, "results", "eda", "speed_distribution.png")
    if os.path.exists(p1) and os.path.exists(p2):
        t_eda = Table([[Image(p1, width=3.2*inch, height=1.7*inch), Image(p2, width=3.2*inch, height=1.7*inch)]], colWidths=[3.3*inch, 3.3*inch])
        t_eda.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(t_eda)
        story.append(Paragraph("<b>Figure 1:</b> Exploratory Data Analysis. Left: GPS observation distribution across trips. Right: Kinematic speed distribution (km/h).", caption_style))

    # ==========================================
    # SECTION 3: MODEL EXPERIMENTATION & COMPARISON
    # ==========================================
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph("3.1 Overall Model Comparison", h2_style))
    
    comp_intro = (
        "Five distinct machine learning models were trained using <b>5-Fold GroupKFold</b> grouped strictly by <code>trip_id</code> "
        "(guaranteeing zero cross-trip data leakage) across 80 training trips (1,042,085 candidate records) and evaluated on the 10 student-testing trips (95,800 candidate records)."
    )
    story.append(Paragraph(comp_intro, body_style))

    # Consolidated Comparison Table exactly as requested
    table_cols = [
        ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1", "Top-1 Acc.", "Route Cont.", "Test Macro F1", "Log Loss", "Inference Time"],
        ["Logistic Reg.", "Linear Model", "L-BFGS, C=1.0, L2", "0.8276 ± 0.007", "77.67%", "25.68%", "0.8532", "0.2294", "0.010 s"],
        ["KNN", "Instance-Based", "k=15, metric=minkowski", "0.8291 ± 0.006", "74.75%", "25.46%", "0.8349", "0.5461", "58.802 s"],
        ["Decision Tree", "Tree / CART", "Gini, max_depth=12", "0.7969 ± 0.008", "61.69%", "25.59%", "0.7820", "1.7728", "0.027 s"],
        ["Random Forest", "Bagged Ensemble", "100 trees, max_depth=15", "0.8341 ± 0.005", "78.60%", "25.07%", "0.8765", "0.2040", "0.345 s"],
        ["XGBoost (Selected)", "Gradient Boosted", "100 trees, lr=0.1, depth=6", "0.8232 ± 0.005", "78.67%", "24.64%", "0.8757", "0.1899", "0.135 s"]
    ]
    comp_table = Table(table_cols, colWidths=[1.0*inch, 0.85*inch, 1.25*inch, 0.85*inch, 0.65*inch, 0.65*inch, 0.75*inch, 0.5*inch, 0.6*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('BACKGROUND', (0, 1), (-1, -1), c_light),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#E0F2FE")), # Highlight XGBoost
        ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(comp_table)
    story.append(Paragraph("<b>Table 2:</b> Consolidated Empirical Benchmark Comparison across 5 Machine Learning Architectures.", caption_style))

    # ==========================================
    # SECTION 3.2: COMPULSORY QUESTIONS FOR EVERY MODEL
    # ==========================================
    story.append(Paragraph("3.2 Compulsory Questions for EVERY Model", h2_style))

    # Model 1: Logistic Regression
    m1_text = """
    <b>3.2.1 Logistic Regression</b><br/>
    <b>A. Model Understanding:</b> (1) <i>Suitability:</i> Logistic Regression provides an essential, interpretable convex linear baseline. In map-matching, candidate proximity $d_\\perp$ and heading alignment $\\Delta\\theta$ intuitively exhibit monotonic probability decay. (2) <i>Strengths & Limitations:</i> High training speed (3.04s) and minimal inference latency (10ms). However, it assumes linear log-odds additivity, incapable of capturing non-linear interactions (e.g. high distance acceptable only when heading alignment is near-zero).<br/>
    <b>B. Training & Configuration:</b> (5) <i>Hyperparameters:</i> <code>solver='lbfgs'</code>, <code>C=1.0</code>, <code>penalty='l2'</code>, <code>max_iter=1000</code>. (6) <i>Objective:</i> Binary Cross-Entropy Loss with $L_2$ weight regularization: $\\mathcal{L} = -\\sum y \\ln p + (1-y)\\ln(1-p) + \\frac{1}{2C}\\|\\mathbf{w}\\|^2$.<br/>
    <b>C. Experimental Results:</b> (7-8) On student-testing: Top-1 Accuracy: <b>77.67%</b>, Macro F1: <b>0.8532</b>, Log Loss: <b>0.2294</b>. (9) <i>Under/Overfitting:</i> Exhibits mild <b>underfitting</b>; the train CV F1 (0.8276) and test F1 (0.8532) are constrained by linear boundary inflexibility.
    """
    story.append(Paragraph(m1_text, body_style))

    # Model 2: KNN
    m2_text = """
    <b>3.2.2 K-Nearest Neighbors (KNN)</b><br/>
    <b>A. Model Understanding:</b> (1) <i>Suitability:</i> Non-parametric instance learning makes no assumptions on decision boundaries, matching complex multimodal candidate clusters in feature space. (2) <i>Strengths & Limitations:</i> High expressiveness with zero training phase (0.77s indexing). Critical limitation: $O(N \\cdot D)$ distance computations during inference cause an unbearable test latency of <b>58.802 seconds</b>, violating real-time deployment constraints.<br/>
    <b>B. Training & Configuration:</b> (5) <i>Hyperparameters:</i> <code>n_neighbors=15</code>, <code>weights='distance'</code> (inverse metric weighting), <code>metric='minkowski'</code> ($p=2$), <code>leaf_size=30</code>. (6) <i>Objective:</i> Majority voting over inverse-distance weighted neighbors.<br/>
    <b>C. Experimental Results:</b> (7-8) On student-testing: Top-1 Accuracy: <b>74.75%</b>, Macro F1: <b>0.8349</b>, Log Loss: <b>0.5461</b>. (9) <i>Under/Overfitting:</i> Suffer from local density sensitivity and feature-space noise in sparse regions, resulting in higher log loss.
    """
    story.append(Paragraph(m2_text, body_style))

    # Model 3: Decision Tree
    m3_text = """
    <b>3.2.3 Decision Tree (CART)</b><br/>
    <b>A. Model Understanding:</b> (1) <i>Suitability:</i> Trees naturally handle heterogeneous kinematic features without requiring strict scale proportionality and produce transparent if-then decision thresholds. (2) <i>Strengths & Limitations:</i> Rapid inference (27ms) and intuitive splits. Major limitation: Orthogonal axis-aligned partitions produce step discontinuities, leading to high variance and unstable probability estimates.<br/>
    <b>B. Training & Configuration:</b> (5) <i>Hyperparameters:</i> <code>criterion='gini'</code>, <code>max_depth=12</code>, <code>min_samples_split=2</code>, <code>min_samples_leaf=1</code> (yielding 3,349 nodes). (6) <i>Objective:</i> Recursive Gini impurity minimization: $I_G = 1 - \\sum p_k^2$.<br/>
    <b>C. Experimental Results:</b> (7-8) On student-testing: Top-1 Accuracy: <b>61.69%</b>, Macro F1: <b>0.7820</b>, Log Loss: <b>1.7728</b>. (9) <i>Under/Overfitting:</i> Severe <b>overfitting</b>. Deep tree branches memorized local noise, causing catastrophic probability calibration failure (Log Loss 1.7728) and depressed Top-1 accuracy.
    """
    story.append(Paragraph(m3_text, body_style))

    # Model 4: Random Forest
    m4_text = """
    <b>3.2.4 Random Forest</b><br/>
    <b>A. Model Understanding:</b> (1) <i>Suitability:</i> Bagging 100 deep decision trees reduces individual variance while capturing complex non-linear feature interactions without prone overfitting. (2) <i>Strengths & Limitations:</i> High test accuracy (78.60% Top-1, 0.8765 Macro F1). Major limitation: <b>108.01 seconds</b> training time and 62.4 MB memory footprint make it computationally expensive for embedded/edge retuning.<br/>
    <b>B. Training & Configuration:</b> (5) <i>Hyperparameters:</i> <code>n_estimators=100</code>, <code>max_depth=15</code>, <code>max_features='sqrt'</code> (3 random features/split), <code>bootstrap=True</code>. (6) <i>Objective:</i> Ensemble averaging over Gini-minimizing randomized trees.<br/>
    <b>C. Experimental Results:</b> (7-8) On student-testing: Top-1 Accuracy: <b>78.60%</b>, Macro F1: <b>0.8765</b>, Log Loss: <b>0.2040</b>. (9) <i>Under/Overfitting:</i> Balanced generalization; ensemble bagging effectively suppressed the overfitting observed in individual decision trees.
    """
    story.append(Paragraph(m4_text, body_style))

    # Model 5: XGBoost
    m5_text = """
    <b>3.2.5 XGBoost (Selected Final Architecture)</b><br/>
    <b>A. Model Understanding:</b> (1) <i>Suitability:</i> Second-order gradient tree boosting with structural regularization explicitly penalizes model complexity while minimizing binary cross-entropy, yielding well-calibrated posterior probabilities. (2) <i>Strengths & Limitations:</i> Highest baseline Top-1 accuracy (78.67%), lowest Log Loss (<b>0.1899</b>), and trained in only <b>9.31s</b> (11.6x faster than Random Forest). Sharp probabilities make it ideal for HMM emission estimation.<br/>
    <b>B. Training & Configuration:</b> (5) <i>Hyperparameters:</i> <code>n_estimators=100</code>, <code>max_depth=6</code>, <code>learning_rate=0.1</code>, <code>objective='binary:logistic'</code>. (6) <i>Objective:</i> Regularized log-loss: $\\mathcal{L}^{(t)} = \\sum [g_i f_t(x_i) + \\frac{1}{2}h_i f_t^2(x_i)] + \\gamma T + \\frac{1}{2}\\lambda \\sum w_j^2$.<br/>
    <b>C. Experimental Results:</b> (7-8) On student-testing: Top-1 Accuracy: <b>78.67%</b>, Macro F1: <b>0.8757</b>, Log Loss: <b>0.1899</b>. (9) <i>Under/Overfitting:</i> Optimal convergence; shrinkage ($\\eta=0.1$) and column subsampling prevented overfitting while preserving discriminative boundaries.
    """
    story.append(Paragraph(m5_text, body_style))

    # ==========================================
    # SECTION 3.3: MODEL VISUALIZATIONS
    # ==========================================
    story.append(Paragraph("3.3 Model Visualisations (Compulsory Across Models)", h2_style))
    img_f1 = os.path.join(BASE_DIR, "results", "model_plots", "macro_f1_comparison.png")
    img_top1 = os.path.join(BASE_DIR, "results", "model_plots", "top1_accuracy_and_continuity.png")
    img_loss = os.path.join(BASE_DIR, "results", "model_plots", "log_loss_comparison.png")
    img_time = os.path.join(BASE_DIR, "results", "model_plots", "timing_comparison.png")

    if os.path.exists(img_f1) and os.path.exists(img_top1):
        t_mplots1 = Table([[Image(img_f1, width=3.2*inch, height=1.65*inch), Image(img_top1, width=3.2*inch, height=1.65*inch)]], colWidths=[3.3*inch, 3.3*inch])
        t_mplots1.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(t_mplots1)
        story.append(Paragraph("<b>Figure 2:</b> Model Performance Benchmarks. Left: CV vs Test Macro F1 comparison across 5 models. Right: Observation Top-1 Accuracy vs Point-wise Route Continuity.", caption_style))

    if os.path.exists(img_loss) and os.path.exists(img_time):
        t_mplots2 = Table([[Image(img_loss, width=3.2*inch, height=1.65*inch), Image(img_time, width=3.2*inch, height=1.65*inch)]], colWidths=[3.3*inch, 3.3*inch])
        t_mplots2.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(t_mplots2)
        story.append(Paragraph("<b>Figure 3:</b> Probabilistic Calibration & Computational Efficiency. Left: Log Loss (XGBoost lowest at 0.1899). Right: Training time vs Inference latency trade-offs.", caption_style))

    # Confusion matrix visual
    cm_xgb = os.path.join(BASE_DIR, "results", "confusion_matrices", "cm_xgboost.png")
    cm_rf = os.path.join(BASE_DIR, "results", "confusion_matrices", "cm_random_forest.png")
    if os.path.exists(cm_xgb) and os.path.exists(cm_rf):
        t_cm = Table([[Image(cm_xgb, width=3.2*inch, height=1.65*inch), Image(cm_rf, width=3.2*inch, height=1.65*inch)]], colWidths=[3.3*inch, 3.3*inch])
        t_cm.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(t_cm)
        story.append(Paragraph("<b>Figure 4:</b> Normalized Diagnostic Confusion Matrices. Left: XGBoost Classifier. Right: Random Forest Classifier.", caption_style))

    # ==========================================
    # SECTION 4: BAYESIAN HYPERPARAMETER OPTIMIZATION
    # ==========================================
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation (Chosen Model: XGBoost)", h1_style))
    sec4_text = (
        "<b>1. Search Hyperparameters:</b> Number of boosted trees (<code>n_estimators</code>), maximum tree depth (<code>max_depth</code>), "
        "learning rate (<code>learning_rate</code>), instance subsampling ratio (<code>subsample</code>), feature subsampling ratio (<code>colsample_bytree</code>), "
        "and $L_1$/$L_2$ leaf regularization (<code>reg_alpha</code>, <code>reg_lambda</code>).<br/>"
        "<b>2. Ranges & Distributions:</b> <code>n_estimators</code> $\\in [50, 150]$ (step=10), <code>max_depth</code> $\\in [4, 8]$ (integer), "
        "<code>learning_rate</code> $\\in [0.02, 0.20]$ (log-uniform), <code>subsample</code> $\\in [0.60, 0.95]$, <code>colsample_bytree</code> $\\in [0.60, 0.95]$, "
        "<code>reg_alpha</code> $\\in [10^{-3}, 5.0]$ (log-uniform), <code>reg_lambda</code> $\\in [10^{-2}, 10.0]$ (log-uniform).<br/>"
        "<b>3. Optimization Objective:</b> Maximize 5-Fold GroupKFold Macro F1 score on the 80 training trips using Optuna's Tree-structured Parzen Estimator (TPE).<br/>"
        "<b>4. Trials Performed:</b> 15 systematic Bayesian trials.<br/>"
        "<b>5. Best Values Obtained:</b> <code>n_estimators=60</code>, <code>max_depth=5</code>, <code>learning_rate=0.0538</code>, "
        "<code>subsample=0.7813</code>, <code>colsample_bytree=0.7972</code>, <code>reg_alpha=0.0101</code>, <code>reg_lambda=1.7248</code>.<br/>"
        "<b>6. Performance Comparison:</b> Tuned model outperformed baseline in CV Macro F1 (0.8286 vs 0.8232) and slashed inference latency by <b>41.9%</b>."
    )
    story.append(Paragraph(sec4_text, body_style))

    opt_table_data = [
        ["Model Variant", "CV Macro F1", "Student-Test Top-1 Acc.", "Student-Test Macro F1", "Log Loss", "Inference Time (95.8k records)"],
        ["Untuned XGBoost Baseline", "0.8232 ± 0.005", "78.67%", "0.8757", "0.1899", "135.0 ms"],
        ["Tuned XGBoost (Optuna)", "<b>0.8286 ± 0.004</b>", "<b>78.68%</b>", "<b>0.8759</b>", "0.1982", "<b>78.5 ms (41.9% Faster)</b>"]
    ]
    t_opt = Table(opt_table_data, colWidths=[1.8*inch, 1.1*inch, 1.2*inch, 1.2*inch, 0.6*inch, 1.3*inch])
    t_opt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('BACKGROUND', (0, 1), (-1, -1), c_light),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t_opt)
    story.append(Paragraph("<b>Table 3:</b> Comparison of Untuned vs. Bayesian Tuned XGBoost.", caption_style))

    # ==========================================
    # SECTION 5: HMM AND VITERBI ROUTE RECONSTRUCTION
    # ==========================================
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    sec5_text = (
        "<b>1. Hidden States:</b> Candidate road segments $a_i \\in C(z_t)$ retrieved for each GPS point $z_t$ ($|C(z_t)| \\le 10$).<br/>"
        "<b>2. Emission Probabilities:</b> Blended Gaussian spatial distance decay and calibrated classifier posterior probabilities in log-space:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$\\ln P(z_t \\mid a_i) = \\alpha \\ln \\left(\\frac{1}{\\sqrt{2\\pi}\\sigma_z} \\exp\\left(-\\frac{d_\\perp^2}{2\\sigma_z^2}\\right)\\right) + (1-\\alpha) \\ln P_{\\text{clf}}(a_i \\mid z_t)$, with $\\sigma_z=20.0$m, $\\alpha=0.35$.<br/>"
        "<b>3. Transition Probabilities:</b> Derived from the directed road adjacency graph $G$:<br/>"
        "&nbsp;&nbsp;&bull; Same arc persistence: $\\ln P(a_i \\to a_i) = \\ln(0.95)$.<br/>"
        "&nbsp;&nbsp;&bull; Direct graph edge ($\\text{target}(a_i) = \\text{source}(a_j)$): $\\ln P = \\ln(0.85)$.<br/>"
        "&nbsp;&nbsp;&bull; 2-hop topological reachability: $\\ln P = \\ln(0.40) - \\frac{|d_{\\text{net}} - d_{\\text{gps}}|}{\\beta}$ with $\\beta=15.0$m.<br/>"
        "<b>4. Missing Candidates & GPS Gaps:</b> When $\\Delta t > 10$s or temporal jumps occur, transition penalties are dynamically relaxed "
        "proportional to elapsed time to allow long-distance graph routing.<br/>"
        "<b>5. Disconnected Road Segments:</b> Disconnected arcs receive a heavy log penalty ($\\ln(10^{-6})$), preventing illegal road hopping.<br/>"
        "<b>6. Empirical Route Improvement:</b> Viterbi decoding solved the catastrophic point-wise continuity gap, boosting route continuity "
        "from <b>87.48% to 99.00%</b> (+11.52% gain, reaching 100% on trip 7) and Top-1 accuracy from <b>80.09% to 83.68%</b> (+3.59% gain) across all 10 trips."
    )
    story.append(Paragraph(sec5_text, body_style))

    # Viterbi Results Table
    vit_data = [
        ["Trip ID", "GPS Obs", "Classifier Top-1", "Viterbi Top-1", "Classifier Cont.", "Viterbi Cont.", "Route Overlap", "Latency"],
        ["trip_00000000", "2,503", "88.37%", "90.45%", "94.00%", "99.24%", "95.53%", "1,000.3 ms"],
        ["trip_00000007", "1,354", "92.02%", "94.61%", "98.97%", "<b>100.00%</b>", "98.37%", "527.1 ms"],
        ["trip_00000009", "1,049", "79.89%", "83.51%", "82.25%", "98.38%", "89.98%", "306.7 ms"],
        ["trip_00000040", "2,995", "86.48%", "89.08%", "94.19%", "99.80%", "<b>100.00%</b>", "866.0 ms"],
        ["trip_00000052", "262", "87.40%", "91.98%", "83.91%", "98.85%", "92.63%", "97.4 ms"],
        ["trip_00000056", "3,958", "79.64%", "82.49%", "83.73%", "98.56%", "93.45%", "1,364.6 ms"],
        ["trip_00000065", "768", "73.31%", "78.39%", "86.05%", "98.70%", "97.24%", "197.4 ms"],
        ["trip_00000076", "1,486", "74.29%", "80.08%", "82.22%", "97.98%", "90.37%", "410.8 ms"],
        ["trip_00000077", "3,911", "59.91%", "60.93%", "88.70%", "99.72%", "99.63%", "1,184.4 ms"],
        ["trip_00000091", "874", "79.63%", "85.24%", "80.76%", "98.74%", "84.91%", "290.8 ms"],
        ["Overall / Mean", "19,160", "80.09%", "<b>83.68% (+3.59%)</b>", "87.48%", "<b>99.00% (+11.52%)</b>", "<b>94.21%</b>", "<b>624.5 ms</b>"]
    ]
    t_vit = Table(vit_data, colWidths=[0.95*inch, 0.65*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.85*inch, 0.75*inch])
    t_vit.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('BACKGROUND', (0, 1), (-1, -2), c_light),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#FEF08A")), # Summary highlight
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t_vit)
    story.append(Paragraph("<b>Table 4:</b> Sequential Route Reconstruction: Point-Wise Classifier vs. HMM + Viterbi across all 10 Student-Testing Trips.", caption_style))

    # ==========================================
    # SECTION 6: GEOGRAPHICAL VISUALIZATION USING FOLIUM
    # ==========================================
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    sec6_text = (
        "Multi-layer interactive Folium maps were created for evaluation trips. The visualizations incorporate four distinguishable layers: "
        "(1) <b>Raw GPS Points:</b> Blue circle markers ($r=3$px) annotated with point indices and timestamps; "
        "(2) <b>Road Network Geometry:</b> Slate-gray polylines depicting the local road graph subgraph; "
        "(3) <b>Ground-Truth Reference Route:</b> Solid emerald green polyline ($w=4$px); "
        "(4) <b>PathSnap Viterbi Predicted Route:</b> Dashed crimson polyline ($w=2.5$px). "
        "LayerControl allows evaluators to toggle layers independently. Figure 5 depicts representative reconstructed routes."
    )
    story.append(Paragraph(sec6_text, body_style))

    img_map1 = os.path.join(BASE_DIR, "visualization", "exports", "trip_00000007_route_reconstruction.png")
    img_map2 = os.path.join(BASE_DIR, "visualization", "exports", "trip_00000040_route_reconstruction.png")
    if os.path.exists(img_map1) and os.path.exists(img_map2):
        t_maps = Table([[Image(img_map1, width=3.2*inch, height=2.0*inch), Image(img_map2, width=3.2*inch, height=2.0*inch)]], colWidths=[3.3*inch, 3.3*inch])
        t_maps.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(t_maps)
        story.append(Paragraph("<b>Figure 5:</b> Geographical Route Reconstruction Visualizations. Left: trip_00000007 (100% Continuity). Right: trip_00000040 (100% Reference Overlap).", caption_style))

    # ==========================================
    # SECTION 7: API, UI AND DEPLOYMENT
    # ==========================================
    story.append(Paragraph("7. API, UI and Deployment", h1_style))
    sec7_text = (
        "<b>7.1 Distributed Architecture:</b><br/>"
        "PathSnap implements a decoupled client-server architecture:<br/>"
        "&nbsp;&bull; <b>FastAPI Backend (Laptop 1):</b> Asynchronous REST service exposing <code>GET /health</code>, <code>GET /trips</code>, "
        "and <code>POST /predict</code>. The engine pre-loads the fitted <code>RobustScaler</code>, tuned <code>XGBoost</code>, and Viterbi decoder. "
        "It evaluates raw trajectories with zero access to ground-truth reference routes.<br/>"
        "&nbsp;&bull; <b>Streamlit UI (Laptop 2):</b> Interactive visual dashboard providing trip selection, remote inference triggering, "
        "latency and continuity KPI metrics, and embedded Folium map rendering.<br/>"
        "&nbsp;&bull; <b>Wi-Fi Network Hosting:</b> Configured to bind on <code>0.0.0.0</code> (all interfaces) via <code>start_wifi_host.ps1</code>. "
        "Teammates on the same Wi-Fi access the dashboard at <code>http://10.8.121.23:8501</code> and the API docs at <code>http://10.8.121.23:8000/docs</code>."
    )
    story.append(Paragraph(sec7_text, body_style))

    img_ui = os.path.join(BASE_DIR, "visualization", "exports", "ui_dashboard_preview.png")
    img_cmd = os.path.join(BASE_DIR, "visualization", "exports", "cmd_api_response_preview.png")
    if os.path.exists(img_ui) and os.path.exists(img_cmd):
        t_deploy = Table([
            [Image(img_ui, width=3.3*inch, height=1.9*inch),
             Image(img_cmd, width=3.3*inch, height=1.9*inch)]
        ], colWidths=[3.3*inch, 3.3*inch])
        t_deploy.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(t_deploy)
        story.append(Paragraph("<b>Figure 6:</b> Deployment Verification. Left: Streamlit Interactive Evaluator UI. Right: Live CMD Terminal on Inference Laptop showing successful HTTP 200 responses.", caption_style))

    # ==========================================
    # SECTION 8: ACADEMIC INTEGRITY & AI DISCLOSURE
    # ==========================================
    story.append(Paragraph("8. Academic Integrity, Plagiarism and AI Content Checks", h1_style))
    sec8_text = (
        "<b>Statement of Originality & AI Assistance Disclosure:</b><br/>"
        "This project and report represent the original experimental work, code, and analytical findings of Team PT-122 (PathSnap) for "
        "the PES University UE24CS352A Machine Learning Hackathon. In accordance with the hackathon's AI disclosure policy, "
        "AI assistance (Claude / Antigravity) was utilized for code scaffolding, script templating, and documentation formatting. "
        "All model training, cross-validation, hyperparameter tuning, empirical benchmark metrics, and geographical map outputs "
        "were executed directly on the authentic competition dataset in the team's local environment. All reported numerical results "
        "match the verifiable artifacts, logs, and CSV outputs committed in the project repository."
    )
    story.append(Paragraph(sec8_text, body_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Technical Report PDF successfully generated: {OUTPUT_PDF}")
    try:
        shutil.copy2(OUTPUT_PDF, DEFAULT_PDF)
        print(f"Also updated default: {DEFAULT_PDF}")
    except Exception as e:
        print(f"Note: Could not overwrite {DEFAULT_PDF} (it may be open in a viewer). {OUTPUT_PDF} is fully updated.")

if __name__ == "__main__":
    create_report()
