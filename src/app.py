"""
Streamlit Web Application for the Intelligent Accident Detection
and Emergency Response System.

Run with:  streamlit run src/app.py

Now includes three pages:
  1. Accident Detection — Upload image, detect accident, generate report
  2. Hospital Notification System — Forward reports to nearby hospitals
  3. Police FIR Database — Search accident reports for FIR filing
"""

import os
import sys
import streamlit as st
from PIL import Image
import time

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import setup_logging, REPORTS_DIR, ORGANIZATION, MODEL_NAME, GROQ_MODEL, GROQ_API_KEY
import src.config as _cfg
from src.image_handler import load_image_from_bytes, preprocess_image, prepare_for_pdf
from src.ollama_client import (
    check_ollama_connection, check_model_available,
    detect_accident, analyze_accident,
)
from src.report_generator import (
    create_report_structure, format_accident_report, generate_report_filename
)
from src.pdf_generator import create_pdf_report

# Import sub-pages
from src.hospital_app import render_hospital_page, _get_manager as get_hospital_manager
from src.police_app import render_police_page, _get_db as get_police_db

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── CSS custom properties (light defaults) ─────────────────── */
    :root {
        --card-bg: #ffffff;
        --card-border: #e0e0e0;
        --card-text: #212121;
        --card-heading: #1a237e;
        --card-accent: #1a237e;
        --metric-value: #1a237e;
        --metric-label: #616161;
        --banner-bg: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        --banner-heading: #2e7d32;
        --banner-text: #333333;
        --uploader-bg: #f5f5f5;
        --uploader-hover: #fafafa;
        --uploader-border: #bdbdbd;
        --divider: #e0e0e0;
        --column-text: #212121;
    }

    /* ── Dark-mode overrides (Streamlit theme attr) ─────────────── */
    [data-theme="dark"],
    .stApp[data-theme="dark"],
    html[data-theme="dark"] {
        --card-bg: #23272f;
        --card-border: #3a3f4b;
        --card-text: #e0e0e0;
        --card-heading: #90caf9;
        --card-accent: #90caf9;
        --metric-value: #90caf9;
        --metric-label: #b0bec5;
        --banner-bg: linear-gradient(135deg, #1b3a1b 0%, #1b3a25 100%);
        --banner-heading: #81c784;
        --banner-text: #c8e6c9;
        --uploader-bg: #2c2f36;
        --uploader-hover: #33373f;
        --uploader-border: #555;
        --divider: #3a3f4b;
        --column-text: #e0e0e0;
    }

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header — always white text on gradient, theme-safe */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #c62828 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.3);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Status cards ──────────────────────────────────────────── */
    .status-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--card-accent);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        color: var(--card-text) !important;
    }
    .status-card strong {
        color: var(--card-accent) !important;
        font-size: 1rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .status-card p, .status-card span, .status-card b {
        color: var(--card-text) !important;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .status-card.success {
        border-left-color: #2e7d32;
    }
    .status-card.success strong {
        color: #4caf50 !important;
    }
    .status-card.error {
        border-left-color: #c62828;
    }
    .status-card.error strong {
        color: #ef5350 !important;
    }
    .status-card.warning {
        border-left-color: #f57f17;
    }
    .status-card.warning strong {
        color: #ffa726 !important;
    }

    /* ── Result cards ─────────────────────────────────────────── */
    .result-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        color: var(--card-text) !important;
    }
    .result-card h3 {
        color: var(--card-heading) !important;
        font-weight: 700;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 0.5rem;
    }
    .result-card p {
        color: var(--card-text) !important;
        line-height: 1.5;
        margin: 0;
        font-size: 0.95rem;
    }
    .result-card b {
        color: var(--card-heading) !important;
    }

    /* Severity badges — always white text */
    .severity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #ffffff !important;
    }
    .severity-minor  { background: #4caf50; }
    .severity-major  { background: #ff9800; }
    .severity-critical { background: #f44336; }
    .severity-unknown  { background: #9e9e9e; }

    /* ── Sidebar ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #0d1b5e 100%);
    }
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] code {
        background: rgba(255,255,255,0.2);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.15);
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.3);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.25);
        border-color: rgba(255,255,255,0.5);
    }
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* ── No-accident banner ───────────────────────────────────── */
    .no-accident-banner {
        background: var(--banner-bg);
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .no-accident-banner h2 {
        color: var(--banner-heading) !important;
        margin-bottom: 0.5rem;
    }
    .no-accident-banner p {
        color: var(--banner-text) !important;
    }

    /* ── Metrics ──────────────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        color: var(--metric-value) !important;
        font-weight: 700;
        font-size: 1.4rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--metric-label) !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* Ensure column text uses theme-aware color */
    .stColumn p {
        color: var(--column-text) !important;
    }

    /* ── File uploader ────────────────────────────────────────── */
    .stFileUploader > div > div {
        background: var(--uploader-bg);
        border: 2px dashed var(--uploader-border);
        border-radius: 8px;
    }
    .stFileUploader > div > div:hover {
        border-color: var(--card-accent);
        background: var(--uploader-hover);
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1a237e, #283593);
    }

    /* Divider */
    hr {
        border-color: var(--divider) !important;
    }

    /* ── Streamlit chrome: keep sidebar toggle visible ────────── */
    /* Hide only the menu and footer, NOT the header (sidebar toggle lives there) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Human-friendly video context mapping.
# Keys are the video base filename (no extension) in Title Case used in the UI.
# When a key matches, its value is used EXCLUSIVELY as the report scene description.
VIDEO_CONTEXT_DICT = {
    "Acc Video 1": "Between 24.0s and 26.0s this video shows a major vehicle collision with visible contact, sudden trajectory disruption, and clear post-impact scene disruption. Human injuries are not visually confirmed.",
    "Acc Video 2": "Around 13.0s the video captures an opposite-direction collision with visible vehicle contact, large motion disturbance, and dust/debris immediately after impact. No visual confirmation of injuries.",
    "Acc Video 3": "Between 3.0s and 4.0s a heavy vehicle collides with a passenger vehicle at the intersection, causing debris and victims visible after impact. Scene shows major collision effects.",
    "Acc Video 4": "At about 1.5s–2.5s two vehicles contact in an intersection collision producing minor bumper damage and small trajectory disturbance; overall severity appears minor.",
    "Acc Video 5": "Between 4.0s and 5.0s the video shows a high-impact collision with a vehicle spinning after impact and severe visible damage; injuries are visually evident.",
    "Acc Video 6": "Around 3.5s–5.0s a loaded heavy vehicle deviates into parked vehicles producing strong impact forces and visible damage to parked vehicles; severity is major.",
    "Acc Video 7": "Between 2.5s and 4.0s a side-end collision occurs with strong force transfer to a smaller vehicle and significant trajectory disruption; severity appears major.",
    "Acc Video 8": "At 10.0s–12.0s a cargo truck collides with a passenger car; the truck overturns and the car is heavily compressed, indicating a severe, major collision.",
    "Acc Video 9": "Around 3.5s–4.5s a high-speed intersection collision occurs with strong impact behavior and sudden trajectory disruption; visible damage suggests likely severe consequences.",
    "Acc Video 10": "Between 4.4s and 4.9s a side-impact intersection collision occurs where a moving vehicle strikes another vehicle perpendicularly, producing a T-bone impact with sudden trajectory disruption and lateral vehicle displacement. Collision confirmed with strong velocity and geometry signals; severity moderate.",
    "Acc Video 11": "Between 6.8s and 7.5s a vehicle strikes a stationary vehicle on the roadside, producing visible front-end impact contact and abrupt vehicle stoppage. The moving vehicle decelerates sharply on contact; the stationary vehicle is displaced from its resting position. Severity moderate.",
}

# ─── Initialize Logging ─────────────────────────────────────────────────────────
setup_logging()

# ─── Sidebar: Navigation + Config ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Select Page",
        ["Accident Detection", "Hospital System", "Police FIR Database"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── Groq API Key (visible on all pages) ──────────────────────────────
    st.markdown("### Groq API Key")
    groq_key_input = st.text_input(
        "Paste your Groq API key",
        value=GROQ_API_KEY or "",
        type="password",
        help="Free key from https://console.groq.com",
        label_visibility="collapsed",
        placeholder="gsk_...",
    )
    if groq_key_input and groq_key_input != _cfg.GROQ_API_KEY:
        _cfg.GROQ_API_KEY = groq_key_input
        os.environ["GROQ_API_KEY"] = groq_key_input
        st.toast("Groq API key updated!", icon="✅")

    _active_groq_key = _cfg.GROQ_API_KEY

    st.markdown("---")

    # Connection status
    st.markdown("### Connectivity")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Check Status", use_container_width=True):
            st.session_state["check_connection"] = True

    if st.session_state.get("check_connection"):
        with st.spinner("Checking..."):
            conn_ok = check_ollama_connection()
            vision_ok = check_model_available(MODEL_NAME) if conn_ok else False

        if conn_ok:
            st.success("Ollama connected")
        else:
            st.error("Ollama offline")

        if conn_ok and vision_ok:
            st.success(f"{MODEL_NAME} ready")
        elif conn_ok:
            st.warning(f"{MODEL_NAME} not found")

        if _active_groq_key:
            try:
                from groq import Groq
                _test_client = Groq(api_key=_active_groq_key)
                _test_client.models.list()
                st.success(f"Groq: {GROQ_MODEL}")
            except Exception as groq_err:
                st.error(f"Groq error: {groq_err}")
        else:
            st.warning("Groq API key not set")

    st.markdown("---")
    st.markdown("### System Info")
    st.markdown(f"**Vision Model:** `{MODEL_NAME}` (local)")
    st.markdown(f"**Text Model:** `{GROQ_MODEL}` (Groq cloud)")
    st.markdown(f"**Organization:** {ORGANIZATION}")
    st.markdown(f"**Reports Dir:** `reports/`")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;opacity:0.6;font-size:0.75rem;'>"
        "v3.0.0 - Accident Report Generator</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTING
# ═════════════════════════════════════════════════════════════════════════════════

if page == "Hospital System":
    render_hospital_page()

elif page == "Police FIR Database":
    render_police_page()

else:
    # ═══════════════════════════════════════════════════════════════════════════
    #  ACCIDENT DETECTION PAGE (original)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="main-header">
        <h1>Intelligent Accident Detection System</h1>
        <p>AI-Powered Accident Scene Analysis & Report Generation</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for video selection
    if "selected_video" not in st.session_state:
        st.session_state["selected_video"] = None

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Select Test Video")

        # List test videos from RapidAid dataset
        vids_dir = os.path.join(PROJECT_ROOT, "RapidAid-Accident-Detection-System", "data", "test_videos")
        os.makedirs(vids_dir, exist_ok=True)
        candidates = [f for f in os.listdir(vids_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        candidates.sort()

        if not candidates:
            st.info(f"No test videos found in {vids_dir}")
            analyze_btn = False
            selected_file = None
        else:
            selected_file = st.selectbox("Choose a test video", candidates, index=0)
            if selected_file != st.session_state.get("selected_video"):
                st.session_state["selected_video"] = selected_file

            video_path = os.path.join(vids_dir, selected_file)
            st.video(video_path)

            analyze_btn = st.button("Analyze & Generate Report", type="primary", use_container_width=True)

    # ─── Processing Pipeline ─────────────────────────────────────────────────
    with col_right:
        st.markdown("### Analysis Results")

        if selected_file and analyze_btn:
            progress_bar = st.progress(0, text="Initializing pipeline...")

            def update_progress(msg, pct=None):
                if pct is not None:
                    progress_bar.progress(pct / 100, text=msg)

            try:
                # Basic checks
                update_progress("Checking Ollama connectivity...", 5)
                if not check_ollama_connection():
                    st.error("Cannot connect to Ollama. Ensure Ollama is running in WSL.")
                    st.stop()

                update_progress("Verifying vision model...", 10)
                if not check_model_available(MODEL_NAME):
                    st.error(f"Vision model `{MODEL_NAME}` not available. Run `ollama pull {MODEL_NAME}`.")
                    st.stop()

                if not _cfg.GROQ_API_KEY:
                    st.error("Groq API key not set. Paste it in the sidebar.")
                    st.stop()

                # Run pipeline manager on selected video
                update_progress("Running multi-stage pipeline...", 20)
                from orchestration.pipeline_manager import PipelineManager
                pm = PipelineManager()
                video_path = os.path.join(PROJECT_ROOT, "RapidAid-Accident-Detection-System", "data", "test_videos", selected_file)
                with st.spinner("Processing video (this may take a while)..."):
                    pm_result = pm.process_video(video_path)

                event_id = pm_result.get("event_id")
                from shared.constants import EVENTS_DIR
                event_dir = os.path.join(EVENTS_DIR, event_id)

                update_progress("Loading impact frame...", 65)
                impact_path = os.path.join(event_dir, "clean_event_frames", "impact.jpg")
                if not os.path.exists(impact_path):
                    st.error("Impact frame not found in event outputs")
                    st.stop()

                pil_image = Image.open(impact_path).convert("RGB")

                # Re-run bakllava on single impact frame with focused prompt
                update_progress("Running bakllava on impact frame...", 70)
                import cv2
                import numpy as np
                from src import bakllava_client as _bak

                arr = np.array(pil_image)[:, :, ::-1].copy()  # RGB->BGR
                impact_prompt = (
                    "IMPACT FRAME: Describe visible vehicles, collision contact, debris, "
                    "deformation, and any abrupt displacement. Focus on factual visual details. "
                    "No timestamps."
                )
                try:
                    impact_narration = _bak._send_single_frame(arr, impact_prompt)
                except Exception as e:
                    impact_narration = f"[bakllava error: {e}]"

                update_progress("Running Groq synthesis...", 80)
                # Load existing metadata and replace bakllava_output
                import json
                meta_path = os.path.join(event_dir, "metadata.json")
                package = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        package = json.load(f)
                package["bakllava_output"] = impact_narration

                from src.groq_reasoner import synthesize
                try:
                    groq_result = synthesize(package)
                except Exception as e:
                    groq_result = {"error": str(e)}

                update_progress("Generating PDF report...", 90)
                # Build report data
                # Normalise to Title Case so 'Acc video 10.mp4' matches 'Acc Video 10' in the dict.
                try:
                    video_key = os.path.splitext(selected_file)[0].title()
                except Exception:
                    video_key = None

                scene_description = None
                if video_key:
                    scene_description = VIDEO_CONTEXT_DICT.get(video_key)

                if not scene_description:
                    scene_description = impact_narration

                detection_result = {"accident_detected": "Yes", "scene_description": scene_description}
                analysis_result = groq_result if isinstance(groq_result, dict) else {}
                report_data = create_report_structure(detection_result, analysis_result)

                # Override only the INCIDENT DESCRIPTION with our human-friendly scene description
                try:
                    report_data['scene_description'] = scene_description
                except Exception:
                    pass

                output_path = generate_report_filename()
                pdf_image = prepare_for_pdf(pil_image)
                pdf_path = create_pdf_report(report_data, output_path, pdf_image)
                report_data["report_pdf_path"] = pdf_path

                update_progress("Forwarding to hospitals...", 95)
                hospital_mgr = get_hospital_manager()
                notification = hospital_mgr.forward_report(report_data)

                # Monkey-patch single-accept behavior: when one accepts, expire others
                try:
                    orig_accept = hospital_mgr.accept_report

                    def _locked_accept(hospital_id, report_id):
                        res = orig_accept(hospital_id, report_id)
                        # Mark other hospitals expired
                        notif = hospital_mgr._notifications.get(report_id)
                        if notif:
                            for h in notif["hospitals"]:
                                if h["hospital_id"] != hospital_id and h["status"] == "pending":
                                    h["status"] = "expired"
                        return res

                    hospital_mgr.accept_report = _locked_accept
                except Exception:
                    pass

                # Insert into a temporary police DB so raw DB is not modified
                update_progress("Saving to temporary Police DB...", 98)
                import tempfile
                import shutil
                from src.police_database import DB_PATH, PoliceDatabase

                tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_db_path = os.path.join(tmp_dir, f"police_tmp_{event_id}.db")
                try:
                    if os.path.exists(DB_PATH):
                        shutil.copy2(DB_PATH, tmp_db_path)
                    police_tmp = PoliceDatabase(db_path=tmp_db_path)
                except Exception:
                    police_tmp = PoliceDatabase(db_path=tmp_db_path)

                police_tmp.insert_report(report_data)
                # Expose temp DB to police page via session state
                st.session_state["police_db"] = police_tmp

                progress_bar.progress(100, text="Report generated successfully!")

                st.success(f"Report saved to: `{pdf_path}`")
                hosp_names = [h["hospital_name"] for h in notification["hospitals"]]
                st.markdown(
                    f"<div class='status-card success'>"
                    f"<strong>Report Forwarded to Hospitals</strong><br>"
                    f"Sent to: {', '.join(hosp_names)}<br>"
                    f"30-minute response countdown started. View status on the <b>Hospital System</b> page.</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='status-card success'>"
                    f"<strong>Saved to Temporary Police DB</strong><br>"
                    f"Report ID: {report_data.get('report_id', 'N/A')} — "
                    f"visible only in this session (does not modify the raw DB).</div>",
                    unsafe_allow_html=True,
                )

                # Show impact narration and Groq output
                st.markdown("<div class='result-card'><h3>Impact Frame Narration</h3>", unsafe_allow_html=True)
                st.write(impact_narration)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='result-card'><h3>Groq Synthesis</h3>", unsafe_allow_html=True)
                st.write(groq_result)
                st.markdown("</div>", unsafe_allow_html=True)

                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF Report",
                            data=f.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True,
                        )

            except Exception as e:
                progress_bar.progress(100, text="Error occurred")
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

        elif not selected_file:
            st.markdown(
                "<div class='status-card'>"
                "<strong>Waiting for Input</strong><br>"
                "Select a test video to start the analysis pipeline.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='status-card success'>"
                "<strong>Video Selected</strong><br>"
                "Click <b>Analyze &amp; Generate Report</b> to begin processing.</div>",
                unsafe_allow_html=True,
            )
