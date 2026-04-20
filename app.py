"""
╔═══════════════════════════════════════════════════════════════════╗
║   INDUSTRIAL PREDICTIVE MAINTENANCE & OPERATIONS INTELLIGENCE     ║
║   Production-Grade AI Platform | SLB / Siemens Energy Grade       ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os
import warnings
import threading

warnings.filterwarnings("ignore")

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IntelliOps AI — Industrial Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "IntelliOps Industrial AI v2.0 | Production Grade Predictive Maintenance",
    },
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* === GLOBAL === */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Rajdhani:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary:   #050B18;
        --bg-secondary: #0A1628;
        --bg-card:      #0D1B35;
        --bg-hover:     #112040;
        --accent-cyan:  #00D4FF;
        --accent-green: #00FF9D;
        --accent-amber: #FFB800;
        --accent-red:   #FF1744;
        --accent-purple:#B000FF;
        --text-primary: #E8F4FD;
        --text-muted:   #6B8CAE;
        --border:       #1A3A5C;
        --glow-cyan:    0 0 20px rgba(0,212,255,0.3);
        --glow-green:   0 0 20px rgba(0,255,157,0.3);
        --glow-amber:   0 0 20px rgba(255,184,0,0.3);
        --glow-red:     0 0 25px rgba(255,23,68,0.4);
        --sidebar-width: 280px;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .stApp {
        background: linear-gradient(135deg, #050B18 0%, #071020 40%, #050B18 100%);
    }

    /* === RESPONSIVE LAYOUT === */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .kpi-card .kpi-value {
            font-size: 1.8rem !important;
        }
        .sensor-grid {
            grid-template-columns: 1fr 1fr !important;
        }
        .intelliops-header {
            flex-wrap: wrap;
            padding: 12px 16px !important;
        }
        .header-title {
            font-size: 1.2rem !important;
        }
        .header-status {
            width: 100%;
            flex-direction: row !important;
            justify-content: flex-start !important;
            gap: 10px !important;
        }
    }

    @media (max-width: 480px) {
        .kpi-card .kpi-value {
            font-size: 1.4rem !important;
        }
        .kpi-label {
            font-size: 0.6rem !important;
        }
        .machine-card {
            padding: 10px !important;
        }
        .header-logo {
            font-size: 1.8rem !important;
        }
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060E1F 0%, #0A1628 100%) !important;
        border-right: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* Sidebar toggle button styling */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #0A1628, #112040) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0 8px 8px 0 !important;
        color: var(--accent-cyan) !important;
        transition: all 0.2s ease !important;
        box-shadow: 2px 0 10px rgba(0,212,255,0.1) !important;
    }
    [data-testid="collapsedControl"]:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: var(--accent-cyan) !important;
        stroke: var(--accent-cyan) !important;
    }

    /* Make sidebar collapse arrow visible */
    button[kind="header"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 50% !important;
        color: var(--accent-cyan) !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    button[kind="header"]:hover {
        background: var(--bg-hover) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
    }

    /* Sidebar collapse button arrow icon */
    [data-testid="stSidebarCollapseButton"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 50% !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: var(--accent-cyan) !important;
    }

    /* Sidebar expand button when collapsed */
    [data-testid="stSidebarCollapsedControl"] {
        background: linear-gradient(135deg, #060E1F, #0A1628) !important;
        border: 1px solid var(--border) !important;
        border-left: none !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 8px !important;
        box-shadow: 2px 0 15px rgba(0,212,255,0.15) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
        background: var(--bg-hover) !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: var(--accent-cyan) !important;
        stroke: var(--accent-cyan) !important;
    }

    /* === HEADER BAR === */
    .intelliops-header {
        background: linear-gradient(90deg, #060E1F, #0A1628, #060E1F);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        position: relative;
        overflow: hidden;
        flex-wrap: wrap;
    }
    .intelliops-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-green), transparent);
    }
    .header-logo { font-size: 2.8rem; flex-shrink: 0; }
    .header-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: clamp(1.2rem, 3vw, 1.9rem);
        font-weight: 700;
        color: var(--accent-cyan);
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
    }
    .header-sub {
        font-size: clamp(0.6rem, 1.5vw, 0.8rem);
        color: var(--text-muted);
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .header-status {
        margin-left: auto;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 4px;
    }
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        box-shadow: var(--glow-green);
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50% { opacity:0.5; transform:scale(1.3); }
    }

    /* === KPI CARDS === */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: clamp(12px, 2vw, 20px);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,212,255,0.1);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .kpi-card.cyan::before { background: linear-gradient(90deg, var(--accent-cyan), transparent); }
    .kpi-card.green::before { background: linear-gradient(90deg, var(--accent-green), transparent); }
    .kpi-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), transparent); }
    .kpi-card.red::before { background: linear-gradient(90deg, var(--accent-red), transparent); }
    .kpi-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: clamp(1.6rem, 3vw, 2.8rem);
        font-weight: 700;
        line-height: 1;
        margin: 8px 0 4px;
    }
    .kpi-label {
        font-size: clamp(0.6rem, 1.2vw, 0.75rem);
        color: var(--text-muted);
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .kpi-icon { font-size: clamp(1rem, 2vw, 1.4rem); }
    .kpi-sub {
        font-size: 0.7rem;
        margin-top: 6px;
        color: var(--text-muted);
    }

    /* === MACHINE CARDS === */
    .machine-card {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid var(--border);
        transition: all 0.2s ease;
        position: relative;
    }
    .machine-card:hover { border-color: var(--accent-cyan); transform: translateY(-2px); }
    .machine-card.critical { border-left: 3px solid var(--accent-red); }
    .machine-card.warning { border-left: 3px solid var(--accent-amber); }
    .machine-card.degrading { border-left: 3px solid #FF7043; }
    .machine-card.healthy { border-left: 3px solid var(--accent-green); }

    .machine-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .machine-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--accent-cyan);
    }
    .machine-type { font-size: 0.72rem; color: var(--text-muted); }

    .sensor-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 10px; }
    .sensor-item { text-align: center; }
    .sensor-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; }
    .sensor-name { font-size: 0.62rem; color: var(--text-muted); letter-spacing: 1px; }

    /* === ALERT CARDS === */
    .alert-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid;
        position: relative;
    }
    .alert-critical { background: rgba(255,23,68,0.08); border-color: var(--accent-red); }
    .alert-warning { background: rgba(255,184,0,0.08); border-color: var(--accent-amber); }
    .alert-info { background: rgba(33,150,243,0.08); border-color: #2196F3; }

    .alert-machine {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .alert-message { font-size: 0.82rem; color: var(--text-muted); }
    .alert-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .badge-critical { background: rgba(255,23,68,0.25); color: var(--accent-red); }
    .badge-warning { background: rgba(255,184,0,0.25); color: var(--accent-amber); }
    .badge-info { background: rgba(33,150,243,0.25); color: #2196F3; }

    /* === SECTION HEADERS === */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
        flex-wrap: wrap;
    }
    .section-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: clamp(1rem, 2vw, 1.2rem);
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-primary);
    }

    /* === HEALTH BAR === */
    .health-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 6px;
        margin-top: 6px;
        overflow: hidden;
    }
    .health-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

    /* === PROGRESS === */
    .stProgress > div > div { background-color: var(--accent-cyan) !important; }

    /* === METRICS === */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] { color: var(--accent-cyan) !important; font-family: 'Rajdhani', sans-serif !important; }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border) !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        white-space: nowrap !important;
        font-size: clamp(0.7rem, 1.5vw, 0.9rem) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
    }

    /* === BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, #0A1628, #112040) !important;
        color: var(--accent-cyan) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
    }

    /* === SELECTBOX === */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }

    /* === INSIGHT BOX === */
    .insight-box {
        background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(0,255,157,0.04));
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: clamp(0.75rem, 1.5vw, 0.9rem);
        line-height: 1.6;
        font-style: italic;
        position: relative;
    }
    .insight-box::before {
        content: '"';
        position: absolute;
        top: -5px; left: 12px;
        font-size: 3rem;
        color: var(--accent-cyan);
        opacity: 0.3;
        font-family: serif;
    }

    /* === DATA TABLE === */
    .stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow-x: auto !important; }

    /* === HIDE STREAMLIT DEFAULTS === */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }

    /* === LIVE BADGE === */
    .live-badge {
        background: rgba(0,255,157,0.15);
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .live-dot {
        display: inline-block;
        width: 6px; height: 6px;
        background: var(--accent-green);
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 1.5s infinite;
    }

    /* === SCROLLBAR === */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

    /* === RESPONSIVE COLUMNS === */
    @media (max-width: 600px) {
        [data-testid="column"] {
            min-width: 100% !important;
        }
    }

    /* === SIDEBAR TOGGLE BUTTON ENHANCED === */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }

    /* Sidebar arrow/chevron button */
    .st-emotion-cache-1cypcdb,
    [data-testid="stSidebarNav"] + button,
    button[data-testid="baseButton-headerNoPadding"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--accent-cyan) !important;
        border-radius: 50% !important;
        color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
    }

    /* Force sidebar toggle arrow color */
    [data-testid="stSidebar"] button svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }

    /* Sidebar collapsed control (arrow when sidebar is hidden) */
    div[data-testid="collapsedControl"] {
        top: 50% !important;
        background: linear-gradient(135deg, #060E1F, #0A1628) !important;
        border: 1px solid var(--accent-cyan) !important;
        border-left: none !important;
        border-radius: 0 8px 8px 0 !important;
        box-shadow: 4px 0 20px rgba(0,212,255,0.2) !important;
        padding: 12px 6px !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="collapsedControl"]:hover {
        background: var(--bg-hover) !important;
        box-shadow: var(--glow-cyan) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Imports ───────────────────────────────────────────────────────────────────

from data_generator import SensorDataGenerator, generate_alerts
from ml_pipeline import IndustrialMLPipeline, train_or_load_pipeline
from utils import (
    compute_fleet_kpis, get_state_color, get_state_icon,
    format_rul, format_sensor_value, sensor_display_name,
    health_to_color, get_recommendations, failure_prob_to_severity,
    STATE_COLORS,
)


# ─── Session State Initialization ─────────────────────────────────────────────

def init_session_state():
    if "generator" not in st.session_state:
        st.session_state.generator = SensorDataGenerator(n_machines=12)
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "training_done" not in st.session_state:
        st.session_state.training_done = False
    if "live_readings" not in st.session_state:
        st.session_state.live_readings = None
    if "historical_df" not in st.session_state:
        st.session_state.historical_df = None
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    if "selected_machine" not in st.session_state:
        st.session_state.selected_machine = None
    if "train_metrics" not in st.session_state:
        st.session_state.train_metrics = {}


# ─── Plotly Theme ──────────────────────────────────────────────────────────────

PLOT_BG = "#060E1F"
GRID_COLOR = "#1A3A5C"
TEXT_COLOR = "#6B8CAE"
PLOT_FONT = "Space Grotesk"

# Base layout WITHOUT yaxis/xaxis to avoid conflicts when overriding
plotly_layout = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family=PLOT_FONT, color="#E8F4FD", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
    margin=dict(l=40, r=20, t=40, b=30),
    hovermode="x unified",
)

# Separate axis defaults to apply via update_xaxes / update_yaxes
axis_defaults = dict(
    gridcolor=GRID_COLOR,
    zerolinecolor=GRID_COLOR,
    tickfont=dict(color=TEXT_COLOR),
)

def apply_axis_defaults(fig):
    """Apply consistent axis styling without conflicting with range overrides."""
    fig.update_xaxes(**axis_defaults)
    fig.update_yaxes(**axis_defaults)
    return fig


# ─── Header ────────────────────────────────────────────────────────────────────

def render_header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="intelliops-header">
        <div class="header-logo">🧠</div>
        <div>
            <div class="header-title">IntelliOps AI</div>
            <div class="header-sub">Industrial Predictive Maintenance & Operations Intelligence Platform</div>
        </div>
        <div class="header-status">
            <div><span class="status-dot"></span><span style="font-size:0.8rem;color:#6B8CAE;">SYSTEM ONLINE</span></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4A6A8A;">{now} UTC</div>
            <div class="live-badge"><span class="live-dot"></span>LIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(gen: SensorDataGenerator, pipeline: IndustrialMLPipeline):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px;">
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.4rem;color:#00D4FF;font-weight:700;letter-spacing:3px;">⚙️ CONTROL</div>
            <div style="font-size:0.7rem;color:#4A6A8A;letter-spacing:2px;margin-top:2px;">PANEL</div>
        </div>
        <hr style="border-color:#1A3A5C;margin:10px 0;">
        """, unsafe_allow_html=True)

        # System Status
        st.markdown("### 📡 System Status")
        if st.session_state.training_done:
            st.success("✅ AI Models Active")
        else:
            st.warning("⚠️ Models Not Trained")

        if pipeline and pipeline.train_metrics:
            m = pipeline.train_metrics
            st.metric("Failure AUC", f"{m.get('failure_auc', 0):.3f}")
            st.metric("RUL MAE", f"{m.get('rul_mae_hours', 0):.1f}h")
            st.metric("Training Samples", f"{m.get('total_samples', 0):,}")

        st.markdown("---")

        # Controls
        st.markdown("### 🔧 Settings")
        n_machines = st.slider("Fleet Size", 4, 12, 12, step=2)
        if n_machines != gen.n_machines:
            st.session_state.generator = SensorDataGenerator(n_machines=n_machines)
            st.rerun()

        refresh_rate = st.selectbox("Refresh Rate", ["5s", "10s", "30s", "60s", "Manual"], index=1)
        st.session_state.auto_refresh = refresh_rate != "Manual"

        st.markdown("---")

        # Training
        st.markdown("### 🤖 AI Training")
        history_hours = st.slider("Training Data (hours)", 240, 1440, 720, step=120)

        col1, col2 = st.columns(2)
        with col1:
            train_btn = st.button("🚀 Train AI", use_container_width=True)
        with col2:
            force_btn = st.button("🔄 Retrain", use_container_width=True)

        if train_btn or force_btn:
            if force_btn:
                import glob
                for f in glob.glob("models/*.pkl"):
                    os.remove(f)

            with st.spinner("Training AI models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("📊 Generating historical data...")
                progress_bar.progress(15)
                hist_df = gen.generate_historical_data(n_hours=history_hours, interval_minutes=30)
                st.session_state.historical_df = hist_df

                status_text.text("🔍 Training Anomaly Detector...")
                progress_bar.progress(35)

                pipeline_new = IndustrialMLPipeline()
                status_text.text("⚠️  Training Failure Predictor...")
                progress_bar.progress(55)

                status_text.text("⏱️  Training RUL Estimator...")
                progress_bar.progress(70)

                metrics = pipeline_new.train(hist_df, verbose=False)
                progress_bar.progress(90)

                st.session_state.pipeline = pipeline_new
                st.session_state.training_done = True
                st.session_state.train_metrics = metrics
                progress_bar.progress(100)
                status_text.text("✅ Training Complete!")
                time.sleep(0.5)
                st.rerun()

        st.markdown("---")

        # About
        st.markdown("""
        <div style="font-size:0.7rem;color:#4A6A8A;text-align:center;line-height:1.8;">
            <div style="margin-bottom:6px;color:#6B8CAE;">IntelliOps AI v2.0</div>
            Modeled after SLB / Siemens Energy / El Sewedy Industrial Systems<br><br>
            🔒 100% Local Processing<br>
            🤖 3-Model AI Pipeline<br>
            📊 Real-time SCADA Simulation
        </div>
        """, unsafe_allow_html=True)

    return refresh_rate


# ─── KPI Row ───────────────────────────────────────────────────────────────────

def render_kpi_row(kpis: dict):
    cols = st.columns(6)
    kpi_data = [
        ("TOTAL MACHINES", kpis.get("total_machines", 0), "🏭", "cyan", None),
        ("HEALTHY", kpis.get("healthy_count", 0), "✅", "green", f"{kpis.get('availability_pct', 0)}% availability"),
        ("DEGRADING", kpis.get("degrading_count", 0), "📉", "amber", "Early degradation"),
        ("CRITICAL", kpis.get("critical_count", 0), "🚨", "red", "Immediate attention"),
        ("SYSTEM HEALTH", f"{kpis.get('overall_health', 0):.1f}%", "💚", "cyan", "Fleet average"),
        ("MIN RUL", format_rul(kpis.get("min_rul_hours", 0)), "⏱️", "amber", "Critical machine"),
    ]
    for col, (label, val, icon, color, sub) in zip(cols, kpi_data):
        with col:
            sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value" style="color:{'#00D4FF' if color=='cyan' else '#00FF9D' if color=='green' else '#FFB800' if color=='amber' else '#FF1744'};">
                    {val}
                </div>
                <div class="kpi-label">{label}</div>
                {sub_html}
            </div>
            """, unsafe_allow_html=True)


# ─── Fleet Overview Chart ──────────────────────────────────────────────────────

def render_fleet_overview(readings_df: pd.DataFrame):
    col1, col2, col3 = st.columns([2, 1.5, 1])

    # ── Health Score Bar Chart
    with col1:
        df_sorted = readings_df.sort_values("health_score")
        colors = [get_state_color(s) for s in df_sorted["state"]]

        fig = go.Figure(go.Bar(
            x=df_sorted["machine_id"],
            y=df_sorted["health_score"],
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in df_sorted["health_score"]],
            textposition="outside",
            textfont=dict(size=9, color="#6B8CAE"),
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="#00FF9D", line_width=1,
                      annotation_text="Healthy Threshold", annotation_font_size=9)
        fig.add_hline(y=50, line_dash="dash", line_color="#FF7043", line_width=1,
                      annotation_text="Warning Threshold", annotation_font_size=9)
        # FIX: Apply layout first, then update axes separately to avoid conflict
        fig.update_layout(**plotly_layout, title="Fleet Health Scores", height=280)
        fig.update_yaxes(**axis_defaults, range=[0, 115])
        fig.update_xaxes(**axis_defaults)
        st.plotly_chart(fig, use_container_width=True)

    # ── State Distribution Donut
    with col2:
        state_counts = readings_df["state"].value_counts()
        colors_donut = [get_state_color(s) for s in state_counts.index]

        fig2 = go.Figure(go.Pie(
            labels=state_counts.index,
            values=state_counts.values,
            hole=0.65,
            marker_colors=colors_donut,
            textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>%{value} machines<br>%{percent}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b>{readings_df['health_score'].mean():.0f}%</b><br>Health",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#00D4FF", family="Rajdhani"),
        )
        fig2.update_layout(**plotly_layout, title="Fleet Status Distribution",
                           showlegend=True, height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Failure Probability Gauge
    with col3:
        fail_col = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
        avg_fail = readings_df[fail_col].mean() if fail_col in readings_df.columns else 25

        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_fail,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Avg Failure Risk", "font": {"size": 13, "color": "#6B8CAE"}},
            number={"suffix": "%", "font": {"size": 28, "color": "#FFB800", "family": "Rajdhani"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 8, "color": "#4A6A8A"}},
                "bar": {"color": "#FFB800", "thickness": 0.25},
                "bgcolor": "#0A1628",
                "bordercolor": "#1A3A5C",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,255,157,0.12)"},
                    {"range": [30, 60], "color": "rgba(255,184,0,0.12)"},
                    {"range": [60, 100], "color": "rgba(255,23,68,0.12)"},
                ],
                "threshold": {"line": {"color": "#FF1744", "width": 2}, "value": 70},
            },
        ))
        fig3.update_layout(**plotly_layout, height=280, margin=dict(l=30, r=30, t=50, b=10))
        st.plotly_chart(fig3, use_container_width=True)


# ─── Live Sensor Charts ────────────────────────────────────────────────────────

def render_sensor_charts(readings_df: pd.DataFrame, selected_machine: str):
    row = readings_df[readings_df["machine_id"] == selected_machine]
    if row.empty:
        st.warning(f"No data for {selected_machine}")
        return
    row = row.iloc[0]

    sensors = ["temperature", "vibration", "pressure", "energy_consumption", "load_factor"]
    sensor_labels = ["Temperature", "Vibration", "Pressure", "Energy", "Load"]

    sensor_values = []
    for s in sensors:
        if s in readings_df.columns:
            fleet_max = readings_df[s].max()
            val = row.get(s, 0) / (fleet_max + 0.001)
            sensor_values.append(round(val, 3))
        else:
            sensor_values.append(0)

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_radar = go.Figure(go.Scatterpolar(
            r=sensor_values + [sensor_values[0]],
            theta=sensor_labels + [sensor_labels[0]],
            fill="toself",
            fillcolor=f"rgba(0,212,255,0.15)",
            line=dict(color="#00D4FF", width=2),
            marker=dict(size=6, color="#00D4FF"),
        ))
        fig_radar.update_layout(
            **plotly_layout,
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR, tickfont=dict(size=8, color=TEXT_COLOR)),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10, color="#E8F4FD")),
            ),
            title=f"Sensor Profile: {selected_machine}",
            height=300,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        gauge_cols = st.columns(3)
        gauge_data = [
            ("temperature", "°C", "#FF6B6B", 0, 500),
            ("vibration", "mm/s", "#FFB800", 0, 15),
            ("pressure", "bar", "#00D4FF", 0, 300),
            ("energy_consumption", "kW", "#00FF9D", 0, 1200),
            ("load_factor", "", "#B000FF", 0, 1),
            ("rpm", "RPM", "#FF7043", 0, 4000),
        ]
        for idx, (sensor, unit, color, min_v, max_v) in enumerate(gauge_data):
            val = row.get(sensor, 0)
            with gauge_cols[idx % 3]:
                sensor_name = sensor_display_name(sensor)
                short_name = sensor_name.split(" ", 1)[1] if " " in sensor_name else sensor
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={"suffix": f" {unit}", "font": {"size": 14, "color": color, "family": "JetBrains Mono"}},
                    title={"text": short_name, "font": {"size": 9, "color": TEXT_COLOR}},
                    gauge={
                        "axis": {"range": [min_v, max_v], "tickfont": {"size": 7}},
                        "bar": {"color": color, "thickness": 0.3},
                        "bgcolor": "#0A1628",
                        "bordercolor": GRID_COLOR,
                    },
                ))
                fig_g.update_layout(**plotly_layout, height=150, margin=dict(l=15, r=15, t=30, b=5))
                st.plotly_chart(fig_g, use_container_width=True)


# ─── Machine Cards Grid ────────────────────────────────────────────────────────

def render_machine_cards(readings_df: pd.DataFrame):
    cols_per_row = 3
    machines = readings_df.to_dict("records")

    for row_start in range(0, len(machines), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, machine in zip(cols, machines[row_start:row_start + cols_per_row]):
            with col:
                state = machine.get("state", "unknown")
                health = machine.get("health_score", 0)
                fail_prob = machine.get("ml_failure_prob", machine.get("failure_probability", 0))
                rul = machine.get("ml_rul_hours", machine.get("rul_hours", 0))
                health_color = health_to_color(health)
                state_icon = get_state_icon(state)

                bar_width = health
                bar_color = health_color

                st.markdown(f"""
                <div class="machine-card {state}">
                    <div class="machine-header">
                        <div>
                            <div class="machine-id">{machine['machine_id']}</div>
                            <div class="machine-type">{machine.get('machine_type', 'Unknown')}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.4rem;">{state_icon}</div>
                            <div style="font-size:0.65rem;color:{health_color};text-transform:uppercase;letter-spacing:1px;">{state}</div>
                        </div>
                    </div>

                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="font-size:0.72rem;color:#4A6A8A;">HEALTH</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:{health_color};font-weight:700;">{health:.0f}%</span>
                    </div>
                    <div class="health-bar-bg">
                        <div class="health-bar-fill" style="width:{bar_width}%;background:{bar_color};"></div>
                    </div>

                    <div class="sensor-grid" style="margin-top:12px;">
                        <div class="sensor-item">
                            <div class="sensor-value" style="color:#FF6B6B;">{machine.get('temperature', 0):.1f}°C</div>
                            <div class="sensor-name">TEMP</div>
                        </div>
                        <div class="sensor-item">
                            <div class="sensor-value" style="color:#FFB800;">{machine.get('vibration', 0):.2f}</div>
                            <div class="sensor-name">VIB mm/s</div>
                        </div>
                        <div class="sensor-item">
                            <div class="sensor-value" style="color:#00D4FF;">{machine.get('pressure', 0):.1f}</div>
                            <div class="sensor-name">PRESS bar</div>
                        </div>
                    </div>

                    <div style="margin-top:12px;padding-top:10px;border-top:1px solid #1A3A5C;display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-size:0.65rem;color:#4A6A8A;">FAIL PROB</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:{'#FF1744' if fail_prob > 70 else '#FFB800' if fail_prob > 40 else '#00FF9D'};font-weight:700;">{fail_prob:.0f}%</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.65rem;color:#4A6A8A;">RUL</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#00D4FF;font-weight:700;">{format_rul(rul)}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─── Alerts Panel ──────────────────────────────────────────────────────────────

def render_alerts(alerts: list):
    if not alerts:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#4A6A8A;">
            <div style="font-size:3rem;margin-bottom:10px;">✅</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;letter-spacing:2px;">ALL SYSTEMS NOMINAL</div>
            <div style="font-size:0.8rem;margin-top:6px;">No active alerts at this time</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for alert in alerts[:10]:
        level = alert.get("level", "info")
        mid = alert.get("machine_id", "UNKNOWN")
        mtype = alert.get("machine_type", "")
        messages = alert.get("messages", [])
        rul = alert.get("rul_hours", 0)
        fail_prob = alert.get("failure_prob", 0)

        badge_class = f"badge-{level}"
        card_class = f"alert-{level}"
        level_icon = "🔴" if level == "critical" else "🟡" if level == "warning" else "🔵"

        msg_html = "<br>".join([f"• {m}" for m in messages])

        st.markdown(f"""
        <div class="alert-card {card_class}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
                <div style="flex:1;min-width:200px;">
                    <div class="alert-badge {badge_class}">{level_icon} {level.upper()}</div>
                    <div class="alert-machine">{mid} <span style="color:#4A6A8A;font-size:0.75rem;font-weight:normal;">— {mtype}</span></div>
                    <div class="alert-message">{msg_html}</div>
                </div>
                <div style="text-align:right;min-width:100px;">
                    <div style="font-size:0.65rem;color:#4A6A8A;margin-bottom:4px;">FAILURE PROB</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:#FF1744;font-weight:700;">{fail_prob:.0f}%</div>
                    <div style="font-size:0.65rem;color:#4A6A8A;margin-top:6px;">RUL</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#FFB800;font-weight:700;">{format_rul(rul)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── AI Insights Panel ────────────────────────────────────────────────────────

def render_ai_insights(readings_df: pd.DataFrame, pipeline: IndustrialMLPipeline):
    if pipeline is None or not pipeline.is_trained:
        st.info("🤖 Train the AI models first to see predictive insights.")
        return

    fail_col = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
    rul_col = "ml_rul_hours" if "ml_rul_hours" in readings_df.columns else "rul_hours"

    at_risk = readings_df.nlargest(6, fail_col)

    st.markdown("""
    <div class="section-header">
        <div class="section-title">🤖 AI Predictive Insights</div>
    </div>
    """, unsafe_allow_html=True)

    for _, row in at_risk.iterrows():
        insight_text = pipeline.generate_insight_text(row)
        fail_p = row.get(fail_col, 0)
        rul_v = row.get(rul_col, 0)
        state = row.get("state", "unknown")
        health = row.get("health_score", 0)

        color = "#FF1744" if fail_p > 70 else "#FFB800" if fail_p > 40 else "#00FF9D"

        c1, c2, c3, c4 = st.columns([2.5, 1, 1, 1.5])
        with c1:
            st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)
        with c2:
            st.metric("Fail Prob", f"{fail_p:.0f}%")
        with c3:
            st.metric("RUL", format_rul(rul_v))
        with c4:
            recs = get_recommendations(state)
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #1A3A5C;border-radius:8px;padding:12px;font-size:0.75rem;">
                <div style="color:#6B8CAE;font-size:0.65rem;letter-spacing:1px;margin-bottom:6px;">RECOMMENDED ACTION</div>
                {recs[0] if recs else '—'}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Model Feature Importance")
    if pipeline.failure_predictor.feature_importances:
        feat_items = sorted(pipeline.failure_predictor.feature_importances.items(),
                            key=lambda x: x[1], reverse=True)[:10]
        feat_names = [f[0].replace("_rolling_mean_6", " (avg)").replace("_delta", " (Δ)").replace("_", " ").title()
                      for f, _ in feat_items]
        feat_vals = [v for _, v in feat_items]
        colors_fi = ["#00D4FF" if i == 0 else "#00FF9D" if i < 3 else "#FFB800" if i < 6 else "#4A6A8A"
                     for i in range(len(feat_vals))]

        fig = go.Figure(go.Bar(
            x=feat_vals,
            y=feat_names,
            orientation="h",
            marker_color=colors_fi,
            marker_line_width=0,
        ))
        fig.update_layout(**plotly_layout, title="Top Predictive Features (Failure Model)", height=350)
        fig.update_yaxes(**axis_defaults, autorange="reversed")
        fig.update_xaxes(**axis_defaults)
        st.plotly_chart(fig, use_container_width=True)


# ─── Historical Analytics ──────────────────────────────────────────────────────

def render_historical_analytics(hist_df: pd.DataFrame, selected_machine: str):
    if hist_df is None or hist_df.empty:
        st.info("📊 Generate historical data by training the AI models first.")
        return

    df_machine = hist_df[hist_df["machine_id"] == selected_machine].copy()
    if df_machine.empty:
        st.info(f"No historical data for {selected_machine}")
        return

    df_machine["timestamp"] = pd.to_datetime(df_machine["timestamp"])
    df_machine = df_machine.sort_values("timestamp")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=["Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)",
                        "Energy Consumption (kW)", "Degradation Level", "RUL (hours)"],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    sensor_pairs = [
        ("temperature", 1, 1, "#FF6B6B"),
        ("vibration", 1, 2, "#FFB800"),
        ("pressure", 2, 1, "#00D4FF"),
        ("energy_consumption", 2, 2, "#00FF9D"),
        ("degradation_level", 3, 1, "#B000FF"),
        ("rul_hours", 3, 2, "#FF7043"),
    ]

    for sensor, row_n, col_n, color in sensor_pairs:
        if sensor not in df_machine.columns:
            continue
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(
            go.Scatter(
                x=df_machine["timestamp"],
                y=df_machine[sensor],
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b},0.08)",
                name=sensor.replace("_", " ").title(),
                hovertemplate=f"{sensor}: %{{y:.2f}}<extra></extra>",
            ),
            row=row_n, col=col_n,
        )

    fig.update_layout(
        **plotly_layout,
        height=650,
        title=f"Historical Sensor Analysis — {selected_machine}",
        showlegend=False,
    )
    # Apply axis styling to all subplots
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(**axis_defaults, row=i, col=j)
            fig.update_yaxes(**axis_defaults, row=i, col=j)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 Fleet Degradation Comparison")
    fig2 = go.Figure()
    machine_ids = hist_df["machine_id"].unique()[:8]
    palette = ["#00D4FF", "#00FF9D", "#FFB800", "#FF7043", "#FF1744",
               "#B000FF", "#2196F3", "#4CAF50"]

    for i, mid in enumerate(machine_ids):
        df_m = hist_df[hist_df["machine_id"] == mid].copy()
        df_m["timestamp"] = pd.to_datetime(df_m["timestamp"])
        df_m = df_m.sort_values("timestamp")
        if "degradation_level" in df_m.columns:
            fig2.add_trace(go.Scatter(
                x=df_m["timestamp"],
                y=df_m["degradation_level"] * 100,
                mode="lines",
                name=mid,
                line=dict(color=palette[i % len(palette)], width=1.5),
                opacity=0.85,
            ))

    fig2.update_layout(**plotly_layout, title="Degradation Over Time — All Machines", height=350)
    fig2.update_xaxes(**axis_defaults)
    fig2.update_yaxes(**axis_defaults, title_text="Degradation %")
    st.plotly_chart(fig2, use_container_width=True)


# ─── Model Performance ─────────────────────────────────────────────────────────

def render_model_performance(pipeline: IndustrialMLPipeline, hist_df: pd.DataFrame):
    if pipeline is None or not pipeline.is_trained:
        st.info("Train the AI models to see performance metrics.")
        return

    metrics = pipeline.train_metrics
    if not metrics:
        return

    st.markdown("### 📊 Model Performance Report")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Failure AUC-ROC", f"{metrics.get('failure_auc', 0):.4f}", help="Area Under ROC Curve (1.0 = perfect)")
    with col2:
        st.metric("RUL MAE", f"{metrics.get('rul_mae_hours', 0):.1f}h", help="Mean Absolute Error in hours")
    with col3:
        st.metric("RUL R²", f"{metrics.get('rul_r2', 0):.4f}", help="R-squared coefficient")
    with col4:
        st.metric("Anomaly Rate", f"{metrics.get('anomaly_rate_pct', 0):.1f}%", help="% flagged as anomalous")

    st.markdown("---")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Training Samples", f"{metrics.get('total_samples', 0):,}")
    with col6:
        st.metric("Feature Count", metrics.get("feature_count", 0))
    with col7:
        st.metric("Training Time", pipeline.training_time.split("T")[0] if pipeline.training_time else "—")

    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    IntelliOps AI Pipeline                           │
    ├─────────────────┬──────────────────────┬───────────────────────────┤
    │  ANOMALY ENGINE │  FAILURE PREDICTOR   │     RUL ESTIMATOR         │
    │                 │                      │                           │
    │ Isolation Forest│ Gradient Boosting    │  Random Forest Regressor  │
    │ n_est=150       │ n_est=200, lr=0.08   │  n_est=200, depth=12      │
    │ contamination=8%│ max_depth=5          │  min_samples_leaf=5       │
    │                 │ subsample=0.8        │                           │
    │ Output:         │ Output:              │  Output:                  │
    │ Anomaly Score   │ P(failure) 0→100%    │  RUL in Hours             │
    └─────────────────┴──────────────────────┴───────────────────────────┘
         ↑                     ↑                         ↑
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   Feature Engineering Layer                        │
    │  Raw Sensors → Rolling Stats → Deltas → Cross-Sensor Features      │
    │  temperature, vibration, pressure, energy, load, runtime, RPM     │
    │  + 6-point rolling mean/std + delta change + temp×vib product     │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """)


# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    gen: SensorDataGenerator = st.session_state.generator

    # Auto-load models on startup
    if st.session_state.pipeline is None:
        pipeline = IndustrialMLPipeline()
        if pipeline.models_exist():
            if pipeline.load_all():
                st.session_state.pipeline = pipeline
                st.session_state.training_done = True
                st.session_state.train_metrics = pipeline.train_metrics
        else:
            st.session_state.pipeline = pipeline

    pipeline: IndustrialMLPipeline = st.session_state.pipeline

    # Get live readings
    readings_df = gen.generate_all_readings()

    # Apply ML predictions if trained
    if pipeline and pipeline.is_trained:
        try:
            readings_df = pipeline.predict_fleet(readings_df)
        except Exception:
            pass

    st.session_state.live_readings = readings_df
    st.session_state.refresh_count += 1

    # Sidebar
    refresh_rate = render_sidebar(gen, pipeline)

    # Header
    render_header()

    # KPIs
    kpis = compute_fleet_kpis(readings_df)
    render_kpi_row(kpis)

    # ── Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏭 Fleet Overview",
        "📡 Live Monitoring",
        "🚨 Alert Center",
        "🤖 AI Insights",
        "📊 Historical Analytics",
        "⚙️ Model Performance",
    ])

    # Machine selector (shared)
    machine_ids = list(readings_df["machine_id"].unique())
    selected_machine = st.sidebar.selectbox(
        "📍 Selected Machine",
        machine_ids,
        index=0,
        key="machine_selector",
    )

    # ── Tab 1: Fleet Overview
    with tab1:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🏭 Fleet Overview</div>
        </div>
        """, unsafe_allow_html=True)
        render_fleet_overview(readings_df)

        st.markdown("""
        <div class="section-header">
            <div class="section-title">🖥️ Machine Status Grid</div>
        </div>
        """, unsafe_allow_html=True)
        render_machine_cards(readings_df)

    # ── Tab 2: Live Monitoring
    with tab2:
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title">📡 Live Sensor Monitoring</div>
            <div class="live-badge"><span class="live-dot"></span>STREAMING</div>
            <div style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4A6A8A;">
                Refresh #{st.session_state.refresh_count} · {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_sensor_charts(readings_df, selected_machine)

        st.markdown("### 📋 Current Sensor Readings — All Machines")
        display_cols = ["machine_id", "machine_type", "state", "temperature", "vibration",
                        "pressure", "energy_consumption", "load_factor", "health_score",
                        "runtime_hours"]
        if "ml_failure_prob" in readings_df.columns:
            display_cols += ["ml_failure_prob", "ml_rul_hours"]

        # Only keep columns that actually exist
        display_cols = [c for c in display_cols if c in readings_df.columns]
        display_df = readings_df[display_cols].copy()
        display_df.columns = [c.replace("_", " ").title().replace("Ml ", "AI ") for c in display_cols]

        # Only apply gradient if the column exists after rename
        health_col_renamed = "Health Score"
        if health_col_renamed in display_df.columns:
            styled = display_df.style.background_gradient(subset=[health_col_renamed], cmap="RdYlGn").format(precision=2)
        else:
            styled = display_df.style.format(precision=2)

        st.dataframe(styled, use_container_width=True, height=300)

    # ── Tab 3: Alert Center
    with tab3:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🚨 Alert Center</div>
        </div>
        """, unsafe_allow_html=True)

        alerts = generate_alerts(readings_df)
        n_critical = sum(1 for a in alerts if a["level"] == "critical")
        n_warning = sum(1 for a in alerts if a["level"] == "warning")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="kpi-card red">
                <div class="kpi-icon">🔴</div>
                <div class="kpi-value" style="color:#FF1744;">{n_critical}</div>
                <div class="kpi-label">Critical Alerts</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card amber">
                <div class="kpi-icon">🟡</div>
                <div class="kpi-value" style="color:#FFB800;">{n_warning}</div>
                <div class="kpi-label">Warnings</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            total_at_risk = kpis.get("at_risk_count", 0)
            st.markdown(f"""
            <div class="kpi-card cyan">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-value" style="color:#00D4FF;">{total_at_risk}</div>
                <div class="kpi-label">Machines at Risk</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        render_alerts(alerts)

    # ── Tab 4: AI Insights
    with tab4:
        render_ai_insights(readings_df, pipeline)

    # ── Tab 5: Historical Analytics
    with tab5:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">📊 Historical Analytics</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.historical_df is None and pipeline and pipeline.is_trained:
            with st.spinner("Loading historical data..."):
                st.session_state.historical_df = gen.generate_historical_data(n_hours=720, interval_minutes=30)

        render_historical_analytics(st.session_state.historical_df, selected_machine)

    # ── Tab 6: Model Performance
    with tab6:
        render_model_performance(pipeline, st.session_state.historical_df)

    # ── Auto Refresh
    if st.session_state.auto_refresh:
        rates = {"5s": 5, "10s": 10, "30s": 30, "60s": 60}
        interval = rates.get(refresh_rate, 10)
        time.sleep(interval)
        st.rerun()


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
