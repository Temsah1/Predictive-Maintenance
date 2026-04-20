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

warnings.filterwarnings("ignore")

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IntelliOps AI — Industrial Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "IntelliOps Industrial AI v2.0 | Production Grade Predictive Maintenance",
    },
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
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
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #050B18 0%, #071020 40%, #050B18 100%);
    }

    /* === HIDE SIDEBAR COMPLETELY === */
    [data-testid="stSidebar"]              { display: none !important; }
    [data-testid="collapsedControl"]       { display: none !important; }
    [data-testid="stSidebarCollapseButton"]{ display: none !important; }
    section[data-testid="stSidebar"]       { display: none !important; }

    /* === HIDE STREAMLIT CHROME === */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    .block-container {
        padding-top: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* ════════════════════════════════
       TOP NAVBAR
    ════════════════════════════════ */
    .top-navbar {
        position: sticky;
        top: 0;
        z-index: 9999;
        background: linear-gradient(90deg, #04091A 0%, #060E1F 50%, #04091A 100%);
        border-bottom: 1px solid var(--border);
        padding: 0 24px;
        display: flex;
        align-items: center;
        gap: 0;
        height: 58px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.6);
        margin-bottom: 16px;
        overflow: hidden;
    }
    .top-navbar::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg,transparent,var(--accent-cyan),var(--accent-green),var(--accent-cyan),transparent);
        opacity: 0.5;
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-right: 28px;
        flex-shrink: 0;
    }
    .navbar-logo  { font-size: 1.6rem; }
    .navbar-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--accent-cyan);
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 0 0 15px rgba(0,212,255,0.4);
        white-space: nowrap;
    }
    .navbar-divider { width:1px; height:30px; background:var(--border); margin:0 18px; flex-shrink:0; }
    .navbar-stats {
        display: flex;
        align-items: center;
        gap: 18px;
        flex: 1;
        overflow-x: auto;
        scrollbar-width: none;
    }
    .navbar-stats::-webkit-scrollbar { display: none; }
    .nav-stat { text-align: center; flex-shrink: 0; }
    .nav-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .nav-stat-lbl {
        font-size: 0.58rem;
        color: var(--text-muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .nav-sep { width:1px; height:24px; background:rgba(26,58,92,0.6); flex-shrink:0; }
    .navbar-right {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-left: auto;
        flex-shrink: 0;
    }
    .navbar-time  { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#4A6A8A; }
    .navbar-live  {
        background: rgba(0,255,157,0.1);
        border: 1px solid rgba(0,255,157,0.35);
        color: var(--accent-green);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .live-dot {
        width:6px; height:6px;
        background:var(--accent-green);
        border-radius:50%;
        animation:pdot 1.5s infinite;
    }
    @keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(1.5)} }

    /* ════════════════════════════════
       KPI CARDS
    ════════════════════════════════ */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 12px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s, box-shadow 0.25s;
        height: 100%;
    }
    .kpi-card:hover { transform:translateY(-3px); box-shadow:0 8px 25px rgba(0,0,0,0.35); }
    .kpi-card::before {
        content:'';
        position:absolute;
        top:0; left:0; right:0;
        height:3px;
        border-radius:12px 12px 0 0;
    }
    .kpi-card.cyan::before   { background:linear-gradient(90deg,var(--accent-cyan),transparent); }
    .kpi-card.green::before  { background:linear-gradient(90deg,var(--accent-green),transparent); }
    .kpi-card.amber::before  { background:linear-gradient(90deg,var(--accent-amber),transparent); }
    .kpi-card.red::before    { background:linear-gradient(90deg,var(--accent-red),transparent); }
    .kpi-card.purple::before { background:linear-gradient(90deg,var(--accent-purple),transparent); }
    .kpi-icon  { font-size:1.2rem; margin-bottom:4px; }
    .kpi-value {
        font-family:'Rajdhani',sans-serif;
        font-size:clamp(1.4rem,2.2vw,2.4rem);
        font-weight:700; line-height:1.1; margin:4px 0;
    }
    .kpi-label { font-size:0.64rem; color:var(--text-muted); letter-spacing:1.5px; text-transform:uppercase; }
    .kpi-sub   { font-size:0.62rem; color:var(--text-muted); margin-top:4px; }

    /* ════════════════════════════════
       SECTION HEADERS
    ════════════════════════════════ */
    .section-header {
        display:flex; align-items:center; gap:12px;
        margin:22px 0 14px;
        padding-bottom:10px;
        border-bottom:1px solid var(--border);
        flex-wrap:wrap;
    }
    .section-title {
        font-family:'Rajdhani',sans-serif;
        font-size:1.1rem; font-weight:600;
        letter-spacing:2px; text-transform:uppercase;
        color:var(--text-primary);
    }
    .section-badge {
        background:rgba(0,212,255,0.1);
        border:1px solid rgba(0,212,255,0.28);
        color:var(--accent-cyan);
        padding:2px 8px; border-radius:20px;
        font-size:0.63rem; font-weight:700; letter-spacing:1px; text-transform:uppercase;
    }

    /* ════════════════════════════════
       MACHINE CARDS
    ════════════════════════════════ */
    .machine-card {
        background:var(--bg-card);
        border-radius:12px;
        padding:16px;
        border:1px solid var(--border);
        transition:transform 0.2s, box-shadow 0.2s;
        position:relative;
        overflow:hidden;
        margin-bottom:14px;
    }
    .machine-card:hover { transform:translateY(-3px); box-shadow:0 10px 28px rgba(0,0,0,0.3); }
    .machine-card.critical { border-left:4px solid var(--accent-red); }
    .machine-card.warning  { border-left:4px solid var(--accent-amber); }
    .machine-card.degrading{ border-left:4px solid #FF7043; }
    .machine-card.healthy  { border-left:4px solid var(--accent-green); }

    .mc-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }
    .mc-id     { font-family:'JetBrains Mono',monospace; font-size:0.92rem; font-weight:600; color:var(--accent-cyan); }
    .mc-type   { font-size:0.68rem; color:var(--text-muted); margin-top:2px; }
    .mc-icon   { font-size:1.5rem; }
    .mc-state  { font-size:0.58rem; text-transform:uppercase; letter-spacing:1px; text-align:right; margin-top:2px; }

    .hbar-row  { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
    .hbar-lbl  { font-size:0.65rem; color:#4A6A8A; }
    .hbar-val  { font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:700; }
    .hbar-bg   { background:rgba(255,255,255,0.05); border-radius:4px; height:5px; overflow:hidden; margin-bottom:10px; }
    .hbar-fill { height:100%; border-radius:4px; }

    .sgrid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-bottom:10px; }
    .sitem { text-align:center; padding:5px 3px; background:rgba(255,255,255,0.02); border-radius:6px; }
    .sval  { font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:600; }
    .slbl  { font-size:0.56rem; color:var(--text-muted); letter-spacing:0.5px; margin-top:2px; }

    .mc-footer { display:flex; justify-content:space-between; padding-top:8px; border-top:1px solid rgba(26,58,92,0.5); }
    .fstat-lbl { font-size:0.58rem; color:#4A6A8A; text-transform:uppercase; letter-spacing:1px; }
    .fstat-val { font-family:'JetBrains Mono',monospace; font-size:0.86rem; font-weight:700; margin-top:2px; }

    /* ════════════════════════════════
       ALERTS
    ════════════════════════════════ */
    .alert-card {
        border-radius:10px; padding:14px 16px; margin-bottom:10px;
        border-left:4px solid; transition:transform 0.2s;
    }
    .alert-card:hover { transform:translateX(4px); }
    .alert-critical { background:rgba(255,23,68,0.07);  border-color:var(--accent-red); }
    .alert-warning  { background:rgba(255,184,0,0.07); border-color:var(--accent-amber); }
    .alert-info     { background:rgba(33,150,243,0.07); border-color:#2196F3; }
    .alert-top  { display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; }
    .alert-left { flex:1; min-width:180px; }
    .abadge {
        display:inline-block; padding:2px 8px; border-radius:20px;
        font-size:0.63rem; font-weight:700; letter-spacing:1px;
        text-transform:uppercase; margin-bottom:5px;
    }
    .bc { background:rgba(255,23,68,0.2);  color:var(--accent-red); }
    .bw { background:rgba(255,184,0,0.2);  color:var(--accent-amber); }
    .bi { background:rgba(33,150,243,0.2); color:#2196F3; }
    .amachine { font-family:'JetBrains Mono',monospace; font-size:0.88rem; font-weight:700; margin-bottom:4px; }
    .amsg     { font-size:0.78rem; color:var(--text-muted); line-height:1.5; }
    .aright   { text-align:right; min-width:100px; }

    /* ════════════════════════════════
       INSIGHT BOX
    ════════════════════════════════ */
    .insight-box {
        background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(0,255,157,0.03));
        border:1px solid rgba(0,212,255,0.18);
        border-radius:10px;
        padding:13px 16px 13px 26px;
        margin:8px 0;
        font-size:0.85rem; line-height:1.6; font-style:italic;
        position:relative;
    }
    .insight-box::before {
        content:'"';
        position:absolute; top:-6px; left:9px;
        font-size:2.8rem; color:var(--accent-cyan); opacity:0.22; font-family:serif;
    }

    /* ════════════════════════════════
       TABS
    ════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background:var(--bg-secondary) !important;
        border-bottom:1px solid var(--border) !important;
        gap:2px !important;
        overflow-x:auto; scrollbar-width:none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display:none; }
    .stTabs [data-baseweb="tab"] {
        color:var(--text-muted) !important;
        font-family:'Rajdhani',sans-serif !important;
        font-weight:600 !important; letter-spacing:1px !important;
        font-size:0.82rem !important; padding:10px 14px !important;
        white-space:nowrap !important;
    }
    .stTabs [aria-selected="true"] {
        color:var(--accent-cyan) !important;
        border-bottom:2px solid var(--accent-cyan) !important;
        background:rgba(0,212,255,0.06) !important;
    }

    /* ════════════════════════════════
       BUTTONS, INPUTS, METRICS
    ════════════════════════════════ */
    .stButton > button {
        background:linear-gradient(135deg,#0A1628,#112040) !important;
        color:var(--accent-cyan) !important;
        border:1px solid var(--border) !important;
        border-radius:8px !important;
        font-family:'Rajdhani',sans-serif !important;
        font-weight:600 !important; letter-spacing:1px !important;
        transition:all 0.2s !important;
    }
    .stButton > button:hover {
        border-color:var(--accent-cyan) !important;
        box-shadow:var(--glow-cyan) !important;
    }
    .stSelectbox > div > div {
        background:var(--bg-card) !important;
        border:1px solid var(--border) !important;
        color:var(--text-primary) !important;
        border-radius:8px !important;
    }
    [data-testid="stMetric"] {
        background:var(--bg-card) !important;
        border:1px solid var(--border) !important;
        border-radius:10px !important;
        padding:14px !important;
    }
    [data-testid="stMetricValue"] {
        color:var(--accent-cyan) !important;
        font-family:'Rajdhani',sans-serif !important;
        font-size:1.5rem !important;
    }
    [data-testid="stMetricLabel"] { color:var(--text-muted) !important; font-size:0.73rem !important; }

    .stProgress > div > div { background-color:var(--accent-cyan) !important; }
    .stDataFrame { border:1px solid var(--border) !important; border-radius:8px !important; }

    /* ════════════════════════════════
       SCROLLBAR
    ════════════════════════════════ */
    ::-webkit-scrollbar { width:4px; height:4px; }
    ::-webkit-scrollbar-track { background:var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
    ::-webkit-scrollbar-thumb:hover { background:var(--accent-cyan); }

    /* ════════════════════════════════
       RESPONSIVE
    ════════════════════════════════ */
    @media (max-width:768px) {
        .top-navbar { padding:0 10px; height:50px; }
        .navbar-title { font-size:1rem; }
        .navbar-time  { display:none; }
        .block-container { padding-left:0.4rem !important; padding-right:0.4rem !important; }
    }
    @media (max-width:480px) {
        .navbar-divider { display:none; }
        .nav-sep        { display:none; }
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


# ─── Plotly Theme ──────────────────────────────────────────────────────────────

PLOT_BG    = "#060E1F"
GRID_COLOR = "#1A3A5C"
TEXT_COLOR = "#6B8CAE"
PLOT_FONT  = "Space Grotesk"

# Base layout — NEVER include xaxis/yaxis keys here to avoid dict-merge TypeError
plotly_layout = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family=PLOT_FONT, color="#E8F4FD", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, font=dict(size=10)),
    margin=dict(l=40, r=20, t=40, b=30),
    hovermode="x unified",
)

_AX = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR))

def _ax_apply(fig, **extra_y):
    """Apply consistent axis styling — separate from update_layout to avoid conflicts."""
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX, **extra_y)
    return fig


# ─── Session State ─────────────────────────────────────────────────────────────

def init_session_state():
    defaults = dict(
        generator=None, pipeline=None, training_done=False,
        live_readings=None, historical_df=None,
        refresh_count=0, auto_refresh=True,
        selected_machine=None, train_metrics={},
        n_machines=12, refresh_rate="10s", history_hours=720,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.generator is None:
        st.session_state.generator = SensorDataGenerator(n_machines=st.session_state.n_machines)


# ─── TOP NAVBAR ────────────────────────────────────────────────────────────────

def render_top_navbar(kpis: dict, n_alerts: int):
    now      = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%b %d, %Y")
    nc = kpis.get("critical_count", 0)
    crit_c = "#FF1744" if nc > 0 else "#00FF9D"
    alert_html = (f'<span style="background:rgba(255,23,68,.2);border:1px solid rgba(255,23,68,.4);'
                  f'color:#FF1744;padding:2px 8px;border-radius:10px;font-size:.65rem;font-weight:700;">'
                  f'🔴 {n_alerts} ALERTS</span>') if n_alerts > 0 else \
                 '<span style="font-size:.65rem;color:#00FF9D;font-weight:600;">✅ NOMINAL</span>'

    st.markdown(f"""
    <div class="top-navbar">
        <div class="navbar-brand">
            <span class="navbar-logo">🧠</span>
            <span class="navbar-title">IntelliOps AI</span>
        </div>
        <div class="navbar-divider"></div>
        <div class="navbar-stats">
            <div class="nav-stat">
                <div class="nav-stat-val" style="color:#00D4FF;">{kpis.get('total_machines',0)}</div>
                <div class="nav-stat-lbl">Fleet</div>
            </div>
            <div class="nav-sep"></div>
            <div class="nav-stat">
                <div class="nav-stat-val" style="color:#00FF9D;">{kpis.get('healthy_count',0)}</div>
                <div class="nav-stat-lbl">Healthy</div>
            </div>
            <div class="nav-sep"></div>
            <div class="nav-stat">
                <div class="nav-stat-val" style="color:#FFB800;">{kpis.get('at_risk_count',0)}</div>
                <div class="nav-stat-lbl">At Risk</div>
            </div>
            <div class="nav-sep"></div>
            <div class="nav-stat">
                <div class="nav-stat-val" style="color:{crit_c};">{nc}</div>
                <div class="nav-stat-lbl">Critical</div>
            </div>
            <div class="nav-sep"></div>
            <div class="nav-stat">
                <div class="nav-stat-val" style="color:#00D4FF;">{kpis.get('overall_health',0):.1f}%</div>
                <div class="nav-stat-lbl">Health</div>
            </div>
            <div class="nav-sep"></div>
            <div class="nav-stat">
                <div class="nav-stat-val" style="color:#FFB800;">{format_rul(kpis.get('min_rul_hours',0))}</div>
                <div class="nav-stat-lbl">Min RUL</div>
            </div>
        </div>
        <div class="navbar-right">
            <span class="navbar-time">📅 {date_str} &nbsp;⏱ {now}</span>
            {alert_html}
            <div class="navbar-live"><span class="live-dot"></span>LIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── CONTROL BAR ──────────────────────────────────────────────────────────────

def render_control_bar(gen: SensorDataGenerator, pipeline: IndustrialMLPipeline):
    with st.expander("⚙️  Control Panel — Fleet settings, refresh rate & AI training", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1.4, 1.1, 1.1, 1.5, 2.2])

        with c1:
            st.caption("Fleet Size")
            n_machines = st.slider("Fleet", 4, 16, st.session_state.n_machines,
                                   step=2, label_visibility="collapsed", key="fs")
            if n_machines != st.session_state.n_machines:
                st.session_state.n_machines = n_machines
                st.session_state.generator  = SensorDataGenerator(n_machines=n_machines)
                st.rerun()

        with c2:
            st.caption("Refresh Rate")
            rr = st.selectbox("RR", ["5s","10s","30s","60s","Manual"],
                              index=1, label_visibility="collapsed", key="rr")
            st.session_state.refresh_rate  = rr
            st.session_state.auto_refresh  = rr != "Manual"

        with c3:
            st.caption("Training Data (hrs)")
            hh = st.slider("HH", 240, 1440, st.session_state.history_hours,
                           step=120, label_visibility="collapsed", key="hh")
            st.session_state.history_hours = hh

        with c4:
            st.caption("Selected Machine")
            if st.session_state.live_readings is not None:
                mids = list(st.session_state.live_readings["machine_id"].unique())
                sel  = st.selectbox("M", mids, index=0,
                                    label_visibility="collapsed", key="msel")
                st.session_state.selected_machine = sel

        with c5:
            st.caption("AI Model Status")
            if st.session_state.training_done and pipeline and pipeline.train_metrics:
                m = pipeline.train_metrics
                st.markdown(f"""
                <div style="background:rgba(0,255,157,.07);border:1px solid rgba(0,255,157,.22);
                            border-radius:8px;padding:6px 10px;font-size:.71rem;">
                    <b style="color:#00FF9D;">✅ ACTIVE</b>&nbsp;|&nbsp;
                    AUC <b style="color:#00D4FF;">{m.get('failure_auc',0):.3f}</b>&nbsp;|&nbsp;
                    RUL MAE <b style="color:#FFB800;">{m.get('rul_mae_hours',0):.0f}h</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(255,184,0,.07);border:1px solid rgba(255,184,0,.22);
                            border-radius:8px;padding:6px 10px;font-size:.71rem;color:#FFB800;">
                    ⚠️ Models not trained — click Train
                </div>""", unsafe_allow_html=True)

        with c6:
            st.caption("Train AI")
            b1, b2 = st.columns(2)
            with b1:
                train_btn  = st.button("🚀 Train",   use_container_width=True, key="trn")
            with b2:
                force_btn  = st.button("🔄 Retrain", use_container_width=True, key="rtrn")
            if train_btn or force_btn:
                if force_btn:
                    import glob
                    for f in glob.glob("models/*.pkl"): os.remove(f)
                _run_training(gen, st.session_state.history_hours)

    return st.session_state.refresh_rate


def _run_training(gen, history_hours):
    with st.spinner("🤖 Training AI models…"):
        pb  = st.progress(0)
        stx = st.empty()
        stx.text("📊 Generating training data…");   pb.progress(10)
        hist_df = gen.generate_historical_data(n_hours=history_hours, interval_minutes=30)
        st.session_state.historical_df = hist_df;  pb.progress(30)
        stx.text("🔍 Training Anomaly Detector…");  pb.progress(45)
        p = IndustrialMLPipeline()
        stx.text("⚠️ Training Failure Predictor…"); pb.progress(60)
        stx.text("⏱️ Training RUL Estimator…");     pb.progress(75)
        metrics = p.train(hist_df, verbose=False);  pb.progress(92)
        st.session_state.pipeline      = p
        st.session_state.training_done = True
        st.session_state.train_metrics = metrics
        pb.progress(100); stx.text("✅ Done!"); time.sleep(0.5); st.rerun()


# ─── KPI ROW ───────────────────────────────────────────────────────────────────

def render_kpi_row(kpis: dict, readings_df: pd.DataFrame):
    fail_col = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
    avg_fail = readings_df[fail_col].mean() if fail_col in readings_df.columns else 0
    min_rul  = kpis.get("min_rul_hours", 0)

    kpi_data = [
        ("FLEET",     kpis.get("total_machines",0),          "🏭","cyan",  "Total machines"),
        ("HEALTHY",   kpis.get("healthy_count",0),           "✅","green", f"{kpis.get('availability_pct',0):.0f}% avail."),
        ("DEGRADING", kpis.get("degrading_count",0),         "📉","amber", "Early degradation"),
        ("CRITICAL",  kpis.get("critical_count",0),          "🚨","red",   "Needs action"),
        ("HEALTH",    f"{kpis.get('overall_health',0):.1f}%","💚","cyan",  "Fleet average"),
        ("FAIL RISK", f"{avg_fail:.1f}%",                    "⚡","amber", "Avg probability"),
        ("MIN RUL",   format_rul(min_rul),                   "⏱️","red" if min_rul<48 else "amber","Lowest useful life"),
        ("UPTIME",    f"{kpis.get('availability_pct',0):.1f}%","🔋","green","Fleet availability"),
    ]
    vc_map = {"cyan":"#00D4FF","green":"#00FF9D","amber":"#FFB800","red":"#FF1744","purple":"#B000FF"}
    cols = st.columns(len(kpi_data))
    for col, (lbl, val, icon, color, sub) in zip(cols, kpi_data):
        vc = vc_map[color]
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value" style="color:{vc};">{val}</div>
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


# ─── FLEET OVERVIEW ────────────────────────────────────────────────────────────

def render_fleet_overview(readings_df: pd.DataFrame):
    c1, c2, c3 = st.columns([2.2, 1.5, 1.3])

    with c1:
        df_s   = readings_df.sort_values("health_score")
        colors = [get_state_color(s) for s in df_s["state"]]
        fig = go.Figure(go.Bar(
            x=df_s["machine_id"], y=df_s["health_score"],
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.0f}%" for v in df_s["health_score"]],
            textposition="outside", textfont=dict(size=9, color="#6B8CAE"),
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="#00FF9D", line_width=1,
                      annotation_text="Healthy ▶", annotation_font_size=9, annotation_font_color="#00FF9D")
        fig.add_hline(y=50, line_dash="dash", line_color="#FF7043", line_width=1,
                      annotation_text="Warning ▶", annotation_font_size=9, annotation_font_color="#FF7043")
        fig.update_layout(**plotly_layout, title="Fleet Health Scores", height=300)
        # ✅ FIX: axes applied separately — no dict key conflict
        fig.update_xaxes(**_AX)
        fig.update_yaxes(**_AX, range=[0, 118])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sc     = readings_df["state"].value_counts()
        cdont  = [get_state_color(s) for s in sc.index]
        fig2   = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=0.68,
            marker_colors=cdont, textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>%{value} machines (%{percent})<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b>{readings_df['health_score'].mean():.0f}%</b><br>AVG",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#00D4FF", family="Rajdhani"),
        )
        # ✅ FIX: plain update_layout — no xaxis/yaxis keys
        fig2.update_layout(**plotly_layout, title="Status Distribution",
                           showlegend=True, height=300,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        fc   = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
        avgf = readings_df[fc].mean() if fc in readings_df.columns else 20
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avgf,
            delta={"reference": 30, "valueformat": ".1f"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Fleet Fail Risk", "font": {"size": 12, "color": "#6B8CAE"}},
            number={"suffix": "%", "font": {"size": 26, "color": "#FFB800", "family": "Rajdhani"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 8, "color": "#4A6A8A"}},
                "bar": {"color": "#FFB800", "thickness": 0.28},
                "bgcolor": "#0A1628", "bordercolor": "#1A3A5C",
                "steps": [
                    {"range": [0, 30],  "color": "rgba(0,255,157,0.1)"},
                    {"range": [30, 60], "color": "rgba(255,184,0,0.1)"},
                    {"range": [60, 100],"color": "rgba(255,23,68,0.1)"},
                ],
                "threshold": {"line": {"color": "#FF1744", "width": 2}, "value": 70},
            },
        ))
        fig3.update_layout(**plotly_layout, height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Secondary charts row
    st.markdown("---")
    c4, c5 = st.columns(2)
    fc  = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
    rc  = "ml_rul_hours"    if "ml_rul_hours"    in readings_df.columns else "rul_hours"

    with c4:
        fig4 = go.Figure()
        for state in readings_df["state"].unique():
            ds = readings_df[readings_df["state"] == state]
            fig4.add_trace(go.Scatter(
                x=ds["health_score"],
                y=ds[fc] if fc in ds.columns else [0]*len(ds),
                mode="markers+text",
                name=state,
                text=ds["machine_id"],
                textposition="top center",
                textfont=dict(size=8, color="#6B8CAE"),
                marker=dict(size=12, color=get_state_color(state),
                            line=dict(width=1, color="#1A3A5C")),
                hovertemplate="<b>%{text}</b><br>Health: %{x:.1f}%<br>Fail: %{y:.1f}%<extra></extra>",
            ))
        fig4.update_layout(**plotly_layout, title="Health vs Failure Probability", height=320)
        fig4.update_xaxes(**_AX, title_text="Health Score (%)")
        fig4.update_yaxes(**_AX, title_text="Failure Probability (%)")
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        if rc in readings_df.columns:
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(
                x=readings_df[rc], nbinsx=15,
                marker_color="#00D4FF", opacity=0.8,
                marker_line=dict(color="#1A3A5C", width=1),
            ))
            fig5.add_vline(x=168,  line_dash="dash", line_color="#FFB800", line_width=1.5,
                           annotation_text="1 Week",  annotation_font_color="#FFB800", annotation_font_size=9)
            fig5.add_vline(x=720,  line_dash="dash", line_color="#00FF9D", line_width=1.5,
                           annotation_text="30 Days", annotation_font_color="#00FF9D", annotation_font_size=9)
            fig5.update_layout(**plotly_layout, title="RUL Distribution (Hours)", height=320)
            fig5.update_xaxes(**_AX, title_text="Remaining Useful Life (h)")
            fig5.update_yaxes(**_AX, title_text="Count")
            st.plotly_chart(fig5, use_container_width=True)


# ─── MACHINE CARDS ─────────────────────────────────────────────────────────────

def render_machine_cards(readings_df: pd.DataFrame):
    machines   = readings_df.to_dict("records")
    per_row    = 4
    for start in range(0, len(machines), per_row):
        cols = st.columns(per_row)
        for col, m in zip(cols, machines[start:start+per_row]):
            with col:
                state  = m.get("state", "unknown")
                health = m.get("health_score", 0)
                fp     = m.get("ml_failure_prob", m.get("failure_probability", 0))
                rul    = m.get("ml_rul_hours",    m.get("rul_hours", 0))
                hc     = health_to_color(health)
                icon   = get_state_icon(state)
                fpc    = "#FF1744" if fp  > 70 else "#FFB800" if fp  > 40 else "#00FF9D"
                rulc   = "#FF1744" if rul < 48 else "#FFB800" if rul < 168 else "#00D4FF"
                st.markdown(f"""
                <div class="machine-card {state}">
                    <div class="mc-header">
                        <div>
                            <div class="mc-id">{m['machine_id']}</div>
                            <div class="mc-type">{m.get('machine_type','Unknown')}</div>
                        </div>
                        <div>
                            <div class="mc-icon">{icon}</div>
                            <div class="mc-state" style="color:{hc};">{state}</div>
                        </div>
                    </div>
                    <div class="hbar-row">
                        <span class="hbar-lbl">HEALTH</span>
                        <span class="hbar-val" style="color:{hc};">{health:.0f}%</span>
                    </div>
                    <div class="hbar-bg"><div class="hbar-fill" style="width:{health}%;background:{hc};"></div></div>
                    <div class="sgrid">
                        <div class="sitem"><div class="sval" style="color:#FF6B6B;">{m.get('temperature',0):.1f}°C</div><div class="slbl">TEMP</div></div>
                        <div class="sitem"><div class="sval" style="color:#FFB800;">{m.get('vibration',0):.2f}</div><div class="slbl">VIB mm/s</div></div>
                        <div class="sitem"><div class="sval" style="color:#00D4FF;">{m.get('pressure',0):.1f}</div><div class="slbl">PRESS bar</div></div>
                        <div class="sitem"><div class="sval" style="color:#00FF9D;">{m.get('energy_consumption',0):.0f}</div><div class="slbl">ENERGY kW</div></div>
                        <div class="sitem"><div class="sval" style="color:#B000FF;">{m.get('load_factor',0):.2f}</div><div class="slbl">LOAD</div></div>
                        <div class="sitem"><div class="sval" style="color:#FF7043;">{m.get('rpm',0):.0f}</div><div class="slbl">RPM</div></div>
                    </div>
                    <div class="mc-footer">
                        <div><div class="fstat-lbl">FAIL PROB</div><div class="fstat-val" style="color:{fpc};">{fp:.0f}%</div></div>
                        <div><div class="fstat-lbl">RUL</div><div class="fstat-val" style="color:{rulc};">{format_rul(rul)}</div></div>
                        <div><div class="fstat-lbl">RUNTIME</div><div class="fstat-val" style="color:#6B8CAE;">{m.get('runtime_hours',0):.0f}h</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)


# ─── LIVE SENSOR CHARTS ────────────────────────────────────────────────────────

def render_sensor_charts(readings_df: pd.DataFrame, selected_machine: str):
    row = readings_df[readings_df["machine_id"] == selected_machine]
    if row.empty:
        st.warning(f"No data for {selected_machine}")
        return
    row = row.iloc[0]

    sensors = ["temperature","vibration","pressure","energy_consumption","load_factor"]
    labels  = ["Temperature","Vibration","Pressure","Energy","Load"]
    vals = [round(row.get(s,0) / (readings_df[s].max() + 1e-6), 3)
            if s in readings_df.columns else 0 for s in sensors]

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=labels+[labels[0]],
            fill="toself", fillcolor="rgba(0,212,255,0.12)",
            line=dict(color="#00D4FF", width=2),
            marker=dict(size=7, color="#00D4FF"),
        ))
        fig.update_layout(**plotly_layout,
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_COLOR,
                                tickfont=dict(size=8, color=TEXT_COLOR)),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10, color="#E8F4FD")),
            ),
            title=f"Sensor Profile — {selected_machine}", height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gdata = [
            ("temperature",       "°C",  "#FF6B6B", 0, 500),
            ("vibration",         "mm/s","#FFB800", 0, 15),
            ("pressure",          "bar", "#00D4FF", 0, 300),
            ("energy_consumption","kW",  "#00FF9D", 0, 1200),
            ("load_factor",       "",    "#B000FF", 0, 1),
            ("rpm",               "RPM", "#FF7043", 0, 4000),
        ]
        gcols = st.columns(3)
        for idx, (sensor, unit, color, mn, mx) in enumerate(gdata):
            val = float(row.get(sensor, 0))
            with gcols[idx % 3]:
                fg = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={"suffix": f" {unit}", "font": {"size":12,"color":color,"family":"JetBrains Mono"}},
                    title={"text": sensor.replace("_"," ").title(), "font": {"size":8,"color":TEXT_COLOR}},
                    gauge={
                        "axis": {"range":[mn,mx],"tickfont":{"size":7,"color":"#4A6A8A"}},
                        "bar":  {"color":color,"thickness":0.3},
                        "bgcolor":"#0A1628","bordercolor":GRID_COLOR,
                    },
                ))
                fg.update_layout(**plotly_layout, height=150, margin=dict(l=10,r=10,t=32,b=5))
                st.plotly_chart(fg, use_container_width=True)


# ─── ALERTS ───────────────────────────────────────────────────────────────────

def render_alerts(alerts: list):
    if not alerts:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#4A6A8A;">
            <div style="font-size:4rem;margin-bottom:14px;">✅</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;letter-spacing:3px;color:#6B8CAE;">ALL SYSTEMS NOMINAL</div>
            <div style="font-size:.82rem;margin-top:8px;">No active alerts across the fleet.</div>
        </div>""", unsafe_allow_html=True)
        return

    bmap = {"critical":"bc","warning":"bw","info":"bi"}
    for alert in alerts[:15]:
        level = alert.get("level","info")
        mid   = alert.get("machine_id","UNKNOWN")
        mtype = alert.get("machine_type","")
        msgs  = alert.get("messages",[])
        rul   = alert.get("rul_hours",0)
        fp    = alert.get("failure_prob",0)
        icon  = "🔴" if level=="critical" else "🟡" if level=="warning" else "🔵"
        fpc   = "#FF1744" if fp>70 else "#FFB800" if fp>40 else "#00FF9D"
        rulc  = "#FF1744" if rul<48 else "#FFB800" if rul<168 else "#00D4FF"
        mhtml = "<br>".join(f"• {x}" for x in msgs)
        bc    = bmap.get(level,"bi")
        st.markdown(f"""
        <div class="alert-card alert-{level}">
            <div class="alert-top">
                <div class="alert-left">
                    <div class="abadge {bc}">{icon} {level.upper()}</div>
                    <div class="amachine">{mid} <span style="color:#4A6A8A;font-size:.7rem;font-weight:400;">— {mtype}</span></div>
                    <div class="amsg">{mhtml}</div>
                </div>
                <div class="aright">
                    <div style="font-size:.6rem;color:#4A6A8A;margin-bottom:3px;">FAIL PROB</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:{fpc};font-weight:700;">{fp:.0f}%</div>
                    <div style="font-size:.6rem;color:#4A6A8A;margin:5px 0 3px;">RUL</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.88rem;color:{rulc};font-weight:700;">{format_rul(rul)}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ─── AI INSIGHTS ──────────────────────────────────────────────────────────────

def render_ai_insights(readings_df: pd.DataFrame, pipeline: IndustrialMLPipeline):
    if pipeline is None or not pipeline.is_trained:
        st.info("🤖 Train the AI models first (Control Panel above).")
        return

    fc = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
    rc = "ml_rul_hours"    if "ml_rul_hours"    in readings_df.columns else "rul_hours"
    at_risk = readings_df.nlargest(6, fc)

    st.markdown("""
    <div class="section-header">
        <div class="section-title">🤖 AI Predictive Insights</div>
        <div class="section-badge">Top 6 At-Risk</div>
    </div>""", unsafe_allow_html=True)

    for _, row in at_risk.iterrows():
        insight = pipeline.generate_insight_text(row)
        fp   = row.get(fc, 0)
        rul  = row.get(rc, 0)
        state= row.get("state","unknown")
        c1,c2,c3,c4 = st.columns([2.8,0.9,0.9,1.5])
        with c1: st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        with c2: st.metric("Fail Prob", f"{fp:.0f}%")
        with c3: st.metric("RUL", format_rul(rul))
        with c4:
            recs = get_recommendations(state)
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #1A3A5C;border-radius:8px;
                        padding:12px;font-size:.75rem;min-height:66px;">
                <div style="color:#6B8CAE;font-size:.6rem;letter-spacing:1px;margin-bottom:5px;text-transform:uppercase;">Recommended Action</div>
                <div style="color:#E8F4FD;line-height:1.4;">{recs[0] if recs else '—'}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 🔍 Feature Importance")
        if hasattr(pipeline,"failure_predictor") and pipeline.failure_predictor.feature_importances:
            items = sorted(pipeline.failure_predictor.feature_importances.items(),
                           key=lambda x:x[1], reverse=True)[:12]
            names = [n.replace("_rolling_mean_6"," (avg)").replace("_delta"," (Δ)").replace("_"," ").title()
                     for n,_ in items]
            vals  = [v for _,v in items]
            clrs  = ["#00D4FF","#00FF9D","#00FF9D","#FFB800","#FFB800","#FFB800"] + ["#4A6A8A"]*6
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                                   marker_color=clrs[:len(vals)], marker_line_width=0))
            fig.update_layout(**plotly_layout, height=380, title="Failure Predictor Features")
            fig.update_xaxes(**_AX)
            fig.update_yaxes(**_AX, autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown("### 📊 Sensor Correlation Matrix")
        scols = [c for c in ["temperature","vibration","pressure","energy_consumption","load_factor"]
                 if c in readings_df.columns]
        if scols:
            corr = readings_df[scols].corr()
            fig2 = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0,"#060E1F"],[0.5,"#1A3A5C"],[1,"#00D4FF"]],
                text=np.round(corr.values,2),
                texttemplate="%{text}", textfont=dict(size=10),
                hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.3f}<extra></extra>",
                zmin=-1, zmax=1,
            ))
            fig2.update_layout(**plotly_layout, height=380, title="Sensor Correlation Matrix",
                               margin=dict(l=100,r=20,t=50,b=80))
            st.plotly_chart(fig2, use_container_width=True)


# ─── HISTORICAL ANALYTICS ──────────────────────────────────────────────────────

def render_historical_analytics(hist_df: pd.DataFrame, selected_machine: str):
    if hist_df is None or hist_df.empty:
        st.info("📊 Train AI models first to generate historical data.")
        return
    df_m = hist_df[hist_df["machine_id"]==selected_machine].copy()
    if df_m.empty:
        st.info(f"No historical data for {selected_machine}")
        return
    df_m["timestamp"] = pd.to_datetime(df_m["timestamp"])
    df_m = df_m.sort_values("timestamp")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=["🌡 Temperature (°C)","〰 Vibration (mm/s)",
                        "⬛ Pressure (bar)","⚡ Energy (kW)",
                        "📉 Degradation Level","⏱ RUL (hours)"],
        vertical_spacing=0.13, horizontal_spacing=0.08,
    )
    panels = [
        ("temperature",       1,1,"#FF6B6B"),
        ("vibration",         1,2,"#FFB800"),
        ("pressure",          2,1,"#00D4FF"),
        ("energy_consumption",2,2,"#00FF9D"),
        ("degradation_level", 3,1,"#B000FF"),
        ("rul_hours",         3,2,"#FF7043"),
    ]
    for sensor,r,c,color in panels:
        if sensor not in df_m.columns: continue
        rv,gv,bv = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig.add_trace(go.Scatter(
            x=df_m["timestamp"], y=df_m[sensor], mode="lines",
            line=dict(color=color,width=1.8),
            fill="tozeroy", fillcolor=f"rgba({rv},{gv},{bv},0.07)",
            name=sensor.replace("_"," ").title(),
            hovertemplate=f"%{{y:.2f}}<extra>{sensor}</extra>",
        ), row=r, col=c)

    fig.update_layout(**plotly_layout, height=680,
                      title=f"Historical Analysis — {selected_machine}", showlegend=False)
    for i in range(1,4):
        for j in range(1,3):
            fig.update_xaxes(**_AX, row=i, col=j)
            fig.update_yaxes(**_AX, row=i, col=j)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 Fleet Degradation & RUL Trends")
    ca, cb = st.columns(2)
    palette = ["#00D4FF","#00FF9D","#FFB800","#FF7043","#FF1744","#B000FF",
               "#2196F3","#4CAF50","#E91E63","#009688","#FF5722","#607D8B"]

    with ca:
        fig_d = go.Figure()
        for i, mid in enumerate(hist_df["machine_id"].unique()[:12]):
            dm = hist_df[hist_df["machine_id"]==mid].copy()
            dm["timestamp"] = pd.to_datetime(dm["timestamp"])
            dm = dm.sort_values("timestamp")
            if "degradation_level" in dm.columns:
                fig_d.add_trace(go.Scatter(
                    x=dm["timestamp"], y=dm["degradation_level"]*100,
                    mode="lines", name=mid,
                    line=dict(color=palette[i%len(palette)],width=1.5), opacity=0.85,
                ))
        fig_d.update_layout(**plotly_layout, title="Degradation % Over Time", height=330)
        fig_d.update_xaxes(**_AX)
        fig_d.update_yaxes(**_AX, title_text="Degradation %")
        st.plotly_chart(fig_d, use_container_width=True)

    with cb:
        fig_r = go.Figure()
        for i, mid in enumerate(hist_df["machine_id"].unique()[:12]):
            dm = hist_df[hist_df["machine_id"]==mid].copy()
            dm["timestamp"] = pd.to_datetime(dm["timestamp"])
            dm = dm.sort_values("timestamp")
            if "rul_hours" in dm.columns:
                fig_r.add_trace(go.Scatter(
                    x=dm["timestamp"], y=dm["rul_hours"],
                    mode="lines", name=mid,
                    line=dict(color=palette[i%len(palette)],width=1.5), opacity=0.85,
                ))
        fig_r.update_layout(**plotly_layout, title="RUL Trends (Hours)", height=330)
        fig_r.update_xaxes(**_AX)
        fig_r.update_yaxes(**_AX, title_text="RUL (hours)")
        st.plotly_chart(fig_r, use_container_width=True)


# ─── MODEL PERFORMANCE ─────────────────────────────────────────────────────────

def render_model_performance(pipeline: IndustrialMLPipeline, hist_df: pd.DataFrame):
    if pipeline is None or not pipeline.is_trained:
        st.info("Train the AI models to see performance metrics.")
        return
    m = pipeline.train_metrics
    if not m: return

    st.markdown("### 📊 Model Performance Report")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Failure AUC-ROC", f"{m.get('failure_auc',0):.4f}")
    with c2: st.metric("RUL MAE (hours)", f"{m.get('rul_mae_hours',0):.1f}h")
    with c3: st.metric("RUL R²",          f"{m.get('rul_r2',0):.4f}")
    with c4: st.metric("Anomaly Rate",    f"{m.get('anomaly_rate_pct',0):.1f}%")
    with c5: st.metric("Train Samples",   f"{m.get('total_samples',0):,}")

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 🏗️ Model Architecture")
        st.code("""
┌─────────────────────────────────────────────────────────────┐
│              IntelliOps AI Pipeline v2.0                    │
├──────────────┬──────────────────────┬──────────────────────┤
│ ANOMALY      │ FAILURE PREDICTOR    │ RUL ESTIMATOR        │
│ Isolation    │ Gradient Boosting    │ Random Forest        │
│ Forest       │ n_est=200, lr=0.08   │ n_est=200, depth=12  │
│ n_est=150    │ max_depth=5          │ min_samples=5        │
│ contam=8%    │ subsample=0.8        │                      │
│ → AnomalyScore│ → P(fail) 0–100%   │ → RUL Hours          │
└──────────────┴──────────────────────┴──────────────────────┘
Feature Engineering: Raw → Rolling(6pt) → Delta → Cross-Sensor
Sensors: temp · vib · press · energy · load · runtime · RPM
""", language="text")

    with cb:
        perf = [
            m.get("failure_auc",0)*100,
            max(0, 100 - m.get("rul_mae_hours",100)),
            m.get("rul_r2",0)*100,
        ]
        lbls  = ["Failure AUC","RUL Accuracy","RUL R²"]
        clrs2 = ["#00D4FF","#00FF9D","#FFB800"]
        fig_p = go.Figure()
        for lbl, val, clr in zip(lbls, perf, clrs2):
            fig_p.add_trace(go.Bar(
                x=[val], y=[lbl], orientation="h",
                marker_color=clr, marker_line_width=0,
                text=[f"{val:.1f}%"], textposition="inside",
                textfont=dict(size=11,color="white"), name=lbl,
            ))
        fig_p.update_layout(**plotly_layout, title="Performance Summary",
                            height=280, showlegend=False)
        fig_p.update_xaxes(**_AX, range=[0,105])
        fig_p.update_yaxes(**_AX)
        st.plotly_chart(fig_p, use_container_width=True)


# ─── OPERATIONS DASHBOARD ──────────────────────────────────────────────────────

def render_operations_dashboard(readings_df: pd.DataFrame, kpis: dict):
    st.markdown("""
    <div class="section-header">
        <div class="section-title">🔧 Operations Intelligence</div>
        <div class="section-badge">Maintenance Planning</div>
    </div>""", unsafe_allow_html=True)

    fc = "ml_failure_prob" if "ml_failure_prob" in readings_df.columns else "failure_probability"
    rc = "ml_rul_hours"    if "ml_rul_hours"    in readings_df.columns else "rul_hours"

    cols_sel = ["machine_id","machine_type","state","health_score",fc,rc,"runtime_hours"]
    cols_sel = [c for c in cols_sel if c in readings_df.columns]
    df_ops   = readings_df[cols_sel].copy()
    df_ops.columns = ["Machine","Type","State","Health %","Fail Prob %","RUL (h)","Runtime (h)"][:len(cols_sel)]
    df_ops = df_ops.sort_values("Fail Prob %" if "Fail Prob %" in df_ops.columns else df_ops.columns[0],
                                 ascending=False)
    if "Fail Prob %" in df_ops.columns:
        df_ops["Priority"] = df_ops["Fail Prob %"].apply(
            lambda x: "🔴 CRITICAL" if x>70 else "🟡 HIGH" if x>40 else "🟢 NORMAL")
    if "RUL (h)" in df_ops.columns:
        df_ops["Window"] = df_ops["RUL (h)"].apply(
            lambda x: "IMMEDIATE" if x<48 else f"≤{int(x/24)}d" if x<720 else ">30d")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### 📋 Maintenance Priority Queue")
        fmt_cols = {c: "{:.1f}" for c in ["Health %","Fail Prob %"] if c in df_ops.columns}
        if "RUL (h)" in df_ops.columns:     fmt_cols["RUL (h)"]     = "{:.0f}"
        if "Runtime (h)" in df_ops.columns: fmt_cols["Runtime (h)"] = "{:.0f}"
        grad_col = "Fail Prob %" if "Fail Prob %" in df_ops.columns else None
        styled = df_ops.style
        if grad_col:
            styled = styled.background_gradient(subset=[grad_col], cmap="RdYlGn_r")
        st.dataframe(styled.format(fmt_cols), use_container_width=True, height=380)

    with c2:
        st.markdown("#### 💰 Cost-Benefit Analysis")
        nc = kpis.get("critical_count",0)
        na = kpis.get("at_risk_count",0)
        cr = nc*85000 + na*25000
        cp = (nc+na)*8000
        sv = cr - cp
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:22px;">
            <div style="font-size:.7rem;color:#6B8CAE;letter-spacing:2px;margin-bottom:16px;">COST ANALYSIS (USD)</div>
            <div style="margin-bottom:14px;">
                <div style="font-size:.68rem;color:#4A6A8A;margin-bottom:3px;">Reactive Failure Cost</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;color:#FF1744;font-weight:700;">${cr:,.0f}</div>
            </div>
            <div style="margin-bottom:14px;">
                <div style="font-size:.68rem;color:#4A6A8A;margin-bottom:3px;">Preventive Maint. Cost</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;color:#FFB800;font-weight:700;">${cp:,.0f}</div>
            </div>
            <div style="border-top:1px solid #1A3A5C;padding-top:14px;">
                <div style="font-size:.68rem;color:#4A6A8A;margin-bottom:3px;">Estimated Savings</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;color:#00FF9D;font-weight:700;">${sv:,.0f}</div>
                <div style="font-size:.66rem;color:#00FF9D;margin-top:4px;">↑ {sv/max(cr,1)*100:.0f}% cost reduction</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Maintenance timeline
    st.markdown("#### 📅 Maintenance Schedule Forecast")
    if "RUL (h)" in df_ops.columns and "Machine" in df_ops.columns:
        fig_tl = go.Figure()
        today  = datetime.now()
        palette_t = ["#FF1744","#FFB800","#00FF9D"]
        for _, r in df_ops.iterrows():
            rul_h = r.get("RUL (h)", 0)
            clr   = "#FF1744" if rul_h<48 else "#FFB800" if rul_h<168 else "#00D4FF"
            fig_tl.add_trace(go.Scatter(
                x=[today + timedelta(hours=float(rul_h))],
                y=[r["Machine"]],
                mode="markers+text",
                text=[f"  {format_rul(rul_h)}"],
                textfont=dict(size=9,color=clr),
                marker=dict(size=14,color=clr,symbol="diamond",
                            line=dict(width=1,color="#1A3A5C")),
                name=r["Machine"],
                hovertemplate=f"<b>{r['Machine']}</b><br>Due in {format_rul(rul_h)}<extra></extra>",
                showlegend=False,
            ))
        fig_tl.add_vline(x=today,                     line_dash="solid", line_color="#00D4FF", line_width=2,
                         annotation_text="NOW",       annotation_font_color="#00D4FF",annotation_font_size=10)
        fig_tl.add_vline(x=today+timedelta(days=7),   line_dash="dash",  line_color="#FFB800", line_width=1,
                         annotation_text="7d",        annotation_font_color="#FFB800",annotation_font_size=9)
        fig_tl.add_vline(x=today+timedelta(days=30),  line_dash="dash",  line_color="#00FF9D", line_width=1,
                         annotation_text="30d",       annotation_font_color="#00FF9D",annotation_font_size=9)
        fig_tl.update_layout(**plotly_layout, height=380, title="Fleet Maintenance Timeline", showlegend=False)
        fig_tl.update_xaxes(**_AX, title_text="Date")
        fig_tl.update_yaxes(**_AX)
        st.plotly_chart(fig_tl, use_container_width=True)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    gen: SensorDataGenerator = st.session_state.generator

    # Auto-load saved models
    if st.session_state.pipeline is None:
        p = IndustrialMLPipeline()
        if p.models_exist() and p.load_all():
            st.session_state.pipeline      = p
            st.session_state.training_done = True
            st.session_state.train_metrics = p.train_metrics
        else:
            st.session_state.pipeline = p

    pipeline: IndustrialMLPipeline = st.session_state.pipeline

    # Generate live readings
    readings_df = gen.generate_all_readings()
    if pipeline and pipeline.is_trained:
        try: readings_df = pipeline.predict_fleet(readings_df)
        except Exception: pass

    st.session_state.live_readings  = readings_df
    st.session_state.refresh_count += 1

    kpis   = compute_fleet_kpis(readings_df)
    alerts = generate_alerts(readings_df)

    # ── TOP NAVBAR
    render_top_navbar(kpis, len(alerts))

    # ── CONTROL BAR
    refresh_rate = render_control_bar(gen, pipeline)

    # ── Selected machine fallback
    mids = list(readings_df["machine_id"].unique())
    if not st.session_state.selected_machine or st.session_state.selected_machine not in mids:
        st.session_state.selected_machine = mids[0]
    selected_machine = st.session_state.selected_machine

    # ── KPI ROW
    render_kpi_row(kpis, readings_df)

    # ── MAIN TABS
    tabs = st.tabs([
        "🏭  Fleet Overview",
        "🖥️  Machine Grid",
        "📡  Live Monitoring",
        "🚨  Alert Center",
        "🤖  AI Insights",
        "📊  Historical",
        "🔧  Operations",
        "⚙️  Models",
    ])

    with tabs[0]:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🏭 Fleet Overview</div>
            <div class="section-badge">Real-time</div>
        </div>""", unsafe_allow_html=True)
        render_fleet_overview(readings_df)

    with tabs[1]:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🖥️ Machine Status Grid</div>
        </div>""", unsafe_allow_html=True)
        render_machine_cards(readings_df)

    with tabs[2]:
        rc_txt = st.session_state.refresh_count
        ts     = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title">📡 Live Sensor Monitoring</div>
            <div class="live-badge"><span class="live-dot"></span>STREAMING</div>
            <div style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#4A6A8A;">
                Refresh #{rc_txt} · {ts}
            </div>
        </div>""", unsafe_allow_html=True)
        render_sensor_charts(readings_df, selected_machine)

        st.markdown("### 📋 Live Table — All Machines")
        dcols = ["machine_id","machine_type","state","temperature","vibration","pressure",
                 "energy_consumption","load_factor","health_score","runtime_hours"]
        if "ml_failure_prob" in readings_df.columns: dcols += ["ml_failure_prob","ml_rul_hours"]
        dcols = [c for c in dcols if c in readings_df.columns]
        ddf   = readings_df[dcols].copy()
        ddf.columns = [c.replace("_"," ").title().replace("Ml ","AI ") for c in dcols]
        hc = "Health Score"
        styled = ddf.style.background_gradient(subset=[hc],cmap="RdYlGn").format(precision=2) \
                 if hc in ddf.columns else ddf.style.format(precision=2)
        st.dataframe(styled, use_container_width=True, height=320)

    with tabs[3]:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🚨 Alert Center</div>
        </div>""", unsafe_allow_html=True)
        nc2 = sum(1 for a in alerts if a["level"]=="critical")
        nw2 = sum(1 for a in alerts if a["level"]=="warning")
        ac1,ac2,ac3 = st.columns(3)
        for col,(val,lbl,color,icon) in zip([ac1,ac2,ac3],[
            (nc2,"Critical Alerts","red","🔴"),
            (nw2,"Warnings","amber","🟡"),
            (kpis.get("at_risk_count",0),"Machines At Risk","cyan","⚠️"),
        ]):
            vc = {"red":"#FF1744","amber":"#FFB800","cyan":"#00D4FF"}[color]
            with col:
                st.markdown(f"""
                <div class="kpi-card {color}">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value" style="color:{vc};">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        render_alerts(alerts)

    with tabs[4]:
        render_ai_insights(readings_df, pipeline)

    with tabs[5]:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">📊 Historical Analytics</div>
        </div>""", unsafe_allow_html=True)
        if st.session_state.historical_df is None and pipeline and pipeline.is_trained:
            with st.spinner("Loading historical data…"):
                st.session_state.historical_df = gen.generate_historical_data(n_hours=720, interval_minutes=30)
        render_historical_analytics(st.session_state.historical_df, selected_machine)

    with tabs[6]:
        render_operations_dashboard(readings_df, kpis)

    with tabs[7]:
        render_model_performance(pipeline, st.session_state.historical_df)

    # ── Auto refresh
    if st.session_state.auto_refresh:
        rates    = {"5s":5,"10s":10,"30s":30,"60s":60}
        interval = rates.get(st.session_state.refresh_rate, 10)
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
