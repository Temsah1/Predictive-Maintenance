# ===================================================================
# Nexus Predictive Maintenance - SLB Project
# A comprehensive predictive maintenance solution with admin analytics
# Author: Data Science Team
# Version: 2.0.0
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import io
import base64
import hashlib
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report,
                             mean_squared_error, r2_score)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import shap
import openpyxl
from io import BytesIO
import random
import string
import json
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# ===================================================================
# Page Configuration & Custom Dark Theme
# ===================================================================
st.set_page_config(
    page_title="Nexus Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and enhanced UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .css-1kyxreq {
        background-color: #1E1E2E;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Cards and containers */
    .st-bw, .st-cb, .st-cm, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu {
        background-color: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 1px solid #2D2D3D;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        box-shadow: 0 4px 8px rgba(255,75,75,0.3);
        transform: translateY(-2px);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1E1E2E !important;
        color: white !important;
    }
    
    /* File uploader */
    .st-bq, .st-br {
        border-color: #FF4B4B !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2D2D3D;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E2E;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* Login form */
    .login-box {
        background: rgba(30,30,46,0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        border: 1px solid #3A3A4A;
    }
    
    /* Animated background */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1A2E 100%);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1E1E2E;
    }
    ::-webkit-scrollbar-thumb {
        background: #FF4B4B;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Session State Initialization
# ===================================================================
def init_session_state():
    """Initialize all session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'maintenance_schedule' not in st.session_state:
        st.session_state.maintenance_schedule = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

init_session_state()

# ===================================================================
# Authentication System
# ===================================================================
ADMIN_EMAIL = "kareemeltemsah7@gmail.com"
ADMIN_PASSWORD = "temsah1"

def hash_password(password: str) -> str:
    """Simple password hashing for demo purposes"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(email: str, password: str) -> Tuple[bool, bool]:
    """
    Authenticate user and determine admin status.
    Returns: (success, is_admin)
    """
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return True, True
    elif email and password:  # Any non-empty credentials for regular users
        return True, False
    return False, False

def login_page():
    """Render the login page with modern UI"""
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 80vh;">
        <div style="width: 100%; max-width: 450px;">
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        # Logo and Title
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("https://img.icons8.com/fluency/96/000000/maintenance.png", width=80)
        st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 10px;'>Nexus Predictive</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #FF4B4B; margin-bottom: 30px;'>Maintenance System</h3>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("📧 Email Address", placeholder="your@email.com")
            password = st.text_input("🔒 Password", type="password", placeholder="••••••••")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                submitted = st.form_submit_button("Login →", use_container_width=True)
            
            if submitted:
                with st.spinner("Authenticating..."):
                    time.sleep(0.5)
                    success, is_admin = authenticate(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.is_admin = is_admin
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
        
        st.markdown("""
        <div style='margin-top: 20px; text-align: center; color: #888;'>
            <small>Demo credentials: any email/password (admin: kareemeltemsah7@gmail.com / temsah1)</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# ===================================================================
# Data Generation & Synthetic Dataset Creation
# ===================================================================
@st.cache_data
def generate_synthetic_machine_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset for predictive maintenance.
    Features include sensor readings and operational parameters.
    Target: 'failure' (binary) and 'time_to_failure' (days).
    """
    np.random.seed(42)
    
    # Base parameters
    n_machines = 50
    machine_ids = [f'M{str(i).zfill(3)}' for i in range(1, n_machines+1)]
    
    data = []
    current_date = datetime.now()
    
    for _ in range(n_samples):
        machine_id = np.random.choice(machine_ids)
        
        # Operational parameters
        runtime_hours = np.random.uniform(100, 5000)
        load_percent = np.random.uniform(20, 100)
        ambient_temp = np.random.normal(25, 5)
        
        # Sensor readings (normal operation + degradation patterns)
        vibration_rms = np.random.weibull(1.5) * (1 + runtime_hours/5000) * (1 + (load_percent-50)/100)
        temperature = ambient_temp + 15 + (runtime_hours/1000) * np.random.normal(1, 0.2) + (load_percent/100)*10
        pressure = 100 + np.random.normal(0, 5) + (runtime_hours/2000) * 2
        current_draw = 10 + (load_percent/100)*15 + np.random.normal(0, 1) + (runtime_hours/3000)*3
        voltage = 220 + np.random.normal(0, 2) - (runtime_hours/5000)*1
        
        # Additional features
        oil_viscosity = 40 + np.random.normal(0, 3) - (runtime_hours/4000)*5
        bearing_temp_diff = abs(temperature - ambient_temp - 20) + np.random.normal(0, 2)
        vibration_fft_1x = vibration_rms * np.random.uniform(0.8, 1.2)
        vibration_fft_2x = vibration_fft_1x * np.random.uniform(0.1, 0.5)
        vibration_fft_3x = vibration_fft_1x * np.random.uniform(0.05, 0.2)
        
        # Maintenance history
        days_since_last_maintenance = np.random.exponential(30) + runtime_hours/24
        maintenance_count = np.random.poisson(runtime_hours/1000)
        
        # Failure logic (correlated with degradation)
        failure_prob = (
            0.3 * (vibration_rms > 2.5) +
            0.2 * (temperature > 45) +
            0.2 * (pressure > 110) +
            0.15 * (bearing_temp_diff > 10) +
            0.15 * (days_since_last_maintenance > 60)
        )
        failure = 1 if np.random.random() < failure_prob else 0
        
        # Time to failure (if failure occurred)
        if failure:
            time_to_failure = np.random.uniform(0, 30)
        else:
            time_to_failure = np.random.uniform(30, 365)
        
        # Timestamp
        timestamp = current_date - timedelta(days=np.random.uniform(0, 365))
        
        data.append({
            'timestamp': timestamp,
            'machine_id': machine_id,
            'runtime_hours': runtime_hours,
            'load_percent': load_percent,
            'ambient_temp': ambient_temp,
            'vibration_rms': vibration_rms,
            'temperature': temperature,
            'pressure': pressure,
            'current_draw': current_draw,
            'voltage': voltage,
            'oil_viscosity': oil_viscosity,
            'bearing_temp_diff': bearing_temp_diff,
            'vibration_fft_1x': vibration_fft_1x,
            'vibration_fft_2x': vibration_fft_2x,
            'vibration_fft_3x': vibration_fft_3x,
            'days_since_last_maintenance': days_since_last_maintenance,
            'maintenance_count': maintenance_count,
            'failure': failure,
            'time_to_failure': time_to_failure
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ===================================================================
# Model Training Pipeline
# ===================================================================
@st.cache_resource
def train_and_save_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler, List[str], Dict]:
    """
    Train a Random Forest model with hyperparameter tuning.
    Returns model, scaler, feature names, and metrics.
    """
    # Feature engineering
    feature_cols = [
        'runtime_hours', 'load_percent', 'ambient_temp', 'vibration_rms',
        'temperature', 'pressure', 'current_draw', 'voltage', 'oil_viscosity',
        'bearing_temp_diff', 'vibration_fft_1x', 'vibration_fft_2x', 'vibration_fft_3x',
        'days_since_last_maintenance', 'maintenance_count'
    ]
    
    X = df[feature_cols].copy()
    y = df['failure'].copy()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing pipeline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': auc(*roc_curve(y_test, y_pred_proba)[:2]),
        'best_params': grid_search.best_params_,
        'feature_importance': dict(zip(feature_cols, best_model.feature_importances_)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return best_model, scaler, feature_cols, metrics

# ===================================================================
# Prediction & Maintenance Scheduling
# ===================================================================
def predict_failure(model, scaler, feature_names, input_df: pd.DataFrame) -> pd.DataFrame:
    """Predict failure probability and class for new data"""
    # Ensure all required features are present
    missing_cols = set(feature_names) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
    X = input_df[feature_names].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = model.predict(X_scaled)
    
    result_df = input_df.copy()
    result_df['failure_probability'] = proba
    result_df['predicted_failure'] = pred
    result_df['risk_level'] = pd.cut(
        proba, 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    return result_df

def generate_maintenance_schedule(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate recommended maintenance dates based on failure probability.
    """
    schedule = predictions_df.copy()
    
    # Define maintenance urgency based on risk
    def get_maintenance_days(row):
        prob = row['failure_probability']
        if prob >= 0.7:
            return np.random.randint(1, 4)  # Immediate (1-3 days)
        elif prob >= 0.4:
            return np.random.randint(4, 10)  # Soon (4-9 days)
        elif prob >= 0.2:
            return np.random.randint(10, 21)  # Scheduled (10-20 days)
        else:
            return np.random.randint(30, 60)  # Routine (30-60 days)
    
    schedule['days_until_maintenance'] = schedule.apply(get_maintenance_days, axis=1)
    schedule['recommended_maintenance_date'] = (
        datetime.now() + pd.to_timedelta(schedule['days_until_maintenance'], unit='D')
    ).dt.strftime('%Y-%m-%d')
    
    # Priority ranking
    schedule['priority'] = pd.cut(
        schedule['failure_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return schedule

# ===================================================================
# Excel Template Generation
# ===================================================================
def create_excel_template() -> bytes:
    """Generate an Excel template with required columns for upload"""
    template_cols = [
        'timestamp', 'machine_id', 'runtime_hours', 'load_percent', 'ambient_temp',
        'vibration_rms', 'temperature', 'pressure', 'current_draw', 'voltage',
        'oil_viscosity', 'bearing_temp_diff', 'vibration_fft_1x', 'vibration_fft_2x',
        'vibration_fft_3x', 'days_since_last_maintenance', 'maintenance_count'
    ]
    
    # Create sample rows
    sample_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 3,
        'machine_id': ['M001', 'M002', 'M003'],
        'runtime_hours': [1200.5, 3500.2, 800.0],
        'load_percent': [75.0, 92.0, 45.0],
        'ambient_temp': [25.5, 28.0, 22.0],
        'vibration_rms': [1.8, 3.2, 1.2],
        'temperature': [42.0, 55.0, 35.0],
        'pressure': [102.0, 115.0, 98.0],
        'current_draw': [18.5, 25.0, 12.0],
        'voltage': [219.0, 218.0, 221.0],
        'oil_viscosity': [38.0, 32.0, 42.0],
        'bearing_temp_diff': [12.0, 22.0, 5.0],
        'vibration_fft_1x': [1.5, 2.8, 1.0],
        'vibration_fft_2x': [0.4, 1.2, 0.2],
        'vibration_fft_3x': [0.1, 0.5, 0.05],
        'days_since_last_maintenance': [45, 120, 15],
        'maintenance_count': [2, 5, 1]
    }
    
    df_template = pd.DataFrame(sample_data)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_template.to_excel(writer, sheet_name='Machine Data', index=False)
        # Add instructions sheet
        instructions = pd.DataFrame({
            'Column': template_cols,
            'Description': [
                'Timestamp of reading (YYYY-MM-DD HH:MM:SS)',
                'Unique machine identifier',
                'Total runtime in hours',
                'Current load percentage (0-100)',
                'Ambient temperature (°C)',
                'Vibration RMS (mm/s)',
                'Machine temperature (°C)',
                'Operating pressure (psi)',
                'Current draw (Amps)',
                'Voltage (Volts)',
                'Oil viscosity (cSt)',
                'Bearing temperature difference (°C)',
                'Vibration FFT 1x amplitude',
                'Vibration FFT 2x amplitude',
                'Vibration FFT 3x amplitude',
                'Days since last maintenance',
                'Total maintenance count'
            ],
            'Required': ['Yes'] * len(template_cols)
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    return output.getvalue()

def to_excel_download(df: pd.DataFrame, sheet_name: str = 'Predictions') -> bytes:
    """Convert dataframe to Excel bytes for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# ===================================================================
# Visualization Functions
# ===================================================================
def plot_feature_importance(metrics: Dict) -> go.Figure:
    """Create feature importance bar chart"""
    feat_imp = metrics['feature_importance']
    sorted_feat = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_feat[:15]]
    importances = [x[1] for x in sorted_feat[:15]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{v:.3f}' for v in importances],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Top 15 Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def plot_confusion_matrix(metrics: Dict) -> go.Figure:
    """Plot interactive confusion matrix"""
    cm = np.array(metrics['confusion_matrix'])
    labels = ['No Failure', 'Failure']
    
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def plot_roc_curve(metrics: Dict, y_test, y_pred_proba) -> go.Figure:
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = metrics['roc_auc']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='#FF4B4B', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    return fig

def plot_risk_distribution(predictions_df: pd.DataFrame) -> go.Figure:
    """Pie chart of risk levels"""
    risk_counts = predictions_df['risk_level'].value_counts()
    colors = {'Low': '#00CC96', 'Medium': '#FFA15A', 'High': '#EF553B'}
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker=dict(colors=[colors.get(level, '#808080') for level in risk_counts.index]),
        textinfo='label+percent',
        textfont=dict(color='white')
    )])
    fig.update_layout(
        title="Risk Level Distribution",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def plot_sensor_trends(df: pd.DataFrame) -> go.Figure:
    """Time series plot of key sensors"""
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Vibration RMS', 'Temperature', 'Pressure', 
                        'Current Draw', 'Oil Viscosity', 'Bearing Temp Diff'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['vibration_rms'], mode='lines', name='Vibration',
                   line=dict(color='#FF4B4B')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['temperature'], mode='lines', name='Temperature',
                   line=dict(color='#FFA15A')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['pressure'], mode='lines', name='Pressure',
                   line=dict(color='#00CC96')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['current_draw'], mode='lines', name='Current',
                   line=dict(color='#AB63FA')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['oil_viscosity'], mode='lines', name='Viscosity',
                   line=dict(color='#19D3F3')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bearing_temp_diff'], mode='lines', name='Bearing ΔT',
                   line=dict(color='#FF6692')),
        row=3, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# ===================================================================
# Admin Analytics Dashboard Components
# ===================================================================
def admin_dashboard():
    """Render the admin analytics dashboard"""
    st.title("🔐 Admin Analytics Dashboard")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.info("No data loaded. Generating synthetic dataset for demonstration...")
        with st.spinner("Generating data and training model..."):
            df = generate_synthetic_machine_data(5000)
            st.session_state.data = df
            model, scaler, features, metrics = train_and_save_model(df)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_names = features
            st.session_state.model_metrics = metrics
        st.success("Data and model ready!")
    
    df = st.session_state.data
    metrics = st.session_state.model_metrics
    
    # KPI Row
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{len(df):,}", delta=None)
    with col2:
        failure_rate = df['failure'].mean() * 100
        st.metric("Failure Rate", f"{failure_rate:.2f}%", 
                  delta=f"{failure_rate - 5:.2f}%" if failure_rate > 5 else f"{failure_rate - 5:.2f}%")
    with col3:
        st.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
    with col4:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col5:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    
    st.markdown("---")
    
    # Model Performance Section
    st.subheader("🤖 Model Performance Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance", "Metrics Report"])
    
    with tab1:
        fig_cm = plot_confusion_matrix(metrics)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab2:
        # Need y_test and y_pred_proba for ROC; we can recompute or store in session
        X = df[st.session_state.feature_names]
        y = df['failure']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_scaled = st.session_state.scaler.transform(X_test)
        y_pred_proba = st.session_state.model.predict_proba(X_test_scaled)[:, 1]
        fig_roc = plot_roc_curve(metrics, y_test, y_pred_proba)
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with tab3:
        fig_fi = plot_feature_importance(metrics)
        st.plotly_chart(fig_fi, use_container_width=True)
        
        # SHAP analysis (optional, compute if needed)
        if st.button("Generate SHAP Summary (may take a moment)"):
            with st.spinner("Computing SHAP values..."):
                # Sample for SHAP
                X_sample = X_test_scaled[:100]
                explainer = shap.TreeExplainer(st.session_state.model)
                shap_values = explainer.shap_values(X_sample)
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values[1], X_sample, feature_names=st.session_state.feature_names, 
                                  show=False, plot_type="bar")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab4:
        st.json(metrics['classification_report'])
    
    st.markdown("---")
    
    # Data Exploration
    st.subheader("📈 Data Exploration & Trends")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_trends = plot_sensor_trends(df.sample(min(500, len(df))))
        st.plotly_chart(fig_trends, use_container_width=True)
    with col2:
        # Correlation heatmap
        corr_cols = st.session_state.feature_names + ['failure']
        corr = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_corr.update_layout(
            title="Feature Correlations",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Model Management
    st.subheader("⚙️ Model Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retrain Model on Current Data"):
            with st.spinner("Retraining model with hyperparameter tuning..."):
                model, scaler, features, new_metrics = train_and_save_model(df)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.feature_names = features
                st.session_state.model_metrics = new_metrics
            st.success("Model retrained successfully!")
            st.rerun()
    
    with col2:
        # Export model
        if st.button("💾 Export Model (Joblib)"):
            model_bytes = BytesIO()
            joblib.dump(st.session_state.model, model_bytes)
            model_bytes.seek(0)
            st.download_button(
                label="Download Model File",
                data=model_bytes,
                file_name="nexus_rf_model.joblib",
                mime="application/octet-stream"
            )
    
    # User Management Simulation
    st.subheader("👥 System Users")
    users_df = pd.DataFrame({
        'Email': ['kareemeltemsah7@gmail.com', 'engineer1@slb.com', 'tech2@slb.com'],
        'Role': ['Admin', 'Engineer', 'Technician'],
        'Last Active': ['2024-01-15', '2024-01-14', '2024-01-13'],
        'Predictions Made': [45, 23, 12]
    })
    st.dataframe(users_df, use_container_width=True)

# ===================================================================
# Main Application Interface
# ===================================================================
def main_app():
    """Main application after login"""
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/maintenance.png", width=60)
    st.sidebar.title("Nexus Predictive Maintenance")
    st.sidebar.markdown(f"Welcome, **{st.session_state.user_email}**")
    if st.session_state.is_admin:
        st.sidebar.markdown("🔴 **Admin Mode**")
    st.sidebar.markdown("---")
    
    # Navigation
    if st.session_state.is_admin:
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Dashboard", "📤 Upload & Predict", "📅 Maintenance Schedule", 
             "📊 Analytics", "📈 Real-time Monitor", "⚙️ Settings"]
        )
    else:
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Dashboard", "📤 Upload & Predict", "📅 Maintenance Schedule", "📈 Real-time Monitor"]
        )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Page routing
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📤 Upload & Predict":
        upload_predict_page()
    elif page == "📅 Maintenance Schedule":
        schedule_page()
    elif page == "📊 Analytics" and st.session_state.is_admin:
        admin_dashboard()
    elif page == "📈 Real-time Monitor":
        realtime_monitor_page()
    elif page == "⚙️ Settings" and st.session_state.is_admin:
        settings_page()

def dashboard_page():
    """User dashboard overview"""
    st.title("🏠 Maintenance Dashboard")
    
    if st.session_state.data is None:
        st.info("👋 Welcome! Start by uploading data or use the demo dataset.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Demo Dataset", use_container_width=True):
                with st.spinner("Loading synthetic data..."):
                    df = generate_synthetic_machine_data(2000)
                    st.session_state.data = df
                    model, scaler, features, metrics = train_and_save_model(df)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = features
                    st.session_state.model_metrics = metrics
                st.success("Demo data loaded!")
                st.rerun()
        with col2:
            st.markdown("Or go to **Upload & Predict** to use your own Excel file.")
        return
    
    df = st.session_state.data
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Machines", df['machine_id'].nunique())
    with col2:
        high_risk = 0
        if st.session_state.predictions is not None:
            high_risk = (st.session_state.predictions['risk_level'] == 'High').sum()
        st.metric("High Risk Alerts", high_risk, delta=None)
    with col3:
        next_maintenance = "N/A"
        if st.session_state.maintenance_schedule is not None:
            next_date = st.session_state.maintenance_schedule['recommended_maintenance_date'].min()
            next_maintenance = next_date
        st.metric("Next Maintenance", next_maintenance)
    
    # Recent predictions
    st.subheader("Recent Predictions")
    if st.session_state.predictions is not None:
        preds = st.session_state.predictions.head(10)
        st.dataframe(preds[['machine_id', 'failure_probability', 'risk_level', 'predicted_failure']], 
                     use_container_width=True)
    else:
        st.info("No predictions yet. Upload data to see predictions.")
    
    # Quick stats chart
    if st.session_state.predictions is not None:
        fig_risk = plot_risk_distribution(st.session_state.predictions)
        st.plotly_chart(fig_risk, use_container_width=True)

def upload_predict_page():
    """Upload Excel and run predictions"""
    st.title("📤 Upload & Predict")
    
    # Template download
    st.markdown("### 📋 Download Template")
    template_bytes = create_excel_template()
    st.download_button(
        label="⬇️ Download Excel Template",
        data=template_bytes,
        file_name="nexus_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel file with machine data",
        type=['xlsx', 'xls'],
        help="Upload an Excel file matching the template format."
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        try:
            df_input = pd.read_excel(uploaded_file, sheet_name=0)
            st.success(f"✅ File loaded: {len(df_input)} rows, {len(df_input.columns)} columns")
            
            with st.expander("Preview Uploaded Data", expanded=True):
                st.dataframe(df_input.head(10), use_container_width=True)
            
            # Check if model exists
            if st.session_state.model is None:
                st.warning("No model available. Loading default model...")
                with st.spinner("Training model on synthetic data..."):
                    synth_df = generate_synthetic_machine_data(5000)
                    model, scaler, features, metrics = train_and_save_model(synth_df)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = features
                    st.session_state.model_metrics = metrics
            
            # Predict button
            if st.button("🔮 Predict Failures & Generate Schedule", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Make predictions
                        preds = predict_failure(
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.feature_names,
                            df_input
                        )
                        st.session_state.predictions = preds
                        
                        # Generate schedule
                        schedule = generate_maintenance_schedule(preds)
                        st.session_state.maintenance_schedule = schedule
                        
                        st.success("Predictions completed!")
                        
                        # Show results
                        st.subheader("Prediction Results")
                        st.dataframe(preds[['machine_id', 'failure_probability', 'risk_level', 'predicted_failure']], 
                                     use_container_width=True)
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            excel_preds = to_excel_download(preds, 'Predictions')
                            st.download_button(
                                label="📥 Download Predictions (Excel)",
                                data=excel_preds,
                                file_name="nexus_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        with col2:
                            excel_schedule = to_excel_download(schedule, 'Maintenance_Schedule')
                            st.download_button(
                                label="📅 Download Maintenance Schedule",
                                data=excel_schedule,
                                file_name="nexus_maintenance_schedule.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        # Risk distribution
                        fig_risk = plot_risk_distribution(preds)
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("Please ensure your Excel file contains all required columns. Download the template for reference.")
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

def schedule_page():
    """View and manage maintenance schedule"""
    st.title("📅 Maintenance Schedule")
    
    if st.session_state.maintenance_schedule is None:
        st.info("No maintenance schedule generated yet. Please upload data and run predictions first.")
        return
    
    schedule = st.session_state.maintenance_schedule
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by Priority",
            options=['Low', 'Medium', 'High'],
            default=['High', 'Medium']
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ['recommended_maintenance_date', 'failure_probability', 'priority']
        )
    
    filtered = schedule[schedule['priority'].isin(risk_filter)]
    filtered = filtered.sort_values(sort_by, ascending=(sort_by != 'failure_probability'))
    
    st.dataframe(
        filtered[['machine_id', 'failure_probability', 'priority', 
                  'days_until_maintenance', 'recommended_maintenance_date']],
        use_container_width=True
    )
    
    # Calendar view (simplified)
    st.subheader("Upcoming Maintenance (Next 30 Days)")
    upcoming = schedule[pd.to_datetime(schedule['recommended_maintenance_date']) <= datetime.now() + timedelta(days=30)]
    if len(upcoming) > 0:
        fig = px.timeline(
            upcoming,
            x_start="recommended_maintenance_date",
            x_end=pd.to_datetime(upcoming['recommended_maintenance_date']) + timedelta(days=1),
            y="machine_id",
            color="priority",
            color_discrete_map={'High': '#EF553B', 'Medium': '#FFA15A', 'Low': '#00CC96'},
            title="Maintenance Timeline"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No maintenance scheduled in the next 30 days.")

def realtime_monitor_page():
    """Simulate real-time monitoring with live charts"""
    st.title("📈 Real-time Machine Monitoring")
    st.markdown("Simulated live sensor data (refreshes every 2 seconds)")
    
    # Auto-refresh control
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        time.sleep(2)
        st.rerun()
    
    # Generate random machine data
    n_machines = 5
    machine_ids = [f'M{str(i).zfill(3)}' for i in range(1, n_machines+1)]
    
    cols = st.columns(n_machines)
    for i, mid in enumerate(machine_ids):
        with cols[i]:
            st.markdown(f"**{mid}**")
            vib = np.random.uniform(0.8, 3.5)
            temp = np.random.uniform(30, 55)
            press = np.random.uniform(95, 115)
            
            # Color based on threshold
            vib_color = "normal" if vib < 2.5 else "inverse"
            temp_color = "normal" if temp < 45 else "inverse"
            press_color = "normal" if press < 110 else "inverse"
            
            st.metric("Vibration", f"{vib:.2f} mm/s", delta=None)
            st.metric("Temperature", f"{temp:.1f} °C")
            st.metric("Pressure", f"{press:.1f} psi")
            st.markdown("---")
    
    # Real-time chart
    chart_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=50, freq='S'),
        'vibration': np.random.normal(2, 0.5, 50).cumsum() / 10 + 1.5,
        'temperature': np.random.normal(40, 2, 50).cumsum() / 20 + 35
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=chart_data['timestamp'], y=chart_data['vibration'], name="Vibration",
                   line=dict(color='#FF4B4B')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=chart_data['timestamp'], y=chart_data['temperature'], name="Temperature",
                   line=dict(color='#FFA15A')),
        secondary_y=True
    )
    fig.update_layout(
        title="Live Sensor Trends (Last 50 seconds)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def settings_page():
    """Admin settings page"""
    st.title("⚙️ System Settings")
    
    st.subheader("Model Configuration")
    with st.form("model_settings"):
        n_estimators = st.slider("Number of Trees", 50, 500, 100, step=10)
        max_depth = st.slider("Max Depth", 5, 50, 20)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        
        submitted = st.form_submit_button("Update Model Parameters")
        if submitted:
            st.success("Settings saved (simulation)")
    
    st.subheader("Alert Thresholds")
    high_risk_threshold = st.slider("High Risk Probability Threshold", 0.5, 0.95, 0.7, 0.05)
    medium_risk_threshold = st.slider("Medium Risk Probability Threshold", 0.2, 0.6, 0.4, 0.05)
    
    st.subheader("Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All Data"):
            for key in ['data', 'predictions', 'maintenance_schedule', 'model', 'scaler', 'model_metrics']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Data cleared!")
            st.rerun()
    with col2:
        if st.button("📊 Generate System Report"):
            report = {
                'model_metrics': st.session_state.model_metrics,
                'data_shape': st.session_state.data.shape if st.session_state.data is not None else None,
                'predictions_count': len(st.session_state.predictions) if st.session_state.predictions is not None else 0,
                'timestamp': datetime.now().isoformat()
            }
            st.json(report)

# ===================================================================
# Main Entry Point
# ===================================================================
def main():
    """Main application entry point"""
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()

