# -*- coding: utf-8 -*-
"""
Nexus Predictive Maintenance
============================
A comprehensive predictive maintenance solution with AWS cloud integration,
admin panel, user authentication, Excel-based failure prediction, and
maintenance scheduling. Built with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import io
import base64
import os
import json
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import tempfile
import calendar
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration & Constants
# =============================================================================
st.set_page_config(
    page_title="Nexus Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Configuration (use Streamlit secrets or environment variables)
AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY", ""))
AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_KEY", os.environ.get("AWS_SECRET_KEY", ""))
AWS_REGION = st.secrets.get("AWS_REGION", os.environ.get("AWS_REGION", "us-east-1"))
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", os.environ.get("S3_BUCKET_NAME", "nexus-predictive-maintenance"))

# Admin credentials (hardcoded as per request)
ADMIN_EMAIL = "kareemeltemsah7@gmail.com"  # corrected typo
ADMIN_PASSWORD = "temsah1"

# Model file names
CLASSIFIER_MODEL_FILE = "failure_classifier.pkl"
REGRESSOR_MODEL_FILE = "rul_regressor.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'page' not in st.session_state:
    st.session_state.page = "Login"
if 'users_db' not in st.session_state:
    st.session_state.users_db = {
        ADMIN_EMAIL: {
            "password": hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest(),
            "role": "admin",
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
    }
if 's3_client' not in st.session_state:
    st.session_state.s3_client = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# =============================================================================
# AWS S3 Helper Functions
# =============================================================================
def initialize_s3_client():
    """Initialize boto3 S3 client with credentials."""
    try:
        if AWS_ACCESS_KEY and AWS_SECRET_KEY:
            client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION
            )
            # Test connection
            client.list_buckets()
            st.session_state.s3_client = client
            return True
        else:
            st.warning("AWS credentials not found. Cloud features will be simulated.")
            return False
    except Exception as e:
        st.error(f"AWS S3 connection failed: {e}")
        return False

def upload_file_to_s3(file_obj, s3_key):
    """Upload a file-like object to S3 bucket."""
    if st.session_state.s3_client is None:
        return False
    try:
        st.session_state.s3_client.upload_fileobj(file_obj, S3_BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        st.error(f"S3 upload error: {e}")
        return False

def download_file_from_s3(s3_key):
    """Download file from S3 and return bytes."""
    if st.session_state.s3_client is None:
        return None
    try:
        response = st.session_state.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        st.error(f"S3 download error: {e}")
        return None

def list_s3_objects(prefix=""):
    """List objects in S3 bucket with given prefix."""
    if st.session_state.s3_client is None:
        return []
    try:
        response = st.session_state.s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    except Exception as e:
        st.error(f"S3 list error: {e}")
        return []

def save_model_to_s3(model, filename):
    """Save a model object to S3 using joblib."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return upload_file_to_s3(buffer, f"models/{filename}")

def load_model_from_s3(filename):
    """Load a model object from S3."""
    data = download_file_from_s3(f"models/{filename}")
    if data:
        return joblib.load(io.BytesIO(data))
    return None

# =============================================================================
# Authentication & User Management
# =============================================================================
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(email, password):
    """Check user credentials against stored database."""
    users = st.session_state.users_db
    if email in users:
        if users[email]["password"] == hash_password(password):
            users[email]["last_login"] = datetime.now().isoformat()
            st.session_state.users_db = users
            return users[email]["role"]
    return None

def add_user(email, password, role="user"):
    """Add a new user to the database."""
    if email in st.session_state.users_db:
        return False, "User already exists."
    st.session_state.users_db[email] = {
        "password": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    return True, "User added successfully."

def delete_user(email):
    """Delete a user (admin only)."""
    if email == ADMIN_EMAIL:
        return False, "Cannot delete admin account."
    if email in st.session_state.users_db:
        del st.session_state.users_db[email]
        return True, "User deleted."
    return False, "User not found."

def get_all_users():
    """Return list of all users (for admin panel)."""
    return [
        {
            "email": email,
            "role": data["role"],
            "created_at": data["created_at"],
            "last_login": data["last_login"]
        }
        for email, data in st.session_state.users_db.items()
    ]

# =============================================================================
# Data Generation & Synthetic Dataset Creation
# =============================================================================
def generate_synthetic_machine_data(n_samples=1000, include_failures=True):
    """
    Generate synthetic sensor data for industrial machines.
    Features: temperature, vibration, pressure, humidity, runtime, load.
    Target: failure_type (None, Overheat, Bearing, Seal, Electrical) and RUL.
    """
    np.random.seed(42)
    
    # Normal operating ranges
    temp_mean, temp_std = 75, 15
    vib_mean, vib_std = 0.5, 0.2
    press_mean, press_std = 100, 10
    hum_mean, hum_std = 45, 10
    runtime_mean, runtime_std = 5000, 2000
    load_mean, load_std = 70, 20
    
    # Generate features
    temperature = np.random.normal(temp_mean, temp_std, n_samples)
    vibration = np.abs(np.random.normal(vib_mean, vib_std, n_samples))
    pressure = np.random.normal(press_mean, press_std, n_samples)
    humidity = np.random.normal(hum_mean, hum_std, n_samples)
    runtime = np.abs(np.random.normal(runtime_mean, runtime_std, n_samples))
    load = np.abs(np.random.normal(load_mean, load_std, n_samples))
    
    # Failure modes (if include_failures)
    failure_type = np.array(['None'] * n_samples, dtype=object)
    rul = np.random.uniform(100, 1000, n_samples)  # Remaining Useful Life in hours
    
    if include_failures:
        # Introduce failure patterns
        # Overheat: high temperature, high vibration
        overheat_idx = np.where((temperature > 100) & (vibration > 0.8))[0]
        failure_type[overheat_idx] = 'Overheat'
        rul[overheat_idx] = np.random.uniform(0, 200, len(overheat_idx))
        
        # Bearing failure: high vibration, high runtime
        bearing_idx = np.where((vibration > 1.0) & (runtime > 7000) & (failure_type == 'None'))[0]
        failure_type[bearing_idx] = 'Bearing'
        rul[bearing_idx] = np.random.uniform(0, 150, len(bearing_idx))
        
        # Seal failure: high pressure, high temperature
        seal_idx = np.where((pressure > 120) & (temperature > 90) & (failure_type == 'None'))[0]
        failure_type[seal_idx] = 'Seal'
        rul[seal_idx] = np.random.uniform(0, 100, len(seal_idx))
        
        # Electrical failure: high load, high humidity
        electrical_idx = np.where((load > 90) & (humidity > 60) & (failure_type == 'None'))[0]
        failure_type[electrical_idx] = 'Electrical'
        rul[electrical_idx] = np.random.uniform(0, 80, len(electrical_idx))
        
        # Adjust RUL for non-failure samples to be higher
        none_idx = failure_type == 'None'
        rul[none_idx] = np.random.uniform(200, 1000, np.sum(none_idx))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n_samples, freq='H'),
        'machine_id': np.random.choice(['M001', 'M002', 'M003', 'M004', 'M005'], n_samples),
        'temperature': np.clip(temperature, 20, 150),
        'vibration': np.clip(vibration, 0, 2.5),
        'pressure': np.clip(pressure, 50, 200),
        'humidity': np.clip(humidity, 10, 90),
        'runtime_hours': np.clip(runtime, 0, 10000),
        'load_percent': np.clip(load, 0, 100),
        'failure_type': failure_type,
        'RUL_hours': np.clip(rul, 0, 1000)
    })
    
    # Add some missing values for realism
    for col in ['temperature', 'vibration', 'pressure']:
        df.loc[df.sample(frac=0.02).index, col] = np.nan
    
    return df

def generate_maintenance_schedule(predictions_df):
    """
    Generate recommended maintenance dates based on predicted RUL and failure probability.
    """
    schedule = []
    today = datetime.now().date()
    
    for idx, row in predictions_df.iterrows():
        rul = row.get('predicted_RUL_hours', 500)
        failure_prob = row.get('failure_probability', 0.0)
        failure_type = row.get('predicted_failure_type', 'None')
        
        # Determine urgency
        if failure_type != 'None' or failure_prob > 0.7:
            urgency = "Critical"
            days_ahead = max(1, int(rul / 24))
        elif failure_prob > 0.3:
            urgency = "High"
            days_ahead = max(3, int(rul / 24))
        elif failure_prob > 0.1:
            urgency = "Medium"
            days_ahead = max(7, int(rul / 24))
        else:
            urgency = "Low"
            days_ahead = max(30, int(rul / 24))
        
        recommended_date = today + timedelta(days=days_ahead)
        schedule.append({
            'machine_id': row.get('machine_id', 'Unknown'),
            'failure_type': failure_type,
            'probability': failure_prob,
            'rul_hours': rul,
            'urgency': urgency,
            'recommended_maintenance': recommended_date.strftime('%Y-%m-%d'),
            'notes': f"Predicted {failure_type} failure with {failure_prob:.1%} probability."
        })
    
    return pd.DataFrame(schedule)

# =============================================================================
# Machine Learning Pipeline
# =============================================================================
def preprocess_data(df, train=True, scaler=None, label_encoder=None):
    """
    Preprocess data for ML: handle missing values, scale features, encode labels.
    """
    feature_cols = ['temperature', 'vibration', 'pressure', 'humidity', 'runtime_hours', 'load_percent']
    
    X = df[feature_cols].copy()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X) if train else imputer.transform(X)
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = scaler.transform(X_imputed)
    
    result = {'X_scaled': X_scaled, 'scaler': scaler}
    
    # For training, also process y
    if train and 'failure_type' in df.columns and 'RUL_hours' in df.columns:
        y_class = df['failure_type'].values
        y_reg = df['RUL_hours'].values
        
        if label_encoder is None:
            label_encoder = LabelEncoder()
            y_class_enc = label_encoder.fit_transform(y_class)
        else:
            y_class_enc = label_encoder.transform(y_class)
        
        result['y_class'] = y_class_enc
        result['y_reg'] = y_reg
        result['label_encoder'] = label_encoder
    
    return result

def train_models(df):
    """
    Train Random Forest classifier and regressor on the provided dataframe.
    """
    preproc = preprocess_data(df, train=True)
    X = preproc['X_scaled']
    y_class = preproc['y_class']
    y_reg = preproc['y_reg']
    scaler = preproc['scaler']
    label_encoder = preproc['label_encoder']
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_class_train)
    class_pred = clf.predict(X_test)
    
    # Train regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_reg_train)
    reg_pred = reg.predict(X_test)
    
    # Metrics
    class_report = classification_report(y_class_test, class_pred, target_names=label_encoder.classes_, output_dict=True)
    reg_mse = mean_squared_error(y_reg_test, reg_pred)
    reg_r2 = r2_score(y_reg_test, reg_pred)
    
    models = {
        'classifier': clf,
        'regressor': reg,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'metrics': {
            'classification_report': class_report,
            'regression_mse': reg_mse,
            'regression_r2': reg_r2
        }
    }
    
    # Save to session state and optionally to S3
    st.session_state['trained_models'] = models
    st.session_state.model_trained = True
    
    # Save to S3 if connected
    if st.session_state.s3_client:
        save_model_to_s3(clf, CLASSIFIER_MODEL_FILE)
        save_model_to_s3(reg, REGRESSOR_MODEL_FILE)
        save_model_to_s3(scaler, SCALER_FILE)
        save_model_to_s3(label_encoder, LABEL_ENCODER_FILE)
    
    return models

def load_or_train_models():
    """
    Attempt to load models from S3, else train on synthetic data.
    """
    if st.session_state.s3_client:
        # Try loading from S3
        clf = load_model_from_s3(CLASSIFIER_MODEL_FILE)
        reg = load_model_from_s3(REGRESSOR_MODEL_FILE)
        scaler = load_model_from_s3(SCALER_FILE)
        le = load_model_from_s3(LABEL_ENCODER_FILE)
        if all([clf, reg, scaler, le]):
            models = {
                'classifier': clf,
                'regressor': reg,
                'scaler': scaler,
                'label_encoder': le,
                'metrics': None  # Could recompute if needed
            }
            st.session_state['trained_models'] = models
            st.session_state.model_trained = True
            return models
    
    # Train on synthetic data
    with st.spinner("Training models on synthetic data..."):
        df_train = generate_synthetic_machine_data(n_samples=2000, include_failures=True)
        models = train_models(df_train)
    return models

def predict_from_dataframe(df, models):
    """
    Use trained models to predict failure type and RUL for new data.
    """
    preproc = preprocess_data(df, train=False, scaler=models['scaler'])
    X_scaled = preproc['X_scaled']
    
    clf = models['classifier']
    reg = models['regressor']
    le = models['label_encoder']
    
    # Predictions
    class_pred_enc = clf.predict(X_scaled)
    class_proba = clf.predict_proba(X_scaled)
    rul_pred = reg.predict(X_scaled)
    
    # Get failure probabilities
    failure_proba = 1 - class_proba[:, le.transform(['None'])[0]]
    
    df_result = df.copy()
    df_result['predicted_failure_type'] = le.inverse_transform(class_pred_enc)
    df_result['failure_probability'] = failure_proba
    df_result['predicted_RUL_hours'] = rul_pred
    
    return df_result

# =============================================================================
# UI Helper Functions
# =============================================================================
def local_css():
    """Inject custom CSS for better styling."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def show_logo():
    """Display Nexus logo."""
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=NEXUS+PM", use_column_width=True)

def download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for a DataFrame (Excel)."""
    if isinstance(object_to_download, pd.DataFrame):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            object_to_download.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{download_filename}">📥 {download_link_text}</a>'
        return href
    return ""

def send_email_notification(recipient, subject, body):
    """Simulate sending email (for demonstration)."""
    # In production, configure SMTP settings
    st.info(f"📧 Email notification would be sent to {recipient}:\nSubject: {subject}")
    return True

# =============================================================================
# Page Components
# =============================================================================
def login_page():
    """Render login page."""
    local_css()
    show_logo()
    st.markdown("<h1 class='main-header'>🔐 Nexus Predictive Maintenance</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if not email or not password:
                    st.error("Please enter email and password.")
                else:
                    role = authenticate_user(email, password)
                    if role:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.is_admin = (role == "admin")
                        st.session_state.page = "Dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        
        st.markdown("---")
        st.info("Demo Admin: kareemeltemsah7@gmail.com / temsah1")
        
        # Register option (for demo)
        with st.expander("Register New Account"):
            with st.form("register_form"):
                new_email = st.text_input("New Email")
                new_pass = st.text_input("New Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Register"):
                    if new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    elif len(new_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        success, msg = add_user(new_email, new_pass, role="user")
                        if success:
                            st.success(msg + " You can now login.")
                        else:
                            st.error(msg)

def dashboard_page():
    """Main dashboard after login."""
    local_css()
    st.markdown(f"<h1 class='main-header'>👋 Welcome, {st.session_state.user_email}</h1>", unsafe_allow_html=True)
    
    # Quick stats cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><h3>🏭 Machines</h3><h2>24</h2><p>Monitored</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h3>⚠️ Alerts</h3><h2>3</h2><p>Critical</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h3>📊 Uptime</h3><h2>98.2%</h2><p>Last 30 days</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><h3>🔧 Next Maintenance</h3><h2>2 days</h2><p>Machine M003</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent activity and quick actions
    col_left, col_right = st.columns([2,1])
    
    with col_left:
        st.subheader("📈 System Health Overview")
        # Sample plot
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        health_score = 100 - np.random.uniform(0, 15, 30)
        fig = px.line(x=dates, y=health_score, title="Overall Equipment Health Trend",
                     labels={'x':'Date', 'y':'Health Score (%)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Recent Predictions")
        if 'last_predictions' in st.session_state:
            st.dataframe(st.session_state.last_predictions.head(5), use_container_width=True)
        else:
            st.info("No predictions yet. Go to Predictive Maintenance to upload data.")
    
    with col_right:
        st.subheader("⚡ Quick Actions")
        if st.button("📤 Upload New Data", use_container_width=True):
            st.session_state.page = "Predictive Maintenance"
            st.rerun()
        if st.button("📊 View Reports", use_container_width=True):
            st.session_state.page = "Reports"
            st.rerun()
        if st.button("🔄 Refresh Models", use_container_width=True):
            with st.spinner("Retraining models..."):
                df = generate_synthetic_machine_data(2000)
                models = train_models(df)
                st.success("Models retrained successfully!")
        
        st.subheader("📅 Upcoming Maintenance")
        maintenance_tasks = pd.DataFrame({
            'Machine': ['M001', 'M003', 'M005'],
            'Due Date': [(datetime.now() + timedelta(days=d)).strftime('%Y-%m-%d') for d in [1, 3, 5]],
            'Type': ['Bearing', 'Overheat', 'Seal']
        })
        st.dataframe(maintenance_tasks, use_container_width=True)

def predictive_maintenance_page():
    """Page for uploading Excel and getting predictions."""
    local_css()
    st.markdown("<h1 class='main-header'>🔮 Predictive Maintenance Analysis</h1>", unsafe_allow_html=True)
    
    # Ensure models are loaded
    if not st.session_state.model_trained:
        with st.spinner("Loading/training models..."):
            models = load_or_train_models()
    else:
        models = st.session_state.get('trained_models')
    
    if models is None:
        st.error("Models could not be loaded. Please contact admin.")
        return
    
    # Upload section
    st.markdown("### 📁 Upload Sensor Data (Excel)")
    st.markdown("Expected columns: timestamp, machine_id, temperature, vibration, pressure, humidity, runtime_hours, load_percent")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_excel(uploaded_file)
            st.success(f"✅ File uploaded: {uploaded_file.name} ({len(df_input)} rows)")
            
            # Show preview
            with st.expander("Preview Uploaded Data"):
                st.dataframe(df_input.head(10))
            
            # Check required columns
            required_cols = ['temperature', 'vibration', 'pressure', 'humidity', 'runtime_hours', 'load_percent']
            missing = [col for col in required_cols if col not in df_input.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                return
            
            # Add timestamp and machine_id if missing
            if 'timestamp' not in df_input.columns:
                df_input['timestamp'] = pd.Timestamp.now()
            if 'machine_id' not in df_input.columns:
                df_input['machine_id'] = 'Uploaded'
            
            # Predict
            with st.spinner("Running predictions..."):
                predictions_df = predict_from_dataframe(df_input, models)
                st.session_state.last_predictions = predictions_df
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                failure_count = (predictions_df['predicted_failure_type'] != 'None').sum()
                st.metric("Machines with Predicted Failure", failure_count)
            with col2:
                avg_rul = predictions_df['predicted_RUL_hours'].mean()
                st.metric("Average Remaining Life (hours)", f"{avg_rul:.1f}")
            with col3:
                high_risk = (predictions_df['failure_probability'] > 0.7).sum()
                st.metric("High Risk (>70%)", high_risk)
            
            # Detailed table
            st.dataframe(predictions_df, use_container_width=True)
            
            # Visualization
            st.subheader("📈 Failure Probability Distribution")
            fig = px.histogram(predictions_df, x='failure_probability', nbins=20,
                              title="Histogram of Failure Probability",
                              labels={'failure_probability':'Failure Probability'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Maintenance Schedule
            st.subheader("🗓️ Recommended Maintenance Schedule")
            schedule_df = generate_maintenance_schedule(predictions_df)
            st.dataframe(schedule_df, use_container_width=True)
            
            # Download options
            st.markdown("### 📥 Export Results")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                # Excel download link
                excel_link = download_link(predictions_df, "nexus_predictions.xlsx", "Download Predictions (Excel)")
                st.markdown(excel_link, unsafe_allow_html=True)
            with col_dl2:
                schedule_link = download_link(schedule_df, "maintenance_schedule.xlsx", "Download Maintenance Schedule")
                st.markdown(schedule_link, unsafe_allow_html=True)
            
            # Upload to S3 option
            if st.session_state.s3_client:
                if st.button("☁️ Save Results to Cloud (S3)"):
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                        schedule_df.to_excel(writer, sheet_name='Schedule', index=False)
                    buffer.seek(0)
                    s3_key = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}_nexus_results.xlsx"
                    if upload_file_to_s3(buffer, s3_key):
                        st.success(f"Results saved to S3: {s3_key}")
            
            # Alert notifications
            if failure_count > 0:
                st.warning(f"⚠️ {failure_count} potential failures detected. Review maintenance schedule.")
                if st.button("📧 Send Alert to Maintenance Team"):
                    # Simulate email
                    send_email_notification(
                        "maintenance@nexus.com",
                        f"Alert: {failure_count} Predicted Failures",
                        f"Please review the latest predictive maintenance report. High risk items need immediate attention."
                    )
                    st.success("Alert notification sent (simulated).")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show sample data download
        st.info("👆 Upload an Excel file with sensor readings to get predictions.")
        st.markdown("### 📋 Sample Data Template")
        sample_df = generate_synthetic_machine_data(n_samples=10, include_failures=False).drop(columns=['failure_type','RUL_hours'])
        sample_link = download_link(sample_df, "nexus_sample_data.xlsx", "Download Sample Template")
        st.markdown(sample_link, unsafe_allow_html=True)
        
        # Option to use demo data
        if st.button("🚀 Try with Demo Data"):
            demo_df = generate_synthetic_machine_data(n_samples=50, include_failures=True)
            demo_df = demo_df.drop(columns=['failure_type','RUL_hours'])  # Simulate user uploaded data without labels
            with st.spinner("Running demo predictions..."):
                predictions_df = predict_from_dataframe(demo_df, models)
                st.session_state.last_predictions = predictions_df
                st.success("Demo data processed! See results below.")
                st.dataframe(predictions_df.head(10))
                st.rerun()

def admin_panel_page():
    """Admin dashboard for user management and system oversight."""
    local_css()
    st.markdown("<h1 class='main-header'>🛡️ Admin Control Panel</h1>", unsafe_allow_html=True)
    
    if not st.session_state.is_admin:
        st.error("Access denied. Admin privileges required.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "📊 System Metrics", "🤖 Model Management", "☁️ Cloud Storage"])
    
    with tab1:
        st.subheader("User Accounts")
        users = get_all_users()
        users_df = pd.DataFrame(users)
        st.dataframe(users_df, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add New User")
            with st.form("admin_add_user"):
                new_email = st.text_input("Email")
                new_pass = st.text_input("Password", type="password")
                role = st.selectbox("Role", ["user", "admin"])
                if st.form_submit_button("Add User"):
                    if new_email and new_pass:
                        success, msg = add_user(new_email, new_pass, role)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        with col2:
            st.subheader("Delete User")
            with st.form("admin_delete_user"):
                del_email = st.selectbox("Select user to delete", 
                                        [u['email'] for u in users if u['email'] != ADMIN_EMAIL])
                if st.form_submit_button("Delete User"):
                    success, msg = delete_user(del_email)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    
    with tab2:
        st.subheader("System Usage Metrics")
        # Placeholder metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions Made", np.random.randint(500, 2000))
        col2.metric("Active Users (24h)", len([u for u in users if u['last_login'] and 
                    (datetime.now() - datetime.fromisoformat(u['last_login'])).days < 1]))
        col3.metric("Model Accuracy", "94.2%")
        
        # Activity log
        st.subheader("Recent Activity")
        log_data = pd.DataFrame({
            'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='H'),
            'User': np.random.choice([u['email'] for u in users], 10),
            'Action': np.random.choice(['Login', 'Upload Data', 'Download Report', 'View Dashboard'], 10)
        })
        st.dataframe(log_data, use_container_width=True)
    
    with tab3:
        st.subheader("Machine Learning Models")
        if st.button("🔄 Retrain Models with Latest Data"):
            with st.spinner("Retraining models..."):
                df_train = generate_synthetic_machine_data(n_samples=3000)
                models = train_models(df_train)
                st.success("Models retrained successfully!")
                st.json(models['metrics'])
        
        if 'trained_models' in st.session_state:
            models = st.session_state.trained_models
            st.write("**Current Model Performance:**")
            if models['metrics']:
                st.json(models['metrics'])
            else:
                st.info("Metrics not available.")
        
        st.subheader("Feature Importance")
        if 'trained_models' in st.session_state:
            clf = st.session_state.trained_models['classifier']
            feature_names = ['temperature', 'vibration', 'pressure', 'humidity', 'runtime_hours', 'load_percent']
            importance = clf.feature_importances_
            fig = px.bar(x=feature_names, y=importance, title="Feature Importance (Classifier)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("AWS S3 Storage Management")
        if st.session_state.s3_client:
            if st.button("🔄 Refresh Bucket List"):
                objects = list_s3_objects()
                if objects:
                    st.write(f"Objects in bucket `{S3_BUCKET_NAME}`:")
                    for obj in objects[:20]:  # show first 20
                        st.text(obj)
                else:
                    st.info("Bucket is empty or inaccessible.")
            
            st.markdown("---")
            st.subheader("Upload File to S3")
            uploaded = st.file_uploader("Choose file", key="admin_s3_upload")
            if uploaded:
                s3_path = st.text_input("S3 Key (path)", f"admin_uploads/{uploaded.name}")
                if st.button("Upload to S3"):
                    if upload_file_to_s3(uploaded, s3_path):
                        st.success(f"Uploaded to s3://{S3_BUCKET_NAME}/{s3_path}")
        else:
            st.warning("AWS S3 not configured. Set credentials in secrets or environment variables.")

def reports_page():
    """Generate historical reports and analytics."""
    local_css()
    st.markdown("<h1 class='main-header'>📊 Reports & Analytics</h1>", unsafe_allow_html=True)
    
    st.subheader("Historical Failure Trends")
    # Generate synthetic historical data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    failures = np.random.poisson(2, 90)
    df_trend = pd.DataFrame({'Date': dates, 'Failures': failures})
    fig = px.line(df_trend, x='Date', y='Failures', title="Daily Failure Count (Last 90 Days)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Maintenance Compliance")
    compliance = np.random.uniform(85, 99, 12)
    months = pd.date_range(end=datetime.now(), periods=12, freq='M').strftime('%b %Y')
    fig2 = px.bar(x=months, y=compliance, title="Monthly Maintenance Compliance (%)",
                 labels={'x':'Month', 'y':'Compliance %'})
    fig2.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target 95%")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Download Reports")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Monthly Report (PDF simulation)"):
            st.info("PDF report generation simulated. In production, would create detailed PDF.")
            # Simulate download link
            st.markdown(download_link(pd.DataFrame({'A':[1,2]}), "monthly_report.xlsx", "Download Sample Report"), unsafe_allow_html=True)
    with col2:
        st.date_input("Select Date Range", value=(datetime.now()-timedelta(days=30), datetime.now()))

# =============================================================================
# Main App Logic
# =============================================================================
def main():
    """Main Streamlit app controller."""
    # Initialize S3 client if credentials present
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        initialize_s3_client()
    
    # Sidebar navigation (only when logged in)
    if st.session_state.logged_in:
        with st.sidebar:
            st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=NEXUS", use_column_width=True)
            st.markdown(f"**User:** {st.session_state.user_email}")
            st.markdown(f"**Role:** {'Admin' if st.session_state.is_admin else 'User'}")
            st.markdown("---")
            
            # Navigation
            page = st.radio("Navigation", 
                           ["Dashboard", "Predictive Maintenance", "Reports"] + 
                           (["Admin Panel"] if st.session_state.is_admin else []),
                           index=0 if st.session_state.page not in ["Predictive Maintenance", "Reports", "Admin Panel"] else
                           ["Dashboard", "Predictive Maintenance", "Reports", "Admin Panel"].index(st.session_state.page))
            st.session_state.page = page
            
            st.markdown("---")
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.user_email = ""
                st.session_state.is_admin = False
                st.session_state.page = "Login"
                st.rerun()
    
    # Render appropriate page
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == "Dashboard":
            dashboard_page()
        elif st.session_state.page == "Predictive Maintenance":
            predictive_maintenance_page()
        elif st.session_state.page == "Reports":
            reports_page()
        elif st.session_state.page == "Admin Panel" and st.session_state.is_admin:
            admin_panel_page()
        else:
            dashboard_page()

if __name__ == "__main__":
    main()
