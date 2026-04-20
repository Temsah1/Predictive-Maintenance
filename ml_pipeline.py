"""
Machine Learning Pipeline
Anomaly Detection + Failure Prediction + RUL Estimation
Production-Grade Industrial AI
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, Tuple, Optional, List
from datetime import datetime

warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier,
    RandomForestRegressor, GradientBoostingClassifier
)
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# ─── Constants ─────────────────────────────────────────────────────────────────

MODELS_DIR = "models"
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")
FAILURE_MODEL_PATH = os.path.join(MODELS_DIR, "failure_model.pkl")
RUL_MODEL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

FEATURE_COLUMNS = [
    "temperature", "vibration", "pressure",
    "energy_consumption", "load_factor", "runtime_hours",
    "temperature_rolling_mean_6", "vibration_rolling_mean_6",
    "pressure_rolling_mean_6", "energy_consumption_rolling_mean_6",
    "temperature_rolling_std_6", "vibration_rolling_std_6",
    "temperature_delta", "vibration_delta", "energy_consumption_delta",
    "temp_vib_product", "efficiency_index", "anomaly_score",
]

os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Feature Preparation ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None,
                     fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """Extract and scale features from engineered dataframe"""
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Fill missing engineered features with basic columns if needed
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            base = col.split("_rolling")[0].split("_delta")[0]
            if base in df.columns:
                df[col] = df[base]
            else:
                df[col] = 0.0

    X = df[FEATURE_COLUMNS].fillna(0).values

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    elif scaler is not None:
        X_scaled = scaler.transform(X)
        return X_scaled, scaler
    else:
        return X, None


# ─── Anomaly Detection Model ───────────────────────────────────────────────────

class AnomalyDetector:
    """Isolation Forest-based anomaly detection"""

    def __init__(self, contamination: float = 0.08, n_estimators: int = 150):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
            max_features=0.8,
        )
        self.scaler = StandardScaler()
        self.threshold = -0.15
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        # Compute threshold from training data
        scores = self.model.decision_function(X_scaled)
        self.threshold = np.percentile(scores, 8)  # Bottom 8% = anomaly
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns -1 for anomaly, 1 for normal"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: more negative = more anomalous"""
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def anomaly_probability(self, X: np.ndarray) -> np.ndarray:
        """Convert scores to 0-1 probability of being anomalous"""
        scores = self.score_samples(X)
        # Normalize: lower score = higher anomaly probability
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros(len(scores))
        normalized = (scores - max_s) / (min_s - max_s + 1e-10)
        return np.clip(normalized, 0, 1)

    def save(self, path: str = ANOMALY_MODEL_PATH):
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "threshold": self.threshold, "is_fitted": self.is_fitted}, path)

    @classmethod
    def load(cls, path: str = ANOMALY_MODEL_PATH) -> "AnomalyDetector":
        obj = cls()
        data = joblib.load(path)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.threshold = data["threshold"]
        obj.is_fitted = data["is_fitted"]
        return obj


# ─── Failure Prediction Model ──────────────────────────────────────────────────

class FailurePredictor:
    """Gradient Boosting classifier for failure prediction"""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_importances = None
        self.is_fitted = False
        self.auc_score = 0.0
        self.feature_names = FEATURE_COLUMNS

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FailurePredictor":
        X_scaled = self.scaler.fit_transform(X)

        # Ensure at least 2 classes exist
        if len(np.unique(y)) < 2:
            # Inject synthetic minority class samples
            n_inject = max(5, int(len(y) * 0.05))
            y = np.concatenate([y, np.ones(n_inject)])
            X_scaled = np.concatenate([X_scaled, X_scaled[:n_inject] * 1.15])

        # Handle class imbalance - use class weights only, not sample_weight that zeros out classes
        self.model.fit(X_scaled, y)

        self.feature_importances = dict(zip(
            self.feature_names, self.model.feature_importances_
        ))

        # Evaluate
        try:
            probs = self.model.predict_proba(X_scaled)[:, 1]
            if len(np.unique(y)) > 1:
                self.auc_score = round(roc_auc_score(y, probs), 4)
        except Exception:
            self.auc_score = 0.0

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        if not self.feature_importances:
            return []
        sorted_feats = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_feats[:n]

    def save(self, path: str = FAILURE_MODEL_PATH):
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "feature_importances": self.feature_importances,
            "auc_score": self.auc_score, "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str = FAILURE_MODEL_PATH) -> "FailurePredictor":
        obj = cls()
        data = joblib.load(path)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_importances = data.get("feature_importances", {})
        obj.auc_score = data.get("auc_score", 0.0)
        obj.is_fitted = data["is_fitted"]
        obj.feature_names = data.get("feature_names", FEATURE_COLUMNS)
        return obj


# ─── RUL Estimation Model ──────────────────────────────────────────────────────

class RULEstimator:
    """Random Forest Regressor for Remaining Useful Life estimation"""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.mae = 0.0
        self.r2 = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULEstimator":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        preds = self.model.predict(X_scaled)
        self.mae = round(mean_absolute_error(y, preds), 2)
        self.r2 = round(r2_score(y, preds), 4)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.clip(preds, 0, None)  # RUL can't be negative

    def save(self, path: str = RUL_MODEL_PATH):
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "is_fitted": self.is_fitted, "mae": self.mae, "r2": self.r2,
        }, path)

    @classmethod
    def load(cls, path: str = RUL_MODEL_PATH) -> "RULEstimator":
        obj = cls()
        data = joblib.load(path)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.is_fitted = data["is_fitted"]
        obj.mae = data.get("mae", 0.0)
        obj.r2 = data.get("r2", 0.0)
        return obj


# ─── Master ML Pipeline ────────────────────────────────────────────────────────

class IndustrialMLPipeline:
    """
    Orchestrates: Anomaly Detection + Failure Prediction + RUL Estimation
    Handles training, evaluation, persistence, and inference.
    """

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.failure_predictor = FailurePredictor()
        self.rul_estimator = RULEstimator()
        self.is_trained = False
        self.training_time = None
        self.n_samples_trained = 0
        self.train_metrics = {}

    def train(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """Full training pipeline on historical data"""
        from data_generator import engineer_features

        if verbose:
            print("🔧 Engineering features...")

        df_eng = engineer_features(df)
        X_raw = df_eng[FEATURE_COLUMNS].fillna(0).values
        y_failure = df_eng["failure_label"].values.astype(int) if "failure_label" in df_eng.columns else np.zeros(len(df_eng))
        y_rul = df_eng["rul_hours"].values if "rul_hours" in df_eng.columns else np.ones(len(df_eng)) * 1000

        # ── Train/Test Split
        if verbose:
            print("📊 Splitting train/test sets...")

        X_train, X_test, y_fail_train, y_fail_test, y_rul_train, y_rul_test = train_test_split(
            X_raw, y_failure, y_rul, test_size=0.2, random_state=42
        )

        # ── Anomaly Detector (unsupervised, trained on all data)
        if verbose:
            print("🔍 Training Anomaly Detector (Isolation Forest)...")
        self.anomaly_detector.fit(X_train)

        # ── Failure Predictor
        if verbose:
            print("⚠️  Training Failure Predictor (Gradient Boosting)...")
        self.failure_predictor.fit(X_train, y_fail_train)

        # ── RUL Estimator
        if verbose:
            print("⏱️  Training RUL Estimator (Random Forest Regressor)...")
        self.rul_estimator.fit(X_train, y_rul_train)

        # ── Evaluate
        if verbose:
            print("📈 Evaluating models...")

        # Failure predictor eval
        fail_probs_test = self.failure_predictor.predict_proba(X_test)
        fail_preds_test = (fail_probs_test >= 0.5).astype(int)

        # RUL eval
        rul_preds_test = self.rul_estimator.predict(X_test)
        rul_mae = round(mean_absolute_error(y_rul_test, rul_preds_test), 2)
        rul_r2 = round(r2_score(y_rul_test, rul_preds_test), 4)

        # Anomaly eval
        anomaly_scores = self.anomaly_detector.score_samples(X_test)
        anomaly_rate = (self.anomaly_detector.predict(X_test) == -1).mean()

        self.train_metrics = {
            "failure_auc": self.failure_predictor.auc_score,
            "failure_samples": len(y_fail_train),
            "rul_mae_hours": rul_mae,
            "rul_r2": rul_r2,
            "anomaly_rate_pct": round(anomaly_rate * 100, 1),
            "total_samples": len(X_raw),
            "feature_count": X_raw.shape[1],
            "top_failure_features": self.failure_predictor.top_features(5),
        }

        self.is_trained = True
        self.training_time = datetime.now().isoformat()
        self.n_samples_trained = len(X_raw)

        # Save all models
        self.save_all()

        if verbose:
            print("✅ Training complete!")
            print(f"   Failure AUC: {self.train_metrics['failure_auc']}")
            print(f"   RUL MAE: {rul_mae} hours | R²: {rul_r2}")
            print(f"   Anomaly Rate: {self.train_metrics['anomaly_rate_pct']}%")

        return self.train_metrics

    def predict_machine(self, row: Dict) -> Dict:
        """Run all three models on a single machine reading"""
        from data_generator import engineer_features

        df_single = pd.DataFrame([row])
        df_eng = engineer_features(df_single)
        X = df_eng[FEATURE_COLUMNS].fillna(0).values

        result = {
            "machine_id": row.get("machine_id", "UNKNOWN"),
            "machine_type": row.get("machine_type", "Unknown"),
        }

        if self.anomaly_detector.is_fitted:
            anom_prob = self.anomaly_detector.anomaly_probability(X)[0]
            anom_label = self.anomaly_detector.predict(X)[0]
            result["anomaly_probability"] = round(anom_prob * 100, 1)
            result["is_anomaly"] = anom_label == -1

        if self.failure_predictor.is_fitted:
            fail_prob = self.failure_predictor.predict_proba(X)[0]
            result["failure_probability"] = round(fail_prob * 100, 1)
            result["failure_predicted"] = fail_prob >= 0.5
            result["top_failure_features"] = self.failure_predictor.top_features(3)

        if self.rul_estimator.is_fitted:
            rul = self.rul_estimator.predict(X)[0]
            result["rul_hours"] = round(max(0, rul), 1)
            result["rul_days"] = round(max(0, rul) / 24, 1)

        return result

    def predict_fleet(self, readings_df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on entire fleet"""
        from data_generator import engineer_features

        df_eng = engineer_features(readings_df.copy())
        X = df_eng[FEATURE_COLUMNS].fillna(0).values

        results = readings_df.copy()

        if self.anomaly_detector.is_fitted:
            results["ml_anomaly_prob"] = np.round(
                self.anomaly_detector.anomaly_probability(X) * 100, 1
            )
            results["ml_is_anomaly"] = self.anomaly_detector.predict(X) == -1

        if self.failure_predictor.is_fitted:
            results["ml_failure_prob"] = np.round(
                self.failure_predictor.predict_proba(X) * 100, 1
            )

        if self.rul_estimator.is_fitted:
            results["ml_rul_hours"] = np.round(
                np.clip(self.rul_estimator.predict(X), 0, None), 1
            )

        return results

    def generate_insight_text(self, machine_row: pd.Series) -> str:
        """Generate human-readable insight for a machine"""
        mid = machine_row.get("machine_id", "Machine")
        fail_prob = machine_row.get("ml_failure_prob", machine_row.get("failure_probability", 0))
        rul = machine_row.get("ml_rul_hours", machine_row.get("rul_hours", 0))
        state = machine_row.get("state", "unknown")

        # Determine primary cause
        causes = []
        vib = machine_row.get("vibration", 0)
        temp = machine_row.get("temperature", 0)
        energy = machine_row.get("energy_consumption", 0)

        if self.failure_predictor.is_fitted and self.failure_predictor.feature_importances:
            top_feat = max(self.failure_predictor.feature_importances,
                           key=self.failure_predictor.feature_importances.get)
            feat_display = top_feat.replace("_rolling_mean_6", " trend").replace("_delta", " change")
            causes.append(feat_display)

        if state == "critical":
            severity = "critical failure"
        elif state == "warning":
            severity = "early-stage degradation"
        elif state == "degrading":
            severity = "gradual performance decline"
        else:
            severity = "normal operation"

        cause_str = " and ".join(causes) if causes else "abnormal sensor patterns"
        rul_str = f"{rul:.0f} hours" if rul > 0 else "immediate action required"

        return (
            f"{mid} shows {severity} with {fail_prob:.0f}% failure probability "
            f"within {rul_str} due to {cause_str}."
        )

    def save_all(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.anomaly_detector.save(ANOMALY_MODEL_PATH)
        self.failure_predictor.save(FAILURE_MODEL_PATH)
        self.rul_estimator.save(RUL_MODEL_PATH)
        joblib.dump({
            "is_trained": self.is_trained,
            "training_time": self.training_time,
            "n_samples_trained": self.n_samples_trained,
            "train_metrics": self.train_metrics,
        }, METADATA_PATH)

    def load_all(self) -> bool:
        """Load all models. Returns True if successful."""
        try:
            self.anomaly_detector = AnomalyDetector.load(ANOMALY_MODEL_PATH)
            self.failure_predictor = FailurePredictor.load(FAILURE_MODEL_PATH)
            self.rul_estimator = RULEstimator.load(RUL_MODEL_PATH)

            if os.path.exists(METADATA_PATH):
                meta = joblib.load(METADATA_PATH)
                self.is_trained = meta.get("is_trained", True)
                self.training_time = meta.get("training_time")
                self.n_samples_trained = meta.get("n_samples_trained", 0)
                self.train_metrics = meta.get("train_metrics", {})

            return True
        except Exception as e:
            print(f"⚠️  Could not load models: {e}")
            return False

    def models_exist(self) -> bool:
        return all(os.path.exists(p) for p in [
            ANOMALY_MODEL_PATH, FAILURE_MODEL_PATH, RUL_MODEL_PATH
        ])


# ─── Quick Train Utility ───────────────────────────────────────────────────────

def train_or_load_pipeline(n_hours: int = 720) -> IndustrialMLPipeline:
    """Train pipeline or load from disk if already trained"""
    pipeline = IndustrialMLPipeline()

    if pipeline.models_exist():
        print("📂 Loading existing models from disk...")
        success = pipeline.load_all()
        if success:
            return pipeline

    print("🚀 No existing models found. Training from scratch...")
    from data_generator import SensorDataGenerator

    gen = SensorDataGenerator(n_machines=12)
    hist_df = gen.generate_historical_data(n_hours=n_hours, interval_minutes=30)
    metrics = pipeline.train(hist_df, verbose=True)
    return pipeline
