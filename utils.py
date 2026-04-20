"""
Utility Functions
Industrial Predictive Maintenance Platform
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib


# ─── Color & Status Helpers ────────────────────────────────────────────────────

STATE_COLORS = {
    "healthy":   "#00D4AA",
    "degrading": "#FFB800",
    "warning":   "#FF7043",
    "critical":  "#FF1744",
    "failure":   "#AA00FF",
    "unknown":   "#607D8B",
}

STATE_ICONS = {
    "healthy":   "✅",
    "degrading": "📉",
    "warning":   "⚠️",
    "critical":  "🚨",
    "failure":   "💀",
    "unknown":   "❓",
}

ALERT_LEVEL_COLORS = {
    "critical": "#FF1744",
    "warning":  "#FF7043",
    "info":     "#2196F3",
    "ok":       "#00D4AA",
}


def get_state_color(state: str) -> str:
    return STATE_COLORS.get(state.lower(), STATE_COLORS["unknown"])


def get_state_icon(state: str) -> str:
    return STATE_ICONS.get(state.lower(), STATE_ICONS["unknown"])


def health_to_color(score: float) -> str:
    """Convert 0-100 health score to a hex color"""
    if score >= 80:
        return "#00D4AA"
    elif score >= 60:
        return "#FFB800"
    elif score >= 40:
        return "#FF7043"
    else:
        return "#FF1744"


def failure_prob_to_severity(prob: float) -> str:
    """Convert failure probability % to severity label"""
    if prob >= 75:
        return "critical"
    elif prob >= 50:
        return "warning"
    elif prob >= 25:
        return "degrading"
    else:
        return "healthy"


# ─── Formatting Helpers ────────────────────────────────────────────────────────

def format_rul(hours: float) -> str:
    """Human-readable RUL"""
    if hours <= 0:
        return "⚠️ OVERDUE"
    elif hours < 24:
        return f"{hours:.0f}h"
    elif hours < 24 * 7:
        days = hours / 24
        return f"{days:.1f}d"
    else:
        days = hours / 24
        weeks = days / 7
        return f"{weeks:.1f}w"


def format_hours(hours: float) -> str:
    if hours < 24:
        return f"{hours:.1f}h"
    elif hours < 24 * 30:
        return f"{hours / 24:.1f} days"
    else:
        return f"{hours / 24 / 30:.1f} months"


def format_sensor_value(key: str, value: float) -> str:
    units = {
        "temperature": "°C",
        "vibration": "mm/s",
        "pressure": "bar",
        "energy_consumption": "kW",
        "load_factor": "",
        "rpm": "RPM",
        "oil_quality": "%",
        "runtime_hours": "h",
    }
    unit = units.get(key, "")
    if key == "load_factor":
        return f"{value * 100:.1f}%"
    elif key in ["rpm", "runtime_hours"]:
        return f"{value:.0f} {unit}"
    elif key == "oil_quality":
        return f"{value:.1f}{unit}"
    else:
        return f"{value:.2f} {unit}"


def sensor_display_name(key: str) -> str:
    names = {
        "temperature": "🌡️ Temperature",
        "vibration": "📳 Vibration",
        "pressure": "🔵 Pressure",
        "energy_consumption": "⚡ Energy",
        "load_factor": "⚙️ Load Factor",
        "rpm": "🔄 RPM",
        "oil_quality": "🛢️ Oil Quality",
        "runtime_hours": "⏱️ Runtime",
    }
    return names.get(key, key.replace("_", " ").title())


# ─── Data Processing Helpers ───────────────────────────────────────────────────

def compute_fleet_kpis(readings_df: pd.DataFrame) -> Dict:
    """Compute high-level KPIs for the fleet"""
    total = len(readings_df)
    if total == 0:
        return {}

    state_counts = readings_df["state"].value_counts().to_dict()

    healthy_n = state_counts.get("healthy", 0)
    degrading_n = state_counts.get("degrading", 0)
    warning_n = state_counts.get("warning", 0)
    critical_n = state_counts.get("critical", 0) + state_counts.get("failure", 0)

    overall_health = readings_df["health_score"].mean() if "health_score" in readings_df.columns else 0

    # Machines at risk (failure prob > 50%)
    if "ml_failure_prob" in readings_df.columns:
        at_risk = (readings_df["ml_failure_prob"] > 50).sum()
        avg_fail_prob = readings_df["ml_failure_prob"].mean()
    elif "failure_probability" in readings_df.columns:
        at_risk = (readings_df["failure_probability"] > 50).sum()
        avg_fail_prob = readings_df["failure_probability"].mean()
    else:
        at_risk = 0
        avg_fail_prob = 0

    # Min RUL
    rul_col = "ml_rul_hours" if "ml_rul_hours" in readings_df.columns else "rul_hours"
    min_rul = readings_df[rul_col].min() if rul_col in readings_df.columns else 0

    return {
        "total_machines": total,
        "healthy_count": healthy_n,
        "degrading_count": degrading_n,
        "warning_count": warning_n,
        "critical_count": critical_n,
        "overall_health": round(overall_health, 1),
        "at_risk_count": int(at_risk),
        "avg_failure_prob": round(avg_fail_prob, 1),
        "min_rul_hours": round(min_rul, 1),
        "availability_pct": round((healthy_n + degrading_n) / total * 100, 1),
    }


def get_machine_trend(machine_id: str, historical_df: pd.DataFrame,
                      col: str = "health_score", last_n: int = 20) -> pd.DataFrame:
    """Get trend data for a specific machine"""
    df_machine = historical_df[historical_df["machine_id"] == machine_id].copy()
    if col not in df_machine.columns:
        return pd.DataFrame()
    df_machine["timestamp"] = pd.to_datetime(df_machine["timestamp"])
    df_machine = df_machine.sort_values("timestamp").tail(last_n)
    return df_machine[["timestamp", col]]


def compute_sensor_thresholds(readings_df: pd.DataFrame) -> Dict:
    """Compute dynamic warning/critical thresholds per machine type"""
    thresholds = {}
    for mtype in readings_df["machine_type"].unique():
        subset = readings_df[readings_df["machine_type"] == mtype]
        thresholds[mtype] = {}
        for sensor in ["temperature", "vibration", "pressure", "energy_consumption"]:
            if sensor in subset.columns:
                mean = subset[sensor].mean()
                std = subset[sensor].std()
                thresholds[mtype][sensor] = {
                    "warning":  round(mean + 1.5 * std, 2),
                    "critical": round(mean + 2.5 * std, 2),
                    "mean":     round(mean, 2),
                }
    return thresholds


def classify_sensor_status(value: float, thresholds: Dict) -> str:
    if not thresholds:
        return "healthy"
    if value >= thresholds.get("critical", float("inf")):
        return "critical"
    elif value >= thresholds.get("warning", float("inf")):
        return "warning"
    else:
        return "healthy"


# ─── Synthetic Timeseries ──────────────────────────────────────────────────────

def generate_sparkline_data(base: float, n_points: int = 20, trend: float = 0.0,
                             noise: float = 0.05, seed: int = 42) -> List[float]:
    """Generate mini sparkline timeseries"""
    rng = np.random.RandomState(seed)
    values = [base]
    for i in range(1, n_points):
        drift = trend * i / n_points
        noise_val = rng.normal(0, abs(base) * noise)
        values.append(base * (1 + drift) + noise_val)
    return [round(v, 3) for v in values]


# ─── Maintenance Recommendations ──────────────────────────────────────────────

MAINTENANCE_RECOMMENDATIONS = {
    "critical": [
        "🔴 Immediate shutdown and inspection required",
        "🔴 Replace bearings and check shaft alignment",
        "🔴 Conduct full vibration analysis",
        "🔴 Check for lubrication failure",
        "🔴 Inspect seals and gaskets for leaks",
    ],
    "warning": [
        "🟡 Schedule maintenance within 48 hours",
        "🟡 Check and replenish lubricant levels",
        "🟡 Inspect and clean cooling systems",
        "🟡 Verify sensor calibration",
        "🟡 Monitor closely for further degradation",
    ],
    "degrading": [
        "🟠 Plan preventive maintenance within 1 week",
        "🟠 Run vibration diagnostics",
        "🟠 Check alignment and balance",
        "🟠 Inspect filters and strainers",
        "🟠 Review operating conditions vs. design limits",
    ],
    "healthy": [
        "✅ Continue routine monitoring",
        "✅ Maintain scheduled PM intervals",
        "✅ Keep lubrication records up to date",
    ],
}


def get_recommendations(state: str, machine_type: str = "") -> List[str]:
    recs = MAINTENANCE_RECOMMENDATIONS.get(state, MAINTENANCE_RECOMMENDATIONS["healthy"])
    return recs[:3]


# ─── Hash Utilities ────────────────────────────────────────────────────────────

def machine_seed(machine_id: str) -> int:
    return int(hashlib.md5(machine_id.encode()).hexdigest(), 16) % 100000


# ─── Time Utilities ───────────────────────────────────────────────────────────

def time_since(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s ago"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m ago"
        else:
            return f"{total_seconds // 3600}h ago"
    except Exception:
        return "just now"
