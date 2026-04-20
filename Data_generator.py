"""
Industrial Sensor Data Simulation Engine
Simulates real-world equipment from Oil & Gas, Power Plants, Manufacturing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import hashlib

# ─── Machine Configurations ────────────────────────────────────────────────────

MACHINE_CONFIGS = {
    "Compressor":     {"temp_base": 85,  "vib_base": 2.1, "pressure_base": 145, "energy_base": 320, "load_base": 0.75},
    "Pump":           {"temp_base": 65,  "vib_base": 1.4, "pressure_base": 85,  "energy_base": 180, "load_base": 0.68},
    "Turbine":        {"temp_base": 420, "vib_base": 3.2, "pressure_base": 220, "energy_base": 850, "load_base": 0.82},
    "Motor":          {"temp_base": 75,  "vib_base": 1.8, "pressure_base": 15,  "energy_base": 240, "load_base": 0.70},
    "Heat Exchanger": {"temp_base": 160, "vib_base": 0.8, "pressure_base": 55,  "energy_base": 95,  "load_base": 0.60},
    "Generator":      {"temp_base": 110, "vib_base": 2.5, "pressure_base": 30,  "energy_base": 680, "load_base": 0.85},
    "Fan":            {"temp_base": 45,  "vib_base": 1.1, "pressure_base": 8,   "energy_base": 75,  "load_base": 0.55},
    "Conveyor":       {"temp_base": 55,  "vib_base": 2.0, "pressure_base": 12,  "energy_base": 130, "load_base": 0.65},
}

MACHINE_NAMES = [
    "GAS-COMP-01", "PUMP-A-02", "TURB-GEN-03", "MOTOR-D-04",
    "HEX-UNIT-05", "GEN-MAIN-06", "FAN-COOL-07", "CONV-LINE-08",
    "GAS-COMP-09", "PUMP-B-10", "MOTOR-E-11", "HEX-UNIT-12"
]

MACHINE_TYPES = [
    "Compressor", "Pump", "Turbine", "Motor",
    "Heat Exchanger", "Generator", "Fan", "Conveyor",
    "Compressor", "Pump", "Motor", "Heat Exchanger"
]

# ─── Degradation State Management ──────────────────────────────────────────────

class MachineState:
    """Tracks the degradation state of a single machine"""

    STATES = ["healthy", "degrading", "warning", "critical", "failure"]

    def __init__(self, machine_id: str, machine_type: str, seed: int = None):
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.config = MACHINE_CONFIGS.get(machine_type, MACHINE_CONFIGS["Motor"])

        rng_seed = seed if seed is not None else int(hashlib.md5(machine_id.encode()).hexdigest(), 16) % 10000
        self.rng = np.random.RandomState(rng_seed)

        # State variables
        self.runtime_hours = self.rng.uniform(200, 8000)
        self.degradation_level = self.rng.uniform(0.0, 0.35)
        self.state = self._compute_state()
        self.failure_imminent = self.degradation_level > 0.75
        self.last_maintenance = datetime.now() - timedelta(hours=self.runtime_hours)

        # Drift rates (how fast each sensor degrades per cycle)
        self.drift_rates = {
            "temperature":  self.rng.uniform(0.001, 0.008),
            "vibration":    self.rng.uniform(0.002, 0.012),
            "pressure":     self.rng.uniform(0.0005, 0.004),
            "energy":       self.rng.uniform(0.001, 0.007),
            "load":         self.rng.uniform(0.0003, 0.002),
        }

    def _compute_state(self) -> str:
        if self.degradation_level < 0.25:
            return "healthy"
        elif self.degradation_level < 0.50:
            return "degrading"
        elif self.degradation_level < 0.70:
            return "warning"
        elif self.degradation_level < 0.90:
            return "critical"
        else:
            return "failure"

    def advance(self, steps: int = 1):
        """Advance degradation over time"""
        for _ in range(steps):
            increment = self.rng.uniform(0.0001, 0.0008)
            if self.state in ["warning", "critical"]:
                increment *= 2.5
            self.degradation_level = min(1.0, self.degradation_level + increment)
            self.runtime_hours += 0.5
            self.state = self._compute_state()

    def get_health_score(self) -> float:
        return round((1.0 - self.degradation_level) * 100, 1)

    def get_rul_hours(self) -> float:
        """Remaining Useful Life in hours"""
        if self.degradation_level >= 1.0:
            return 0.0
        remaining_pct = (1.0 - self.degradation_level) / 1.0
        max_life = self.rng.uniform(5000, 15000)
        return round(remaining_pct * max_life * (1 - self.degradation_level * 0.3), 1)

    def get_failure_probability(self) -> float:
        base = self.degradation_level ** 1.5
        noise = self.rng.uniform(-0.03, 0.03)
        return round(min(1.0, max(0.0, base + noise)) * 100, 1)


# ─── Sensor Reading Generator ──────────────────────────────────────────────────

class SensorDataGenerator:
    """Generates realistic multi-sensor data for industrial machines"""

    def __init__(self, n_machines: int = 12):
        self.n_machines = min(n_machines, len(MACHINE_NAMES))
        self.machines: Dict[str, MachineState] = {}

        for i in range(self.n_machines):
            mid = MACHINE_NAMES[i]
            mtype = MACHINE_TYPES[i]
            self.machines[mid] = MachineState(mid, mtype, seed=i * 42)

    def _add_sensor_noise(self, value: float, noise_pct: float, rng: np.random.RandomState) -> float:
        noise = rng.normal(0, abs(value) * noise_pct)
        return round(value + noise, 3)

    def _apply_degradation(self, base: float, degradation: float, factor: float = 1.0) -> float:
        """Increase reading as degradation progresses"""
        return base * (1 + degradation * factor)

    def generate_single_reading(self, machine_id: str) -> Dict:
        """Generate one sensor reading snapshot for a machine"""
        ms = self.machines[machine_id]
        cfg = ms.config
        rng = ms.rng
        deg = ms.degradation_level

        # Temperature: rises significantly during degradation
        temp = self._apply_degradation(cfg["temp_base"], deg, factor=0.45)
        temp = self._add_sensor_noise(temp, 0.015, rng)

        # Vibration: most sensitive degradation indicator
        vib = self._apply_degradation(cfg["vib_base"], deg, factor=1.2)
        vib = max(0.1, self._add_sensor_noise(vib, 0.08, rng))

        # Pressure: can drop or spike depending on fault type
        if deg > 0.6:
            pressure_drift = -0.15 * deg  # pressure drops in late degradation
        else:
            pressure_drift = 0.08 * deg
        pressure = cfg["pressure_base"] * (1 + pressure_drift)
        pressure = self._add_sensor_noise(pressure, 0.02, rng)

        # Energy consumption: rises as efficiency drops
        energy = self._apply_degradation(cfg["energy_base"], deg, factor=0.30)
        energy = self._add_sensor_noise(energy, 0.025, rng)

        # Load factor: varies with operating conditions
        load = cfg["load_base"] + rng.uniform(-0.08, 0.08) + deg * 0.05
        load = round(min(1.0, max(0.1, load)), 3)

        # RPM (derived from machine type)
        base_rpm = 1450 if ms.machine_type in ["Motor", "Pump"] else (3000 if ms.machine_type == "Turbine" else 1800)
        rpm = base_rpm * (1 - deg * 0.08) + rng.normal(0, 15)
        rpm = max(0, round(rpm, 0))

        # Oil quality (0-100, degrades over time)
        oil_quality = max(0, 100 - deg * 85 - rng.uniform(0, 8))

        # Advance degradation slightly each reading
        ms.advance(1)

        return {
            "machine_id": machine_id,
            "machine_type": ms.machine_type,
            "timestamp": datetime.now().isoformat(),
            "temperature": round(temp, 2),
            "vibration": round(vib, 3),
            "pressure": round(pressure, 2),
            "energy_consumption": round(energy, 2),
            "load_factor": load,
            "rpm": int(rpm),
            "oil_quality": round(oil_quality, 1),
            "runtime_hours": round(ms.runtime_hours, 1),
            "health_score": ms.get_health_score(),
            "degradation_level": round(deg, 4),
            "state": ms.state,
            "failure_probability": ms.get_failure_probability(),
            "rul_hours": ms.get_rul_hours(),
        }

    def generate_all_readings(self) -> pd.DataFrame:
        """Generate one snapshot reading for all machines"""
        readings = [self.generate_single_reading(mid) for mid in self.machines]
        return pd.DataFrame(readings)

    def generate_historical_data(self, n_hours: int = 720, interval_minutes: int = 30) -> pd.DataFrame:
        """Generate historical time-series data for ML training"""
        records = []
        n_points = (n_hours * 60) // interval_minutes

        # Reset degradation states for clean history
        history_machines = {}
        for i in range(self.n_machines):
            mid = MACHINE_NAMES[i]
            mtype = MACHINE_TYPES[i]
            ms = MachineState(mid, mtype, seed=i * 7 + 100)
            ms.degradation_level = 0.0  # start fresh
            ms.runtime_hours = 0.0
            history_machines[mid] = ms

        start_time = datetime.now() - timedelta(hours=n_hours)

        for step in range(n_points):
            ts = start_time + timedelta(minutes=step * interval_minutes)

            for mid, ms in history_machines.items():
                cfg = ms.config
                rng = ms.rng
                deg = ms.degradation_level

                temp = cfg["temp_base"] * (1 + deg * 0.45) + rng.normal(0, 2.5)
                vib = max(0.1, cfg["vib_base"] * (1 + deg * 1.2) + rng.normal(0, 0.15))
                pressure = cfg["pressure_base"] * (1 + 0.08 * deg - 0.15 * max(0, deg - 0.6)) + rng.normal(0, 1.5)
                energy = cfg["energy_base"] * (1 + deg * 0.30) + rng.normal(0, 5)
                load = cfg["load_base"] + rng.uniform(-0.08, 0.08) + deg * 0.05
                load = min(1.0, max(0.1, load))

                # Label: failure within next 48 hours
                future_deg = min(1.0, deg + 48 * 0.0005)
                will_fail = 1 if future_deg > 0.85 else 0

                records.append({
                    "machine_id": mid,
                    "machine_type": ms.machine_type,
                    "timestamp": ts.isoformat(),
                    "temperature": round(temp, 2),
                    "vibration": round(vib, 3),
                    "pressure": round(pressure, 2),
                    "energy_consumption": round(energy, 2),
                    "load_factor": round(load, 3),
                    "runtime_hours": round(ms.runtime_hours, 1),
                    "degradation_level": round(deg, 4),
                    "failure_label": will_fail,
                    "rul_hours": ms.get_rul_hours(),
                    "state": ms.state,
                })

                ms.advance(2)

        return pd.DataFrame(records)

    def get_fleet_summary(self) -> Dict:
        """Get high-level fleet health summary"""
        readings = self.generate_all_readings()
        healthy = len(readings[readings["state"] == "healthy"])
        degrading = len(readings[readings["state"] == "degrading"])
        warning = len(readings[readings["state"] == "warning"])
        critical = len(readings[readings["state"].isin(["critical", "failure"])])

        overall_health = readings["health_score"].mean()

        return {
            "total_machines": self.n_machines,
            "healthy": healthy,
            "degrading": degrading,
            "warning": warning,
            "critical": critical,
            "overall_health": round(overall_health, 1),
            "readings": readings,
        }


# ─── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics and derived features for ML"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"])

    numeric_cols = ["temperature", "vibration", "pressure", "energy_consumption", "load_factor"]

    for col in numeric_cols:
        df[f"{col}_rolling_mean_6"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(6, min_periods=1).mean())
        )
        df[f"{col}_rolling_std_6"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0))
        )
        df[f"{col}_delta"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.diff().fillna(0))
        )

    # Cross-sensor features
    df["temp_vib_product"] = df["temperature"] * df["vibration"]
    df["efficiency_index"] = df["load_factor"] / (df["energy_consumption"] / df["energy_consumption"].mean() + 0.001)
    df["anomaly_score"] = (
        (df["temperature"] - df["temperature"].mean()) / (df["temperature"].std() + 0.001)
    ).abs()

    return df


# ─── Alert Generator ───────────────────────────────────────────────────────────

def generate_alerts(readings: pd.DataFrame) -> List[Dict]:
    """Generate structured alerts from current sensor readings"""
    alerts = []

    for _, row in readings.iterrows():
        level = None
        messages = []

        if row["state"] in ["critical", "failure"]:
            level = "critical"
            messages.append(f"CRITICAL: {row['machine_id']} requires immediate inspection")
        elif row["state"] == "warning":
            level = "warning"
            messages.append(f"WARNING: {row['machine_id']} showing degradation patterns")
        elif row["state"] == "degrading":
            level = "info"
            messages.append(f"INFO: {row['machine_id']} early degradation detected")

        if row.get("vibration", 0) > MACHINE_CONFIGS.get(row["machine_type"], {}).get("vib_base", 2) * 1.8:
            messages.append("Abnormal vibration levels")
        if row.get("temperature", 0) > MACHINE_CONFIGS.get(row["machine_type"], {}).get("temp_base", 80) * 1.25:
            messages.append("High temperature detected")
        if row.get("failure_probability", 0) > 75:
            messages.append(f"Failure probability: {row['failure_probability']}%")

        if level:
            alerts.append({
                "machine_id": row["machine_id"],
                "machine_type": row["machine_type"],
                "level": level,
                "messages": messages,
                "rul_hours": row.get("rul_hours", 0),
                "failure_prob": row.get("failure_probability", 0),
                "health_score": row.get("health_score", 0),
                "timestamp": row.get("timestamp", datetime.now().isoformat()),
            })

    return sorted(alerts, key=lambda x: (x["level"] == "critical", x["failure_prob"]), reverse=True)
