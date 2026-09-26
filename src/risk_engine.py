
import numpy as np
from enum import Enum
from dataclasses import dataclass
import time
import math

class ThreatLevel(Enum):
    LEVEL_1_BASELINE = 1
    LEVEL_2_GUARDED = 2
    LEVEL_3_WARNING = 3
    LEVEL_4_CRITICAL = 4

@dataclass
class SensorTelemetry:
    timestamp: float
    is_safe_zone: bool
    hour_of_day: int
    ble_device_count: int
    accel_magnitude: float
    accel_jerk: float
    acoustic_distress_prob: float = 0.0

class DynamicRiskEngine:
    def __init__(self):
        self.current_level = ThreatLevel.LEVEL_1_BASELINE
        self.risk_ema = 0.0
        self.alpha = 0.4  # Smoothing factor
        self.verification_timer_active = False

    def _on_enter_level_1(self):
        print("[HW STATE] Sensing Hub uDSP active (<1mW). NPU & CPU power-gated.")

    def _on_enter_level_2(self):
        print("[HW STATE] LPASS Audio Subsystem activated. Hardware VAD enabled. Hexagon NPU in retention.")

    def _on_enter_level_3(self):
        print("[HW STATE] Hardware interrupt fired! Hexagon NPU burst awake (<15ms). Verification countdown initiated.")

    def _on_enter_level_4(self):
        print("[HW STATE] CRITICAL THREAT: TrustZone lock down engaged. BLE mesh beaconing activated.")

    def compute_risk(self, telemetry: SensorTelemetry):
        # 1. C_context: Time and Location
        c_context = 0.0
        if not telemetry.is_safe_zone:
            c_context += 40
        if 23 <= telemetry.hour_of_day or telemetry.hour_of_day <= 5:
            c_context += 60
        c_context = min(100, c_context)

        # 2. I_isolation: Exponential BLE decay (tau=2)
        # Penalty increases as ble_device_count approaches 0
        i_isolation = 100 * math.exp(-telemetry.ble_device_count / 2.0)

        # 3. K_kinematics: Impact and struggle
        k_kinematics = 0.0
        if telemetry.accel_magnitude > 25:
            k_kinematics += 50
        if telemetry.accel_jerk > 40:
            k_kinematics += 50
        k_kinematics = min(100, k_kinematics)

        # 4. A_acoustics
        a_acoustics = telemetry.acoustic_distress_prob * 100

        # Weights per spec
        w_context, w_isolation, w_kinematic, w_acoustic = 0.2, 0.3, 0.5, 0.8

        # Dynamic Weight Adjustment
        if i_isolation > 70: w_context = 0.4

        r_raw = (w_context * c_context + w_isolation * i_isolation +
                 w_kinematic * k_kinematics + w_acoustic * a_acoustics)

        # Temporal Smoothing (EMA)
        self.risk_ema = (self.alpha * r_raw) + (1 - self.alpha) * self.risk_ema
        return self.risk_ema

    def update_state(self, telemetry: SensorTelemetry):
        risk = self.compute_risk(telemetry)
        old_level = self.current_level

        # Transitions
        if risk > 75 or telemetry.acoustic_distress_prob > 0.8:
            self.current_level = ThreatLevel.LEVEL_4_CRITICAL
        elif risk > 50 and telemetry.acoustic_distress_prob > 0.4:
            self.current_level = ThreatLevel.LEVEL_3_WARNING
        elif risk > 30 or telemetry.accel_magnitude > 15:
            if self.current_level.value < ThreatLevel.LEVEL_2_GUARDED.value:
                self.current_level = ThreatLevel.LEVEL_2_GUARDED

        if old_level != self.current_level:
            print(f"--- Transition: {old_level.name} -> {self.current_level.name} (Risk: {risk:.2f}) ---")
            getattr(self, f"_on_enter_level_{self.current_level.value}")()

        return self.current_level, risk

if __name__ == '__main__':
    engine = DynamicRiskEngine()
    print("Starting AegisEdge Risk Engine Simulation...\n")

    stages = [
        ("Stage 1: Daytime Safe Zone", SensorTelemetry(time.time(), True, 14, 8, 9.8, 0.1, 0.0)),
        ("Stage 2: Isolated Alley 2 AM", SensorTelemetry(time.time(), False, 2, 0, 10.2, 0.5, 0.0)),
        ("Stage 3: Violent Ambush/Struggle", SensorTelemetry(time.time(), False, 2, 0, 28.5, 45.0, 0.45)),
        ("Stage 4: Acoustic Distress (Scream)", SensorTelemetry(time.time(), False, 2, 0, 15.0, 10.0, 0.92))
    ]

    for description, data in stages:
        print(f"\nExecuting {description}")
        level, score = engine.update_state(data)
        print(f"Current Status: {level.name} | Risk Score: {score:.2f}")