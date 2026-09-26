import time
import math
import torch
import numpy as np
import os
from risk_engine import DynamicRiskEngine, SensorTelemetry, ThreatLevel
from acoustic_monitor import AcousticPipeline, BUFFER_SIZE
from mesh_beacon import EvidenceVault, MeshBroadcaster, MeshListener

class AegisEdgeSystem:
    def __init__(self):
        # Initialize Subsystems
        self.risk_engine = DynamicRiskEngine()
        self.acoustic_pipeline = AcousticPipeline()

        # Security & Mesh Initialization
        self.key = os.urandom(32)
        self.vault = EvidenceVault(self.key)
        self.broadcaster = MeshBroadcaster(self.vault, "AEGIS_NODE_S24")
        self.bystander = MeshListener(self.vault)

        self.system_clock = 0.0

    def get_hardware_profile(self, level):
        profiles = {
            ThreatLevel.LEVEL_1_BASELINE: ("Sensing Hub uDSP", "< 1mW", "Always-On Low Power"),
            ThreatLevel.LEVEL_2_GUARDED: ("LPASS Audio VAD + Retention", "~ 5mW", "Audio Pre-warmed"),
            ThreatLevel.LEVEL_3_WARNING: ("Hexagon NPU (HTP)", "< 25mW Burst", "4.2ms NPU Inference"),
            ThreatLevel.LEVEL_4_CRITICAL: ("Kryo CPU + FastConnect + TrustZone", "Max Burst", "Mesh Lockdown")
        }
        return profiles.get(level)

    def run_simulation_phase(self, phase_name, telemetry):
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name}")
        print(f"{'='*60}")

        # 1. State Update
        level, score = self.risk_engine.update_state(telemetry)
        hw_domain, pwr, desc = self.get_hardware_profile(level)

        # 2. Logic Branching based on Level
        latency = 0.0
        if level.value >= 3:
            # Simulate NPU Inference
            dummy_audio = np.random.normal(0, 0.02, BUFFER_SIZE).astype(np.float32)
            _, latency = self.acoustic_pipeline.process_live_buffer(dummy_audio)

        # 3. Critical Mobilization
        mesh_result = None
        if level == ThreatLevel.LEVEL_4_CRITICAL:
            packet = self.broadcaster.emit_alert(37.7749, -122.4194, score, "DISTRESS_DETECTED")
            mesh_result = self.bystander.process_scan(packet)

        # 4. Dashboard Print
        print(f"[DASHBOARD]")
        print(f"  - Active HW Domain:  {hw_domain}")
        print(f"  - Power Profile:    {pwr} ({desc})")
        print(f"  - Composite Risk:   {score:.2f} / 100")
        print(f"  - NPU Latency:      {latency:.2f}ms (Budget: 30ms)")

        if mesh_result:
            print(f"  - Mesh Status:      BROADCASTING - Peer Received Data: {mesh_result['sig']}")

def run_mission():
    aegis = AegisEdgeSystem()

    # Phase Definitions
    phases = [
        ("1. Safe Urban Transit",
         SensorTelemetry(time.time(), True, 14, 12, 9.8, 0.1, 0.0)),

        ("2. Isolated Night Walk",
         SensorTelemetry(time.time(), False, 2, 0, 10.1, 0.2, 0.0)),

        ("3. Physical Ambush",
         SensorTelemetry(time.time(), False, 2, 0, 32.5, 65.0, 0.42)),

        ("4. Vocal Distress Recovery",
         SensorTelemetry(time.time(), False, 2, 0, 12.0, 5.0, 0.95))
    ]

    for name, tel in phases:
        aegis.run_simulation_phase(name, tel)
        time.sleep(0.5)

if __name__ == '__main__':
    run_mission()