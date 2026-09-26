
# AEGISEDGE_SPEC.md

## 1. System Mission & Failure Mode Analysis:

*   **The Intentional Action Barrier:** Manual SOS buttons fail during sudden ambushes due to the "intentional action barrier" – the critical window where a victim is physically unable or too overwhelmed to initiate a distress signal. This creates a fatal delay in emergency response.
*   **Cloud Latency & Centralized Dispatch Failure:** Relying on cloud-based processing introduces significant latency (15–30 minutes for dispatch in critical scenarios), making it unsuitable for immediate, life-threatening events. Centralized dispatch mechanisms are prone to single points of failure and network disruptions.
*   **The Edge AI Solution:** AegisEdge provides continuous, zero-latency, autonomous local threat assessment and micro-acoustic distress detection directly on-device. This eliminates the intentional action barrier and cloud latency, enabling instantaneous, localized response initiation.

## 2. Tiered Snapdragon Hardware Execution Architecture:

AegisEdge leverages a highly optimized, tiered execution model across the Snapdragon SoC to achieve ultra-low power consumption for continuous monitoring and rapid escalation upon threat detection.

*   **Level 1 (Baseline / Safe): Monitored by Qualcomm Sensing Hub (uDSP)**
    *   **Power Consumption:** <1mW
    *   **Activities:** Continuous BLE scanning (for proximity detection), pedometer/IMU data analysis (for sudden impacts/falls), geofencing (for safe zone monitoring).
    *   **Purpose:** Ultra-low power, always-on context gathering to detect initial anomalies or pre-threat indicators.

*   **Level 2 (Guarded / Active Watch): LPASS + Hardware VAD active**
    *   **Power Consumption:** Low, but higher than Level 1.
    *   **Activities:** Low-Power Audio Subsystem (LPASS) activated. Hardware Voice Activity Detection (VAD) continuously monitors acoustic environment for potential distress sounds. Hexagon NPU pre-warmed in low-leakage retention mode, ready for rapid activation.
    *   **Purpose:** Proactive acoustic monitoring without full NPU activation, minimizing power draw while maintaining readiness.

*   **Level 3 (Warning / Escalation): Hexagon NPU burst compute triggered via hardware interrupt**
    *   **Power Consumption:** Increased temporarily during burst.
    *   **Activities:** A hardware interrupt (from VAD or IMU anomaly) instantly triggers Hexagon NPU burst compute. INT8 acoustic inference is performed on a rolling SRAM ring-buffer (up to 5 seconds of audio). A silent haptic countdown (e.g., specific vibration patterns) is initiated to allow for user cancellation.
    *   **Purpose:** Rapid, localized, and energy-efficient inference to confirm a threat event. Provides a brief window for user intervention.

*   **Level 4 (Critical / Dispatch): Full SoC wakeup**
    *   **Power Consumption:** Highest, but for minimal duration.
    *   **Activities:** If the threat is confirmed (or countdown expires), the full SoC wakes up. TrustZone is utilized for AES-256 black-box data encryption of contextual data (e.g., location, time, sensor readings leading to event). FastConnect enables a P2P BLE mesh broadcast of encrypted distress signals to nearby compatible devices, forming an ad-hoc emergency network.
    *   **Purpose:** Secure, resilient, and rapid transmission of critical distress information to a local network, bypassing cloud dependencies.

## 3. Mathematical Threat Matrix Formula:

The AegisEdge threat assessment relies on a multi-modal, weighted formula to dynamically assess the severity of a situation and trigger appropriate escalation levels.

$R_{total} = w_1 \cdot Context + w_2 \cdot Environmental\_Isolation + w_3 \cdot Kinematic\_Acoustic\_Cues$

*   **$R_{total}$:** Total Risk Score (ranges from 0 to 100).
*   **$Context$ (0-100):** Aggregated score from Level 1 sensors.
    *   **Parameters:** `BLE_Proximity_Score` (0-50, inverse relation to number of trusted devices nearby), `Geofence_Violation` (0-30, binary: 0 if in safe zone, 30 if outside), `Time_of_Day_Risk` (0-20, higher at night/early morning).
    *   **Decay Factor:** `Context_Decay_Rate = 0.1` per minute (gradually reduces `Context` score if no further events).
*   **$Environmental\_Isolation$ (0-100):** Assessed from network connectivity and ambient noise levels.
    *   **Parameters:** `Cellular_Signal_Strength` (0-50, inverse), `WiFi_Availability` (0-25, inverse), `Ambient_Noise_Level` (0-25, lower ambient noise indicates higher isolation).
    *   **Thresholds:** `Isolation_Threshold = 70` (above this, `w2` increases).
*   **$Kinematic\_Acoustic\_Cues$ (0-100):** Direct output from Level 3 NPU inference.
    *   **Parameters:** `Acoustic_Distress_Probability` (0-70, from micro-acoustic classifier), `IMU_Impact_Severity` (0-30, from sudden accelerometer/gyroscope changes).
    *   **Scoring:** Weighted sum of probabilities/magnitudes.

*   **Weights:**
    *   $w_1$: Weight for Context (default: 0.2, increases to 0.4 if `Environmental_Isolation` > 70).
    *   $w_2$: Weight for Environmental_Isolation (default: 0.3, increases to 0.5 if `Context` > 50 and `Kinematic_Acoustic_Cues` == 0).
    *   $w_3$: Weight for Kinematic_Acoustic_Cues (default: 0.5, increases to 0.8 if VAD is active and `Acoustic_Distress_Probability` > 0).

*   **Transition Boundaries:**
    *   **Level 1 -> 2:** $R_{total} > 30$ OR `IMU_Impact_Severity` > 15.
    *   **Level 2 -> 3:** $R_{total} > 50$ AND `Acoustic_Distress_Probability` > 0.4.
    *   **Level 3 -> 4:** $R_{total} > 75$ OR `Acoustic_Distress_Probability` > 0.8 (after haptic countdown expiry).

## 4. Target Hardware Constraints & AI Hub Pipeline:

*   **Target Profile:** Qualcomm Hexagon NPU (HTP), Qualcomm Sensing Hub (uDSP), Snapdragon X CRD / Snapdragon 8 Gen 3 Mobile Platform.
*   **Latency Target:** <30ms end-to-end inference budget for Level 3 acoustic classification (from audio buffer fill to NPU output).
*   **Privacy Guarantee:** Strict zero-raw-waveform storage. Log-Mel spectrogram generation (the input to the AI model) is performed strictly in volatile memory (SRAM/DRAM) and is never persisted to non-volatile storage. Only encrypted metadata and threat scores are transmitted.
*   **Model Compilation & Profiling Target:** Qualcomm AI Hub SDK (`qai-hub`) using static INT8 quantized execution for all deep learning models to ensure maximum NPU efficiency and minimal memory footprint.

## 5. Repository & Workspace File Structure:

*   `AEGISEDGE_SPEC.md` (This system blueprint and technical specification)
*   `risk_engine.py` (Multi-factor threat scoring state machine and level transition logic)
*   `acoustic_monitor.py` (PyTorch/ONNX micro-acoustic distress classifier implementation, including Log-Mel spectrogram generation)
*   `mesh_beacon.py` (Decentralized peer-to-peer BLE broadcast & parser for emergency signals)
*   `snapdragon_profiler.py` (Qualcomm AI Hub compilation, latency profiling, and memory benchmarking scripts for on-device optimization)
*   `main.py` (Real-time simulation harness linking all modules and demonstrating end-to-end functionality)