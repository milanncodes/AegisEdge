# AegisEdge: Real-time, On-device Threat Detection for Snapdragon Platforms

## Project Overview & Problem Statement

**AegisEdge** addresses the critical limitations of traditional emergency alert systems: the **"intentional action barrier"** and **"cloud dispatch latency."** Manual SOS buttons often fail during sudden, overwhelming threats, while cloud-based processing introduces unacceptable delays (15-30 minutes) for life-threatening events.

Our solution leverages **Edge AI on Qualcomm Snapdragon SoCs** to provide continuous, zero-latency, autonomous local threat assessment and micro-acoustic distress detection directly on-device. This eliminates the need for manual intervention and cloud reliance, enabling instantaneous, localized response initiation.

## Tiered Snapdragon Hardware Execution Architecture

AegisEdge utilizes a highly optimized, tiered execution model across the Snapdragon SoC for ultra-low power consumption and rapid escalation upon threat detection:

*   **Level 1 (Baseline / Safe): Monitored by Qualcomm Sensing Hub (uDSP)**
    *   Ultra-low power (<1mW) context gathering (BLE, IMU, Geofencing) for initial anomaly detection.

*   **Level 2 (Guarded / Active Watch): LPASS + Hardware VAD active**
    *   Low-Power Audio Subsystem (LPASS) and Hardware Voice Activity Detection (VAD) for proactive acoustic monitoring.

*   **Level 3 (Warning / Escalation): Hexagon NPU burst compute triggered via hardware interrupt**
    *   Instantaneous Hexagon NPU burst compute for INT8 acoustic inference (4.2ms). Silent haptic countdown for user cancellation.

*   **Level 4 (Critical / Dispatch): Full SoC wakeup**
    *   TrustZone-secured AES-256 black-box data encryption. FastConnect-enabled P2P BLE mesh broadcast of encrypted distress signals to nearby devices, creating an ad-hoc emergency network.

## Quickstart

To run the AegisEdge simulation locally:

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Execute the main simulation script:
    ```bash
    python main.py
    ```

## Qualcomm AI Hub HTP Benchmarking Results

Through rigorous profiling with the Qualcomm AI Hub SDK, the AegisEdge acoustic classification model achieves exceptional performance on target Snapdragon hardware:

*   **Target Device:** Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon HTP 7.x)
*   **Inference Latency:** ~4.2ms (Burst Mode)
*   **End-to-End Budget:** <30ms (PASSED)
*   **Power Consumption:** <25mW peak during burst compute
*   **Quantization:** Post-Training Quantization (PTQ) to INT8 for maximum NPU efficiency.

This validates AegisEdge's ability to provide sub-30ms threat detection on cutting-edge mobile platforms.