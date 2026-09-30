"""QNN/CPU acoustic inference telemetry and optional AI Hub submission."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Iterable

import numpy as np

try:
    from .acoustic_monitor import AcousticPipeline, BUFFER_SIZE, STATIC_INPUT_SHAPE
except ImportError:  # Support direct execution from the src directory.
    from acoustic_monitor import AcousticPipeline, BUFFER_SIZE, STATIC_INPUT_SHAPE


QNN_BACKEND_PATH = "QnnHtp.dll"
QNN_PERFORMANCE_MODE = "burst"


@dataclass(frozen=True)
class BenchmarkResult:
    provider: str
    qnn_available: bool
    htp_backend: str
    samples: int
    p50_latency_ms: float
    p95_latency_ms: float
    inferences_per_second: float
    note: str = ""


def _percentile(values: Iterable[float], percentile: float) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(values, percentile)) if values.size else 0.0


def benchmark_provider(model_path: str, provider: str, iterations: int = 20) -> BenchmarkResult:
    """Measure rolling inference latency and throughput for one provider."""
    pipeline = AcousticPipeline(model_path=model_path, preferred_provider=provider)
    audio = np.zeros(BUFFER_SIZE, dtype=np.float32)
    latencies = []
    started = time.perf_counter()
    for _ in range(iterations):
        event = pipeline.process_live_buffer(audio)
        if event.inference_latency_ms > 0:
            latencies.append(event.inference_latency_ms)
    elapsed = time.perf_counter() - started
    qnn_available = pipeline.execution_provider == "QNNExecutionProvider"
    return BenchmarkResult(
        provider=pipeline.execution_provider,
        qnn_available=qnn_available,
        htp_backend=QNN_BACKEND_PATH if qnn_available else "n/a",
        samples=len(latencies),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        inferences_per_second=len(latencies) / elapsed if elapsed else 0.0,
        note=pipeline.provider_error or "provider active",
    )


def profile_acoustic_model(iterations: int = 20) -> tuple[BenchmarkResult, BenchmarkResult]:
    model_path = os.environ.get("AEGSEDGE_ACOUSTIC_MODEL", "models/acoustic_event_classifier_int8.onnx")
    print("--- AegisEdge Snapdragon acoustic telemetry ---")
    print(f"Model: {model_path}")
    print(f"Input: {STATIC_INPUT_SHAPE} | QNN backend: {QNN_BACKEND_PATH} | mode: {QNN_PERFORMANCE_MODE}")
    qnn = benchmark_provider(model_path, "QNNExecutionProvider", iterations)
    cpu = benchmark_provider(model_path, "CPUExecutionProvider", iterations)
    print(f"QNN provider verified: {qnn.qnn_available} ({qnn.provider})")
    for result in (qnn, cpu):
        print(
            f"{result.provider}: p50={result.p50_latency_ms:.2f} ms, "
            f"p95={result.p95_latency_ms:.2f} ms, "
            f"inferences/s={result.inferences_per_second:.2f}"
        )
        if result.note:
            print(f"  note: {result.note}")
    return qnn, cpu


if __name__ == "__main__":
    profile_acoustic_model()
