"""Streaming acoustic classification for Qualcomm Hexagon HTP or CPU.

The ONNX model is expected to consume a normalized log-mel tensor with shape
``(1, 1, 64, 188)``. A quantized model compiled for QNN can be selected with
``AEGSEDGE_ACOUSTIC_MODEL`` or by passing ``model_path`` to ``AcousticPipeline``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time
from typing import Any, Iterable

import numpy as np
from scipy.signal import get_window

try:
    import onnxruntime as ort
except ImportError:  # Keep feature extraction importable on development hosts.
    ort = None


SAMPLE_RATE = 16_000
DURATION_SEC = 3
BUFFER_SIZE = SAMPLE_RATE * DURATION_SEC
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
N_FRAMES = 188
STATIC_INPUT_SHAPE = (1, 1, N_MELS, N_FRAMES)
DEFAULT_MODEL_PATH = "models/acoustic_event_classifier_int8.onnx"
DEFAULT_CATEGORIES = (
    "ambient",
    "distress_voice",
    "impact",
    "glass_break",
    "alarm",
)


@dataclass(frozen=True)
class ThreatEvent:
    """A classification result safe to forward as metadata only."""

    timestamp: float
    category: str
    confidence: float
    inference_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __iter__(self):
        """Preserve the old ``probability, latency`` simulation API."""
        yield self.confidence
        yield self.inference_latency_ms


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Build a triangular mel filterbank without librosa."""
    f_max = f_max or sample_rate / 2.0
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index:index + 3]
        if center > left:
            filters[index, left:center] = np.arange(left, center) - left
            filters[index, left:center] /= center - left
        if right > center:
            filters[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


MEL_FILTERBANK = _mel_filterbank()
WINDOW = get_window("hann", N_FFT, fftbins=True).astype(np.float32)


def compute_log_mel_spectrogram(audio_buffer: np.ndarray) -> np.ndarray:
    """Return a normalized ``(1, 1, 64, 188)`` float32 model input."""
    samples = np.asarray(audio_buffer)
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float32, copy=False)
    samples = np.nan_to_num(samples, copy=False)

    required_samples = N_FFT + (N_FRAMES - 1) * HOP_LENGTH
    if samples.size < required_samples:
        samples = np.pad(samples, (0, required_samples - samples.size))
    else:
        samples = samples[:required_samples]

    frames = np.lib.stride_tricks.sliding_window_view(samples, N_FFT)[::HOP_LENGTH]
    frames = frames[:N_FRAMES] * WINDOW
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel_power = np.maximum(spectrum @ MEL_FILTERBANK.T, 1e-10)
    log_mel = 10.0 * np.log10(mel_power)
    log_mel -= np.max(log_mel, axis=1, keepdims=True)
    log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
    return log_mel.T[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)


class AcousticPipeline:
    """QNN-first streaming acoustic classifier with an explicit CPU fallback."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        categories: Iterable[str] = DEFAULT_CATEGORIES,
        buffer_size: int = BUFFER_SIZE,
        preferred_provider: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or os.environ.get("AEGSEDGE_ACOUSTIC_MODEL", DEFAULT_MODEL_PATH))
        self.categories = tuple(categories)
        self.audio_buffer: deque[float] = deque(maxlen=buffer_size)
        self.session: Any = None
        self.execution_provider = "unavailable"
        self.provider_error: str | None = None
        self.input_name: str | None = None
        self.preferred_provider = preferred_provider
        self._load_session()

    def _load_session(self) -> None:
        if ort is None or not self.model_path.is_file():
            self.provider_error = "onnxruntime or model unavailable"
            return

        qnn_options = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
        }
        available = ort.get_available_providers()
        try:
            if self.preferred_provider == "CPUExecutionProvider":
                raise RuntimeError("CPU provider requested")
            if "QNNExecutionProvider" not in available:
                raise RuntimeError("QNNExecutionProvider is not available")
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=[("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"],
            )
            self.execution_provider = "QNNExecutionProvider" if "QNNExecutionProvider" in self.session.get_providers() else "CPUExecutionProvider"
        except Exception as error:
            self.provider_error = str(error)
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
                self.execution_provider = "CPUExecutionProvider"
            except Exception:
                self.session = None
                self.execution_provider = "unavailable"
        if self.session is not None:
            self.input_name = self.session.get_inputs()[0].name

    @property
    def using_qnn(self) -> bool:
        return self.execution_provider == "QNNExecutionProvider"

    def _classify(self, model_input: np.ndarray) -> tuple[str, float]:
        if self.session is None or self.input_name is None:
            return "model_unavailable", 0.0

        output = np.asarray(self.session.run(None, {self.input_name: model_input})[0]).squeeze()
        if output.ndim == 0 or output.size == 1:
            confidence = float(np.clip(output.item(), 0.0, 1.0))
            category = self.categories[1] if confidence >= 0.5 and len(self.categories) > 1 else self.categories[0]
            return category, confidence if category != self.categories[0] else 1.0 - confidence

        values = output.astype(np.float32)
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(values.sum(), 1.0, atol=0.05):
            values = np.exp(values - np.max(values))
            values /= np.sum(values)
        index = int(np.argmax(values))
        category = self.categories[index] if index < len(self.categories) else f"class_{index}"
        return category, float(values[index])

    def process_live_buffer(self, audio_array: np.ndarray) -> ThreatEvent:
        """Append a raw PCM chunk and classify the newest full rolling window."""
        samples = np.asarray(audio_array).reshape(-1)
        self.audio_buffer.extend(samples.tolist())
        if len(self.audio_buffer) < self.audio_buffer.maxlen:
            return ThreatEvent(time.time(), "buffering", 0.0, 0.0)

        start = time.perf_counter()
        model_input = compute_log_mel_spectrogram(np.asarray(self.audio_buffer, dtype=np.float32))
        category, confidence = self._classify(model_input)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ThreatEvent(time.time(), category, confidence, latency_ms)


if __name__ == "__main__":
    pipeline = AcousticPipeline()
    ambient = np.random.normal(0, 0.01, BUFFER_SIZE).astype(np.float32)
    event = pipeline.process_live_buffer(ambient)
    print(event.as_dict())
