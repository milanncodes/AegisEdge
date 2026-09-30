"""AegisEdge edge-defense monitoring console.

Usage:
    python main.py live
    python main.py simulate
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import time

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.acoustic_monitor import AcousticPipeline, BUFFER_SIZE
from src.mesh_beacon import EvidenceVault, MeshBroadcaster, MeshListener
from src.risk_engine import DynamicRiskEngine, SensorTelemetry, ThreatLevel
from src.snapdragon_profiler import benchmark_provider


console = Console()


class AegisEdgeSystem:
    """Connect acoustic sensing, risk scoring, and encrypted mesh relay."""

    def __init__(self) -> None:
        self.risk_engine = DynamicRiskEngine()
        self.acoustic_pipeline = AcousticPipeline()
        vault = EvidenceVault(os.urandom(32))
        self.broadcaster = MeshBroadcaster(vault, "AEGIS_NODE_S24")
        self.bystander = MeshListener(vault, trusted_device_ids=["AEGIS_NODE_S24"])
        self.event_log: deque[dict] = deque(maxlen=12)
        self.mesh_nodes = {"AEGIS_NODE_S24": "Local", "PEER-HTP-07": "In range"}

    @property
    def provider(self) -> str:
        return self.acoustic_pipeline.execution_provider

    def process_fixture(self, name: str, audio: np.ndarray, telemetry: SensorTelemetry) -> dict:
        event = self.acoustic_pipeline.process_live_buffer(audio)
        level, score = self.risk_engine.update_state(telemetry)
        category = name if event.category in {"model_unavailable", "buffering"} else event.category
        result = {
            "timestamp": event.timestamp,
            "category": category,
            "confidence": max(event.confidence, telemetry.acoustic_distress_prob),
            "latency": event.inference_latency_ms,
            "level": level,
            "score": score,
            "mesh": False,
        }
        self.event_log.append(result)
        if level.value >= ThreatLevel.LEVEL_4_CRITICAL.value:
            frame = self.broadcaster.emit_alert(37.7749, -122.4194, score, category.upper())
            result["mesh"] = self.bystander.process_scan(frame) is not None
        return result


def _fixture_audio(kind: str) -> np.ndarray:
    """Create deterministic raw PCM-like fixtures for the simulation path."""
    samples = np.arange(BUFFER_SIZE, dtype=np.float32) / 16_000
    rng = np.random.default_rng(sum(kind.encode("utf-8")))
    noise = rng.normal(0.0, 0.008, samples.size).astype(np.float32)
    frequencies = {"Glass Break": 3_200, "Siren": 1_100, "Distress Call": 2_600}
    tone = np.sin(2 * np.pi * frequencies[kind] * samples).astype(np.float32)
    envelope = np.exp(-((samples - 1.5) ** 2) / 0.08).astype(np.float32)
    return noise + 0.65 * tone * envelope


def _telemetry_for(kind: str) -> SensorTelemetry:
    distress = {"Glass Break": 0.55, "Siren": 0.68, "Distress Call": 0.95}[kind]
    impact = 32.0 if kind == "Glass Break" else 14.0
    jerk = 48.0 if kind == "Glass Break" else 8.0
    return SensorTelemetry(time.time(), False, 2, 0, impact, jerk, distress)


def _status_panel(system: AegisEdgeSystem, latest: dict | None) -> Panel:
    level = latest["level"].name.replace("LEVEL_", "").replace("_", " ") if latest else "BASELINE"
    status = Table.grid(padding=(0, 2))
    status.add_column(style="bold cyan")
    status.add_column()
    status.add_row("Sensor", f"[green]Listening[/green]  | Threat Level: [bold yellow]{level}[/bold yellow]")
    status.add_row("Risk Score", f"{latest['score']:.1f} / 100" if latest else "0.0 / 100")
    return Panel(status, title="Live Acoustic Sensor Status", border_style="cyan")


def _npu_panel(system: AegisEdgeSystem, latest: dict | None) -> Panel:
    latency = latest["latency"] if latest else 0.0
    provider = system.provider.replace("ExecutionProvider", " EP")
    badge = "[green]ACTIVE[/green]" if system.acoustic_pipeline.using_qnn else "[yellow]CPU FALLBACK[/yellow]"
    return Panel(
        Text.from_markup(f"{badge}  [bold]{provider}[/bold]\nLatency: [bold]{latency:.2f} ms[/bold]  | Budget: 30 ms"),
        title="Hexagon NPU Telemetry",
        border_style="green",
    )


def _event_table(system: AegisEdgeSystem) -> Table:
    table = Table(title="Event Log", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Time", style="dim", width=10)
    table.add_column("Detected Sound")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Risk", justify="right")
    for event in list(system.event_log)[-8:]:
        table.add_row(
            time.strftime("%H:%M:%S", time.localtime(event["timestamp"])),
            event["category"],
            f"{event['confidence']:.0%}",
            f"{event['latency']:.2f} ms",
            f"{event['score']:.1f}",
        )
    if not system.event_log:
        table.add_row("--:--:--", "Awaiting acoustic input", "--", "--", "--")
    return table


def _mesh_panel(system: AegisEdgeSystem) -> Panel:
    table = Table(box=box.MINIMAL, expand=True)
    table.add_column("Node")
    table.add_column("Status")
    for node, status in system.mesh_nodes.items():
        table.add_row(node, f"[green]{status}[/green]")
    table.add_row("Queued alerts", str(system.broadcaster.pending_count))
    return Panel(table, title="Active Mesh Nodes in Range", border_style="magenta")


def render_console(system: AegisEdgeSystem, latest: dict | None = None) -> Group:
    header = Panel(
        Text("AEGISEDGE  /  EDGE DEFENSE MONITOR", style="bold white", justify="center"),
        style="on #12233b",
        border_style="bright_blue",
    )
    top = Table.grid(expand=True)
    top.add_column(ratio=3)
    top.add_column(ratio=2)
    top.add_row(_status_panel(system, latest), _npu_panel(system, latest))
    bottom = Table.grid(expand=True)
    bottom.add_column(ratio=3)
    bottom.add_column(ratio=2)
    bottom.add_row(_event_table(system), _mesh_panel(system))
    return Group(header, top, bottom)


def run_simulation() -> None:
    system = AegisEdgeSystem()
    console.print("[bold cyan]Starting AegisEdge synthetic end-to-end detection[/bold cyan]")
    for kind in ("Glass Break", "Siren", "Distress Call"):
        result = system.process_fixture(kind, _fixture_audio(kind), _telemetry_for(kind))
        mesh = "broadcast + decrypted locally" if result["mesh"] else "local only"
        console.print(
            f"[bold]{kind}[/bold]  category={result['category']}  "
            f"confidence={result['confidence']:.0%}  risk={result['score']:.1f}  mesh={mesh}"
        )
    console.print(render_console(system, system.event_log[-1]))


def run_live(refresh: float = 1.0) -> None:
    system = AegisEdgeSystem()
    latest = None
    try:
        with Live(render_console(system), console=console, refresh_per_second=4, screen=True) as live:
            while True:
                audio = np.random.normal(0, 0.008, BUFFER_SIZE).astype(np.float32)
                telemetry = SensorTelemetry(time.time(), True, time.localtime().tm_hour, 4, 9.8, 0.1, 0.0)
                latest = system.process_fixture("Ambient", audio, telemetry)
                live.update(render_console(system, latest))
                time.sleep(refresh)
    except KeyboardInterrupt:
        console.print("\n[dim]Live monitoring stopped.[/dim]")


def run_benchmark(iterations: int = 20) -> None:
    model_path = os.environ.get("AEGSEDGE_ACOUSTIC_MODEL", "models/acoustic_event_classifier_int8.onnx")
    results = [
        benchmark_provider(model_path, "QNNExecutionProvider", iterations),
        benchmark_provider(model_path, "CPUExecutionProvider", iterations),
    ]
    table = Table(title="Snapdragon Inference Telemetry", box=box.ROUNDED)
    table.add_column("Provider")
    table.add_column("HTP Backend")
    table.add_column("QNN Verified")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("Inferences/s", justify="right")
    for result in results:
        table.add_row(
            result.provider,
            result.htp_backend,
            "YES" if result.qnn_available else "NO",
            f"{result.p50_latency_ms:.2f}",
            f"{result.p95_latency_ms:.2f}",
            f"{result.inferences_per_second:.2f}",
        )
    console.print(Panel(table, title="AegisEdge Benchmark", border_style="bright_blue"))
    for result in results:
        if result.note and result.provider == "unavailable":
            console.print(f"[yellow]Note:[/yellow] {result.note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AegisEdge edge defense monitoring console")
    parser.add_argument("command", choices=("live", "simulate", "benchmark"), help="console mode")
    args = parser.parse_args()
    if args.command == "live":
        run_live()
    elif args.command == "simulate":
        run_simulation()
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
