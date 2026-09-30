"""Encrypted local mesh relay with UDP multicast and store-and-forward."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
import socket
import struct
import time
from typing import Callable, Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


AEGIS_DEMO_ID = 0x0A10
PROTOCOL_VERSION = 0x01
FLAG_LEVEL_4_CRITICAL = 0x80
MESH_MULTICAST_GROUP = "239.255.42.99"
MESH_PORT = 45678
HEADER_FORMAT = ">HB8s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class EvidenceVault:
    """AES-GCM-256 vault shared by trusted nodes in one mesh."""

    def __init__(self, key_256: bytes):
        if len(key_256) != 32:
            raise ValueError("AES-GCM-256 requires a 32-byte network key")
        self.aesgcm = AESGCM(key_256)

    def seal(self, telemetry_dict: dict, associated_data: bytes | None = None) -> bytes:
        data = json.dumps(telemetry_dict, separators=(",", ":")).encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, data, associated_data)
        return nonce + ciphertext

    def open(self, sealed_payload: bytes, associated_data: bytes | None = None) -> dict:
        if len(sealed_payload) < 28:
            raise ValueError("Truncated encrypted mesh payload")
        nonce, ciphertext = sealed_payload[:12], sealed_payload[12:]
        decrypted_data = self.aesgcm.decrypt(nonce, ciphertext, associated_data)
        return json.loads(decrypted_data.decode("utf-8"))


class UdpMeshTransport:
    """Local-subnet multicast transport for internet-independent relay."""

    def __init__(
        self,
        multicast_group: str = MESH_MULTICAST_GROUP,
        port: int = MESH_PORT,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ):
        self.multicast_group = multicast_group
        self.port = port
        self._socket_factory = socket_factory
        self._sender = socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self._receiver: socket.socket | None = None

    def send(self, frame: bytes) -> None:
        self._sender.sendto(frame, (self.multicast_group, self.port))

    def start_receiver(self, timeout: float = 0.1) -> None:
        if self._receiver is not None:
            return
        receiver = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.bind(("", self.port))
        membership = struct.pack(
            "4s4s", socket.inet_aton(self.multicast_group), socket.inet_aton("0.0.0.0")
        )
        receiver.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        receiver.settimeout(timeout)
        self._receiver = receiver

    def receive_once(self, timeout: float = 0.1) -> bytes | None:
        self.start_receiver(timeout)
        try:
            return self._receiver.recv(65535)
        except socket.timeout:
            return None

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
        self._sender.close()


class AegisBeaconPacket:
    def __init__(self, device_id: str, encrypted_payload: bytes):
        self.prefix = AEGIS_DEMO_ID
        self.version_flags = PROTOCOL_VERSION | FLAG_LEVEL_4_CRITICAL
        self.device_hash = hashlib.sha256(device_id.encode("utf-8")).digest()[:8]
        self.payload = encrypted_payload

    def header_bytes(self) -> bytes:
        return struct.pack(HEADER_FORMAT, self.prefix, self.version_flags, self.device_hash)

    def to_bytes(self) -> bytes:
        return self.header_bytes() + self.payload

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> tuple[bytes, int, bytes]:
        if len(raw_bytes) < HEADER_SIZE + 28:
            raise ValueError("Truncated mesh frame")
        prefix, flags, dev_hash = struct.unpack(HEADER_FORMAT, raw_bytes[:HEADER_SIZE])
        if prefix != AEGIS_DEMO_ID or flags & 0x0F != PROTOCOL_VERSION:
            raise ValueError("Invalid mesh protocol header")
        return dev_hash, flags, raw_bytes[HEADER_SIZE:]


class MeshBroadcaster:
    def __init__(
        self,
        vault: EvidenceVault,
        device_id: str,
        transport: UdpMeshTransport | None = None,
        max_pending: int = 128,
    ):
        self.vault = vault
        self.device_id = device_id
        self.transport = transport or UdpMeshTransport()
        self.pending_frames: deque[bytes] = deque(maxlen=max_pending)

    def _build_frame(self, telemetry: dict) -> bytes:
        header = AegisBeaconPacket(self.device_id, b"").header_bytes()
        return header + self.vault.seal(telemetry, associated_data=header)

    def _transmit(self, frame: bytes) -> bool:
        try:
            self.transport.send(frame)
            return True
        except OSError:
            self.pending_frames.append(frame)
            return False

    def flush_pending(self) -> int:
        """Retry queued encrypted events after a peer/network reconnects."""
        sent = 0
        while self.pending_frames:
            frame = self.pending_frames[0]
            try:
                self.transport.send(frame)
            except OSError:
                break
            self.pending_frames.popleft()
            sent += 1
        return sent

    @property
    def pending_count(self) -> int:
        return len(self.pending_frames)

    def emit_alert(self, lat: float, lon: float, score: float, signature: str) -> bytes:
        telemetry = {"ts": time.time(), "gps": [lat, lon], "rsk": score, "sig": signature}
        frame = self._build_frame(telemetry)
        self._transmit(frame)
        return frame


class MeshListener:
    def __init__(
        self,
        vault: EvidenceVault,
        transport: UdpMeshTransport | None = None,
        trusted_device_ids: Iterable[str] | None = None,
    ):
        self.vault = vault
        self.transport = transport
        self.trusted_device_hashes = {
            hashlib.sha256(device_id.encode("utf-8")).digest()[:8]
            for device_id in (trusted_device_ids or ())
        }

    def process_scan(self, raw_frame: bytes) -> dict | None:
        try:
            dev_hash, flags, sealed = AegisBeaconPacket.from_bytes(raw_frame)
            if self.trusted_device_hashes and dev_hash not in self.trusted_device_hashes:
                return None
            if flags & FLAG_LEVEL_4_CRITICAL:
                telemetry = self.vault.open(sealed, associated_data=raw_frame[:HEADER_SIZE])
                telemetry["sender"] = dev_hash.hex()
                return telemetry
        except (InvalidTag, ValueError, TypeError, json.JSONDecodeError):
            return None
        return None

    def receive_once(self, timeout: float = 0.1) -> dict | None:
        if self.transport is None:
            raise RuntimeError("MeshListener requires a transport for receive_once()")
        frame = self.transport.receive_once(timeout)
        return self.process_scan(frame) if frame else None


if __name__ == "__main__":
    key = os.urandom(32)
    transport = UdpMeshTransport()
    vault = EvidenceVault(key)
    broadcaster = MeshBroadcaster(vault, "Snapdragon_CRD_001", transport)
    listener = MeshListener(vault, trusted_device_ids=["Snapdragon_CRD_001"])
    frame = broadcaster.emit_alert(37.7749, -122.4194, 94.5, "DISTRESS_SCREAM")
    print(f"Raw encrypted frame: {len(frame)} bytes; pending: {broadcaster.pending_count}")
    print(f"Local decode: {listener.process_scan(frame)}")
    transport.close()
