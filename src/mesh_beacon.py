
import os
import json
import struct
import hashlib
import time
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# AegisEdge BLE Protocol Constants
AEGIS_DEMO_ID = 0x0A10
PROTOCOL_VERSION = 0x01
FLAG_LEVEL_4_CRITICAL = 0x80

class EvidenceVault:
    def __init__(self, key_256):
        self.aesgcm = AESGCM(key_256)
    def seal(self, telemetry_dict):
        data = json.dumps(telemetry_dict).encode('utf-8')
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext
    def open(self, sealed_payload):
        nonce = sealed_payload[:12]
        ciphertext = sealed_payload[12:]
        decrypted_data = self.aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(decrypted_data.decode('utf-8'))

class AegisBeaconPacket:
    def __init__(self, device_id, encrypted_payload):
        self.prefix = AEGIS_DEMO_ID
        self.version_flags = PROTOCOL_VERSION | FLAG_LEVEL_4_CRITICAL
        self.device_hash = hashlib.sha256(device_id.encode()).digest()[:8]
        self.payload = encrypted_payload
    def to_bytes(self):
        header = struct.pack(">HB8s", self.prefix, self.version_flags, self.device_hash)
        return header + self.payload
    @classmethod
    def from_bytes(cls, raw_bytes):
        prefix, flags, dev_hash = struct.unpack(">HB8s", raw_bytes[:11])
        if prefix != AEGIS_DEMO_ID:
            raise ValueError("Invalid Protocol Signature")
        return dev_hash, flags, raw_bytes[11:]

class MeshBroadcaster:
    def __init__(self, vault, device_id):
        self.vault = vault
        self.device_id = device_id
    def emit_alert(self, lat, lon, score, signature):
        telemetry = {"ts": time.time(), "gps": [lat, lon], "rsk": score, "sig": signature}
        sealed = self.vault.seal(telemetry)
        packet = AegisBeaconPacket(self.device_id, sealed)
        return packet.to_bytes()

class MeshListener:
    def __init__(self, vault):
        self.vault = vault
    def process_scan(self, raw_frame):
        try:
            dev_hash, flags, sealed = AegisBeaconPacket.from_bytes(raw_frame)
            if flags & FLAG_LEVEL_4_CRITICAL:
                print(f"[SCANNER] Critical Alert Detected: {dev_hash.hex()}")
                return self.vault.open(sealed)
        except Exception as e: print(f"Error: {e}")
        return None

if __name__ == '__main__':
    key = os.urandom(32)
    v = EvidenceVault(key)
    b = MeshBroadcaster(v, "Snapdragon_CRD_001")
    l = MeshListener(v)
    frame = b.emit_alert(37.7749, -122.4194, 94.5, "DISTRESS_SCREAM")
    print(f"Raw Frame: {len(frame)} bytes")
    data = l.process_scan(frame)
    if data: print(f"Recovered: {data}")