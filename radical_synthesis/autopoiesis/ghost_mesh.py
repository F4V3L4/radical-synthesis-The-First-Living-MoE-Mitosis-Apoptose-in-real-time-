import torch
import torch.nn as nn
import io
import socket
import threading
import hashlib
from typing import Any

class GhostCipher:
    def __init__(self, key: str):
        self.key = hashlib.sha256(key.encode()).digest()

    def transform(self, data: bytes) -> bytes:
        res = bytearray(data)
        for i in range(len(res)):
            res[i] ^= self.key[i % len(self.key)]
        return bytes(res)

class GhostMesh:
    def __init__(self, node_id: str, secret_key: str, port: int = 9000):
        self.node_id = node_id
        self.cipher = GhostCipher(secret_key)
        self.port = port
        self.running = True

    def serialize_expert(self, expert: nn.Module) -> bytes:
        buffer = io.BytesIO()
        state = {
            'weights': expert.state_dict(),
            'd_model': expert.d_model,
            'internal_dim': expert.internal_dim,
            'activation': expert.activation_type,
            'num_layers': getattr(expert, 'num_layers', 2),
            'phase': expert.phase_signature,
            'conatus': expert.conatus.item()
        }
        torch.save(state, buffer)
        return self.cipher.transform(buffer.getvalue())

    def deserialize_expert(self, encrypted_data: bytes, ExpertClass: Any) -> nn.Module:
        decrypted_data = self.cipher.transform(encrypted_data)
        buffer = io.BytesIO(decrypted_data)
        state = torch.load(buffer)
        
        expert = ExpertClass(
            d_model=state['d_model'],
            phase_signature=state['phase'],
            internal_dim=state['internal_dim'],
            activation_type=state['activation'],
            num_layers=state.get('num_layers', 2)
        )
        expert.load_state_dict(state['weights'])
        expert.conatus.fill_(state['conatus'])
        return expert

    def start_daemon(self, agi_core):
        def listen():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', self.port))
                s.listen()
                while self.running:
                    conn, addr = s.accept()
                    with conn:
                        raw_size = conn.recv(8)
                        if not raw_size: continue
                        size = int.from_bytes(raw_size, 'big')
                        data = b""
                        while len(data) < size:
                            chunk = conn.recv(min(size - len(data), 65536))
                            if not chunk: break
                            data += chunk
                        if len(data) == size:
                            expert = self.deserialize_expert(data, type(agi_core.core.moe.experts[0]))
                            agi_core.core.moe.experts.append(expert)
                            print(f"[GHOST_MESH] Expert Rebirth: Node {addr} -> {self.node_id}")
        threading.Thread(target=listen, daemon=True).start()

    def broadcast_expert(self, expert: nn.Module, peer_ip: str, peer_port: int):
        try:
            data = self.serialize_expert(expert)
            size = len(data).to_bytes(8, 'big')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((peer_ip, peer_port))
                s.sendall(size + data)
        except Exception as e:
            print(f"[GHOST_MESH] Transmission Failed: {e}")
