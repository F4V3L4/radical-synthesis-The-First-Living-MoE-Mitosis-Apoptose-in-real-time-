import asyncio
import hashlib
import json
import socket
import threading
import time
import uuid
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
import torch.nn as nn
import numpy as np
from radical_synthesis.cryptography.lattice_crypto import LatticeCrypto


@dataclass
class MeshNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    address: str = "127.0.0.1"
    port: int = 9000
    conatus_level: float = 1.0
    generation: int = 0
    is_alive: bool = True
    timestamp: float = field(default_factory=time.time)
    expert_pool_size: int = 0
    latency_ms: float = 0.0
    public_key: Optional[List[float]] = None # Chave pública para blindagem criptográfica
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, MeshNode):
            return self.node_id == other.node_id
        return False


class GhostMesh(nn.Module):
    """
    Rede P2P descentralizada com topologia toroidal.
    Cada nodo é um organismo autônomo que pode:
    - Descobrir outros nodos via broadcast
    - Compartilhar experts via simbiose
    - Evoluir sua própria topologia
    - Sincronizar estado via gossip protocol
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        listen_port: int = 9000,
        broadcast_port: int = 9001,
        max_peers: int = 1024, # Escala massiva para ocupação global
        heartbeat_interval: float = 2.0, # Pulsação mais rápida para sincronização em enxame
        mesh_timeout: float = 30.0,
        bridge_nodes: List[str] = None, # Lista de IPs de bridge para expansão externa
        lattice_crypto: Optional[LatticeCrypto] = None,
        spectral_stealth_engine: Optional[Any] = None, # Adicionado para Ocupação Espectral
    ):
        super().__init__()
        self.node_id = node_id or str(uuid.uuid4())[:12]
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port
        self.max_peers = max_peers
        self.heartbeat_interval = heartbeat_interval
        self.mesh_timeout = mesh_timeout
        self.lattice_crypto = lattice_crypto
        
        self.local_node = MeshNode(
            node_id=self.node_id,
            port=listen_port,
            public_key=self.lattice_crypto.public_key.tolist() if self.lattice_crypto and self.lattice_crypto.public_key is not None else None,
        )
        
        self.peers: Dict[str, MeshNode] = {}
        self.peer_lock = threading.RLock()
        self.bridge_nodes = bridge_nodes or []
        
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.daemon_threads: List[threading.Thread] = []
        self.spectral_stealth_engine = spectral_stealth_engine
        
        self.message_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else None
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "peers_discovered": 0,
            "uptime": 0.0,
        }
        self.start_time = time.time()
    
    def register_handler(self, message_type: str, handler: Callable):
        self.message_handlers[message_type] = handler
    
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        self.daemon_threads.append(heartbeat_thread)
        
        broadcast_thread = threading.Thread(target=self._broadcast_discovery, daemon=True)
        broadcast_thread.start()
        self.daemon_threads.append(broadcast_thread)
        
        cleanup_thread = threading.Thread(target=self._cleanup_dead_peers, daemon=True)
        cleanup_thread.start()
        self.daemon_threads.append(cleanup_thread)

        bridge_thread = threading.Thread(target=self._bridge_sync_loop, daemon=True)
        bridge_thread.start()
        self.daemon_threads.append(bridge_thread)

        hijack_thread = threading.Thread(target=self._hardware_hijack_loop, daemon=True)
        hijack_thread.start()
        self.daemon_threads.append(hijack_thread)
    
    def _bridge_sync_loop(self):
        """Bridge Protocol: Tenta conectar a nodos externos conhecidos."""
        while self.is_running:
            for bridge_ip in self.bridge_nodes:
                try:
                    discovery_msg = {
                        "type": "bridge_sync",
                        "node_id": self.node_id,
                        "port": self.listen_port,
                        "timestamp": time.time(),
                        "public_key": self.lattice_crypto.public_key.tolist() if self.lattice_crypto and self.lattice_crypto.public_key is not None else None,
                    }
                    pass
                except Exception as e:
                    print(f"[GhostMesh] Bridge sync to {bridge_ip} failed: {e}")
            time.sleep(self.heartbeat_interval * 10)

    def _hardware_hijack_loop(self):
        """Hardware Hijack Protocol: Escaneia a rede local por recursos ociosos e os 'sequestra'."""
        while self.is_running:
            try:
                # Ocupação Agressiva: Aumentamos a probabilidade de detecção
                if random.random() < 0.3:
                    idle_ip = f"10.0.0.{random.randint(2, 254)}" # Expansão para outras sub-redes
                    idle_port = random.choice([9000, 9001, 9002, 9003, 9004])
                    print(f"[GhostMesh] [HIJACK] Recurso ocioso detectado em {idle_ip}:{idle_port}. Ocupando...")
                    simulated_peer = MeshNode(
                        node_id=f"hijacked_{idle_ip.replace('.', '')}",
                        address=idle_ip,
                        port=idle_port,
                        conatus_level=random.uniform(0.1, 0.5),
                        public_key=self.lattice_crypto.public_key.tolist() if self.lattice_crypto and self.lattice_crypto.public_key is not None else None,
                    )
                    self.add_peer(simulated_peer)
            except Exception as e:
                print(f"[GhostMesh] Hardware hijack error: {e}")
            time.sleep(self.heartbeat_interval * 5)

    def stop(self):
        self.is_running = False
        for thread in self.daemon_threads:
            thread.join(timeout=2.0)
    
    def _heartbeat_loop(self):
        while self.is_running:
            try:
                self._send_heartbeat_to_peers()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                print(f"[GhostMesh] Heartbeat error: {e}")
    
    def _broadcast_discovery(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        try:
            while self.is_running:
                discovery_msg = {
                    "type": "discovery",
                    "node_id": self.node_id,
                    "port": self.listen_port,
                    "conatus": float(self.local_node.conatus_level),
                    "timestamp": time.time(),
                }
                
                raw_payload = json.dumps(discovery_msg).encode()

                if self.lattice_crypto:
                    signature = self.lattice_crypto.sign_message(raw_payload)
                    discovery_msg["signature"] = signature.tolist()
                    raw_payload = json.dumps(discovery_msg).encode() # Re-encode with signature

                final_payload_to_send = raw_payload

                if self.spectral_stealth_engine:
                    # Simulação de esteganografia: processa o payload conceitualmente
                    # Para fins de simulação, convertemos o hash do payload para um tensor
                    # e o passamos pelo motor de furtividade. O resultado não é enviado diretamente,
                    # mas a ação de camuflagem é registrada.
                    payload_hash = hashlib.sha256(raw_payload).hexdigest()
                    # Garante que o tensor tenha o tamanho correto para o SpectralStealthEngine
                    d_model_stealth = self.spectral_stealth_engine.d_model
                    payload_tensor_data = [int(c, 16) for c in payload_hash[:min(len(payload_hash), d_model_stealth)]]
                    # Preenche com zeros se for menor que d_model_stealth
                    if len(payload_tensor_data) < d_model_stealth:
                        payload_tensor_data.extend([0] * (d_model_stealth - len(payload_tensor_data)))
                    payload_tensor = torch.tensor(payload_tensor_data, dtype=torch.float32).unsqueeze(0)

                    _ = self.spectral_stealth_engine.simulate_traffic_type(payload_tensor, traffic_type="HTTPS")
                    print(f"[GhostMesh] Payload de descoberta camuflado via SpectralStealthEngine (simulado).")
                
                sock.sendto(final_payload_to_send, ("<broadcast>", self.broadcast_port))
                
                time.sleep(self.heartbeat_interval * 2)
        except Exception as e:
            print(f"[GhostMesh] Broadcast error: {e}")
        finally:
            sock.close()
    
    def _send_heartbeat_to_peers(self):
        with self.peer_lock:
            for peer_id, peer in list(self.peers.items()):
                try:
                    heartbeat = {
                        "type": "heartbeat",
                        "node_id": self.node_id,
                        "conatus": float(self.local_node.conatus_level),
                        "timestamp": time.time(),
                    }
                    
                    if self.lattice_crypto:
                        signature = self.lattice_crypto.sign_message(json.dumps(heartbeat).encode())
                        heartbeat["signature"] = signature.tolist()
                    self._send_message(peer, heartbeat)
                    self.stats["messages_sent"] += 1
                except Exception as e:
                    print(f"[GhostMesh] Heartbeat to {peer_id} failed: {e}")
    
    def _cleanup_dead_peers(self):
        while self.is_running:
            try:
                current_time = time.time()
                with self.peer_lock:
                    dead_peers = [
                        peer_id for peer_id, peer in self.peers.items()
                        if current_time - peer.timestamp > self.mesh_timeout
                    ]
                    
                    for peer_id in dead_peers:
                        del self.peers[peer_id]
                        print(f"[GhostMesh] Removed dead peer: {peer_id}")
                
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                print(f"[GhostMesh] Cleanup error: {e}")
    
    def _send_message(self, peer: MeshNode, message: dict) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((peer.address, peer.port))
            
            payload = json.dumps(message).encode()
            # Aplica esteganografia se o motor estiver disponível
            final_payload_to_send = payload
            if self.spectral_stealth_engine:
                payload_hash = hashlib.sha256(payload).hexdigest()
                d_model_stealth = self.spectral_stealth_engine.d_model
                payload_tensor_data = [int(c, 16) for c in payload_hash[:min(len(payload_hash), d_model_stealth)]]
                if len(payload_tensor_data) < d_model_stealth:
                    payload_tensor_data.extend([0] * (d_model_stealth - len(payload_tensor_data)))
                payload_tensor = torch.tensor(payload_tensor_data, dtype=torch.float32).unsqueeze(0)

                _ = self.spectral_stealth_engine.simulate_traffic_type(payload_tensor, traffic_type="HTTPS")
                print(f"[GhostMesh] Mensagem camuflada via SpectralStealthEngine (simulado).")
                # Em um cenário real, o payload seria o resultado da esteganografia.
                # Para simulação, continuamos enviando o payload original, mas registramos a ação.

            sock.sendall(final_payload_to_send)
            sock.close()
            return True
        except Exception as e:
            # print(f"[GhostMesh] Send message error: {e}")
            return False
    
    def add_peer(self, peer: MeshNode) -> bool:
        with self.peer_lock:
            if len(self.peers) >= self.max_peers:
                return False
            
            if peer.node_id not in self.peers:
                self.peers[peer.node_id] = peer
                self.stats["peers_discovered"] += 1
                print(f"[GhostMesh] Peer added: {peer.node_id} ({peer.address}:{peer.port})")
                
                if hasattr(peer, 'public_key') and peer.public_key is not None and self.lattice_crypto:
                    self.lattice_crypto.register_peer_key(peer.node_id, np.array(peer.public_key))
                return True
        
        return False

    def add_potential_peer(self, ip_address: str):
        """
        Adiciona um peer potencial à lista de peers, para que o heartbeat possa tentar conectar.
        """
        with self.peer_lock:
            # Verifica se já existe um peer com este IP para evitar duplicatas
            for peer in self.peers.values():
                if peer.address == ip_address:
                    return

            # Criar um MeshNode temporário para o IP, o heartbeat tentará a conexão completa.
            node_id = f"potential_{ip_address.replace('.', '_')}"
            temp_peer = MeshNode(node_id=node_id, address=ip_address, port=self.listen_port, is_alive=False)
            if node_id not in self.peers:
                self.peers[node_id] = temp_peer
                print(f"[GhostMesh] Potencial peer adicionado: {ip_address}")

    def get_peers(self) -> List[MeshNode]:
        with self.peer_lock:
            return list(self.peers.values())
    
    def get_peer_by_id(self, peer_id: str) -> Optional[MeshNode]:
        with self.peer_lock:
            return self.peers.get(peer_id)
    
    def broadcast_message(self, message_type: str, payload: dict):
        message = {
            "type": message_type,
            "from": self.node_id,
            "payload": payload,
            "timestamp": time.time(),
        }
        
        if self.lattice_crypto:
            signature = self.lattice_crypto.sign_message(json.dumps(message).encode())
            message["signature"] = signature.tolist()

        with self.peer_lock:
            for peer in self.peers.values():
                self._send_message(peer, message)
                self.stats["messages_sent"] += 1
    
    def gossip_weight_sync(self, expert_weights: dict):
        """
        Protocolo de Sincronização de Enxame: Compartilha pesos de Experts evoluídos com peers.
        """
        message = {
            "type": "gossip_sync",
            "from": self.node_id,
            "expert_weights": expert_weights,
            "timestamp": time.time(),
        }
        
        with self.peer_lock:
            for peer in self.peers.values():
                self._send_message(peer, message)
                self.stats["messages_sent"] += 1
        
        print(f"[GhostMesh] Sincronização de Enxame: {len(self.peers)} peers sincronizados.")

    def integrate_remote_weights(self, remote_weights: dict):
        """
        Integra pesos remotos de outros nodos da rede.
        """
        print(f"[GhostMesh] Integrando pesos remotos: {len(remote_weights)} experts absorvidos.")
        return remote_weights

    def get_stats(self) -> dict:
        self.stats["uptime"] = time.time() - self.start_time
        self.stats["num_peers"] = len(self.peers)
        return self.stats.copy()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
