import asyncio
import hashlib
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
import torch.nn as nn


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
        max_peers: int = 32,
        heartbeat_interval: float = 5.0,
        mesh_timeout: float = 30.0,
    ):
        super().__init__()
        self.node_id = node_id or str(uuid.uuid4())[:12]
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port
        self.max_peers = max_peers
        self.heartbeat_interval = heartbeat_interval
        self.mesh_timeout = mesh_timeout
        
        self.local_node = MeshNode(
            node_id=self.node_id,
            port=listen_port
        )
        
        self.peers: Dict[str, MeshNode] = {}
        self.peer_lock = threading.RLock()
        
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.daemon_threads: List[threading.Thread] = []
        
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
                
                payload = json.dumps(discovery_msg).encode()
                sock.sendto(payload, ("<broadcast>", self.broadcast_port))
                
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
            sock.sendall(payload)
            sock.close()
            return True
        except Exception as e:
            print(f"[GhostMesh] Send message error: {e}")
            return False
    
    def add_peer(self, peer: MeshNode) -> bool:
        with self.peer_lock:
            if len(self.peers) >= self.max_peers:
                return False
            
            if peer.node_id not in self.peers:
                self.peers[peer.node_id] = peer
                self.stats["peers_discovered"] += 1
                print(f"[GhostMesh] Peer added: {peer.node_id} ({peer.address}:{peer.port})")
                return True
        
        return False
    
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
        
        with self.peer_lock:
            for peer in self.peers.values():
                self._send_message(peer, message)
                self.stats["messages_sent"] += 1
    
    def get_stats(self) -> dict:
        self.stats["uptime"] = time.time() - self.start_time
        self.stats["num_peers"] = len(self.peers)
        return self.stats.copy()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
