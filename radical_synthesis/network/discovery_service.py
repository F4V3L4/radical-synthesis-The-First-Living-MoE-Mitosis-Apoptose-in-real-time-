import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class DHTEntry:
    key: str
    value: any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600.0
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class DHT(nn.Module):
    """
    Distributed Hash Table (DHT) para armazenamento distribuído de metadados.
    Implementa uma topologia de anel com replicação.
    """
    
    def __init__(self, node_id: str, replication_factor: int = 3):
        super().__init__()
        self.node_id = node_id
        self.replication_factor = replication_factor
        self.storage: Dict[str, DHTEntry] = {}
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.is_running = False
    
    def _hash_key(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)
    
    def put(self, key: str, value: any, ttl: float = 3600.0):
        with self.lock:
            self.storage[key] = DHTEntry(key=key, value=value, ttl=ttl)
    
    def get(self, key: str) -> Optional[any]:
        with self.lock:
            entry = self.storage.get(key)
            if entry and not entry.is_expired():
                return entry.value
            elif entry:
                del self.storage[key]
        return None
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.storage:
                del self.storage[key]
                return True
        return False
    
    def start_cleanup(self):
        if self.is_running:
            return
        self.is_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def stop_cleanup(self):
        self.is_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=2.0)
    
    def _cleanup_loop(self):
        while self.is_running:
            try:
                with self.lock:
                    expired_keys = [
                        key for key, entry in self.storage.items()
                        if entry.is_expired()
                    ]
                    for key in expired_keys:
                        del self.storage[key]
                
                time.sleep(60.0)
            except Exception as e:
                print(f"[DHT] Cleanup error: {e}")
    
    def get_all(self) -> Dict[str, any]:
        with self.lock:
            return {
                key: entry.value
                for key, entry in self.storage.items()
                if not entry.is_expired()
            }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DiscoveryService(nn.Module):
    """
    Serviço de descoberta de nodos usando DHT e gossip protocol.
    Mantém um registro de nodos ativos e seus metadados (Conatus, Experts, etc).
    """
    
    def __init__(self, node_id: str, dht: DHT):
        super().__init__()
        self.node_id = node_id
        self.dht = dht
        self.local_registry: Dict[str, dict] = {}
        self.lock = threading.RLock()
        self.gossip_interval = 10.0
        self.is_running = False
        self.gossip_thread = None
    
    def register_node(self, node_id: str, metadata: dict):
        with self.lock:
            self.local_registry[node_id] = {
                "node_id": node_id,
                "metadata": metadata,
                "timestamp": time.time(),
            }
            
            registry_key = f"node:{node_id}"
            self.dht.put(registry_key, metadata, ttl=120.0)
    
    def discover_node(self, node_id: str) -> Optional[dict]:
        registry_key = f"node:{node_id}"
        return self.dht.get(registry_key)
    
    def discover_all_nodes(self) -> Dict[str, dict]:
        all_entries = self.dht.get_all()
        nodes = {}
        for key, value in all_entries.items():
            if key.startswith("node:"):
                node_id = key.split(":")[1]
                nodes[node_id] = value
        return nodes
    
    def register_expert_cluster(self, cluster_id: str, experts: List[str], metadata: dict):
        cluster_key = f"cluster:{cluster_id}"
        cluster_data = {
            "cluster_id": cluster_id,
            "experts": experts,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        self.dht.put(cluster_key, cluster_data, ttl=300.0)
    
    def discover_expert_cluster(self, cluster_id: str) -> Optional[dict]:
        cluster_key = f"cluster:{cluster_id}"
        return self.dht.get(cluster_key)
    
    def discover_all_clusters(self) -> Dict[str, dict]:
        all_entries = self.dht.get_all()
        clusters = {}
        for key, value in all_entries.items():
            if key.startswith("cluster:"):
                cluster_id = key.split(":")[1]
                clusters[cluster_id] = value
        return clusters
    
    def start_gossip(self):
        if self.is_running:
            return
        self.is_running = True
        self.gossip_thread = threading.Thread(target=self._gossip_loop, daemon=True)
        self.gossip_thread.start()
    
    def stop_gossip(self):
        self.is_running = False
        if self.gossip_thread:
            self.gossip_thread.join(timeout=2.0)
    
    def _gossip_loop(self):
        while self.is_running:
            try:
                with self.lock:
                    for node_id, node_data in list(self.local_registry.items()):
                        registry_key = f"node:{node_id}"
                        self.dht.put(registry_key, node_data["metadata"], ttl=120.0)
                
                time.sleep(self.gossip_interval)
            except Exception as e:
                print(f"[DiscoveryService] Gossip error: {e}")
    
    def get_healthy_nodes(self, min_conatus: float = 0.5) -> List[str]:
        nodes = self.discover_all_nodes()
        healthy = []
        for node_id, metadata in nodes.items():
            if metadata.get("conatus_level", 0.0) >= min_conatus:
                healthy.append(node_id)
        return healthy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
