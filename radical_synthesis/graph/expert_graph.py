import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import uuid
import time


@dataclass
class ExpertCluster:
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    expert_ids: List[str] = field(default_factory=list)
    specialization: str = ""
    hierarchy_level: int = 0
    parent_cluster_id: Optional[str] = None
    child_clusters: List[str] = field(default_factory=list)
    conatus_aggregate: float = 1.0
    created_at: float = field(default_factory=time.time)
    
    def add_expert(self, expert_id: str):
        if expert_id not in self.expert_ids:
            self.expert_ids.append(expert_id)
    
    def remove_expert(self, expert_id: str):
        if expert_id in self.expert_ids:
            self.expert_ids.remove(expert_id)


class GraphEdge:
    def __init__(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        edge_type: str = "resonance"
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.edge_type = edge_type
        self.traffic_count = 0
        self.last_used = time.time()
    
    def update_weight(self, delta: float):
        self.weight = max(0.1, min(1.0, self.weight + delta))


class ExpertGraph(nn.Module):
    """
    Grafo dinâmico de Experts com topologia fractal.
    Permite que a AGI reescreva o roteamento em tempo de execução,
    criando clusters hierárquicos que se comunicam via ressonância.
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.experts: Dict[str, nn.Module] = {}
        self.clusters: Dict[str, ExpertCluster] = {}
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}
        self.adjacency_matrix = None
        self.mutation_history: List[dict] = []
    
    def add_expert(self, expert_id: str, expert: nn.Module):
        self.experts[expert_id] = expert
        self._rebuild_adjacency()
    
    def remove_expert(self, expert_id: str):
        if expert_id in self.experts:
            del self.experts[expert_id]
            
            edges_to_remove = [
                (s, t) for s, t in self.edges.keys()
                if s == expert_id or t == expert_id
            ]
            for edge_key in edges_to_remove:
                del self.edges[edge_key]
            
            self._rebuild_adjacency()
    
    def create_cluster(
        self,
        specialization: str,
        expert_ids: List[str],
        hierarchy_level: int = 0,
        parent_cluster_id: Optional[str] = None
    ) -> ExpertCluster:
        cluster = ExpertCluster(
            specialization=specialization,
            expert_ids=expert_ids,
            hierarchy_level=hierarchy_level,
            parent_cluster_id=parent_cluster_id
        )
        
        self.clusters[cluster.cluster_id] = cluster
        
        if parent_cluster_id and parent_cluster_id in self.clusters:
            parent = self.clusters[parent_cluster_id]
            parent.child_clusters.append(cluster.cluster_id)
        
        return cluster
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        edge_type: str = "resonance"
    ):
        if source_id not in self.experts or target_id not in self.experts:
            return
        
        edge_key = (source_id, target_id)
        self.edges[edge_key] = GraphEdge(source_id, target_id, weight, edge_type)
        self._rebuild_adjacency()
    
    def remove_edge(self, source_id: str, target_id: str):
        edge_key = (source_id, target_id)
        if edge_key in self.edges:
            del self.edges[edge_key]
            self._rebuild_adjacency()
    
    def _rebuild_adjacency(self):
        n = len(self.experts)
        if n == 0:
            self.adjacency_matrix = None
            return
        
        expert_ids = list(self.experts.keys())
        adj = np.zeros((n, n), dtype=np.float32)
        
        for (src, tgt), edge in self.edges.items():
            try:
                i = expert_ids.index(src)
                j = expert_ids.index(tgt)
                adj[i, j] = edge.weight
            except ValueError:
                pass
        
        self.adjacency_matrix = torch.from_numpy(adj)
    
    def mutate_topology(self, mutation_rate: float = 0.1) -> dict:
        """
        Reescreve o gráfico de roteamento via mutação estrutural.
        Cria novas conexões, remove as fracas e reorganiza clusters.
        """
        mutation_log = {
            "timestamp": time.time(),
            "edges_added": 0,
            "edges_removed": 0,
            "clusters_merged": 0,
            "clusters_split": 0,
        }
        
        expert_ids = list(self.experts.keys())
        if len(expert_ids) < 2:
            return mutation_log
        
        num_mutations = max(1, int(len(self.edges) * mutation_rate))
        
        for _ in range(num_mutations):
            if np.random.random() < 0.5:
                src = np.random.choice(expert_ids)
                tgt = np.random.choice(expert_ids)
                if src != tgt and (src, tgt) not in self.edges:
                    weight = np.random.uniform(0.3, 1.0)
                    self.add_edge(src, tgt, weight)
                    mutation_log["edges_added"] += 1
            else:
                if self.edges:
                    edge_key = np.random.choice(list(self.edges.keys()))
                    self.remove_edge(edge_key[0], edge_key[1])
                    mutation_log["edges_removed"] += 1
        
        self._reorganize_clusters()
        mutation_log["clusters_merged"] = self._merge_weak_clusters()
        mutation_log["clusters_split"] = self._split_overloaded_clusters()
        
        self.mutation_history.append(mutation_log)
        return mutation_log
    
    def _reorganize_clusters(self):
        for cluster in self.clusters.values():
            conatus_sum = 0.0
            for expert_id in cluster.expert_ids:
                if hasattr(self.experts[expert_id], 'vitality'):
                    conatus_sum += self.experts[expert_id].vitality
            
            if cluster.expert_ids:
                cluster.conatus_aggregate = conatus_sum / len(cluster.expert_ids)
    
    def _merge_weak_clusters(self) -> int:
        merged_count = 0
        clusters_to_merge = [
            cid for cid, cluster in self.clusters.items()
            if cluster.conatus_aggregate < 0.3 and cluster.parent_cluster_id
        ]
        
        for cluster_id in clusters_to_merge:
            cluster = self.clusters[cluster_id]
            parent = self.clusters.get(cluster.parent_cluster_id)
            
            if parent:
                for expert_id in cluster.expert_ids:
                    parent.add_expert(expert_id)
                
                del self.clusters[cluster_id]
                parent.child_clusters.remove(cluster_id)
                merged_count += 1
        
        return merged_count
    
    def _split_overloaded_clusters(self) -> int:
        split_count = 0
        clusters_to_split = [
            cid for cid, cluster in self.clusters.items()
            if len(cluster.expert_ids) > 8 and cluster.conatus_aggregate > 0.8
        ]
        
        for cluster_id in clusters_to_split:
            cluster = self.clusters[cluster_id]
            mid = len(cluster.expert_ids) // 2
            
            new_cluster = self.create_cluster(
                specialization=cluster.specialization + "_split",
                expert_ids=cluster.expert_ids[mid:],
                hierarchy_level=cluster.hierarchy_level,
                parent_cluster_id=cluster.parent_cluster_id
            )
            
            cluster.expert_ids = cluster.expert_ids[:mid]
            split_count += 1
        
        return split_count
    
    def get_cluster_by_specialization(self, specialization: str) -> Optional[ExpertCluster]:
        for cluster in self.clusters.values():
            if cluster.specialization == specialization:
                return cluster
        return None
    
    def get_path_to_expert(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """BFS para encontrar caminho entre experts"""
        if source_id not in self.experts or target_id not in self.experts:
            return None
        
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_id:
                return path
            
            for (src, tgt), edge in self.edges.items():
                if src == current and tgt not in visited:
                    visited.add(tgt)
                    queue.append((tgt, path + [tgt]))
        
        return None
    
    def forward(self, x: torch.Tensor, source_expert_id: str, target_expert_id: str) -> torch.Tensor:
        path = self.get_path_to_expert(source_expert_id, target_expert_id)
        
        if not path or len(path) < 2:
            return x
        
        output = x
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            
            if current_id in self.experts:
                expert = self.experts[current_id]
                output = expert(output)
            
            edge_key = (current_id, next_id)
            if edge_key in self.edges:
                edge = self.edges[edge_key]
                edge.traffic_count += 1
                edge.last_used = time.time()
        
        return output


class GraphMutation(nn.Module):
    """
    Controlador de mutação estrutural do gráfico.
    Aplica seleção natural e evolução ao roteamento.
    """
    
    def __init__(self, graph: ExpertGraph):
        super().__init__()
        self.graph = graph
        self.fitness_history: List[float] = []
    
    def evaluate_fitness(self) -> float:
        if not self.graph.edges:
            return 0.0
        
        total_traffic = sum(edge.traffic_count for edge in self.graph.edges.values())
        if total_traffic == 0:
            return 0.5
        
        avg_traffic = total_traffic / len(self.graph.edges)
        variance = np.var([edge.traffic_count for edge in self.graph.edges.values()])
        
        balance = 1.0 / (1.0 + variance)
        efficiency = min(1.0, avg_traffic / 100.0)
        
        fitness = 0.6 * balance + 0.4 * efficiency
        self.fitness_history.append(fitness)
        return fitness
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
