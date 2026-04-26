import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from .expert_graph import ExpertGraph, ExpertCluster


class TopologyOptimizer(nn.Module):
    """
    Otimizador de topologia que rearranja o gráfico para maximizar
    fluxo de Conatus e minimizar latência de comunicação.
    """
    
    def __init__(self, graph: ExpertGraph, optimization_steps: int = 100):
        super().__init__()
        self.graph = graph
        self.optimization_steps = optimization_steps
        self.optimization_history: List[dict] = []
    
    def compute_centrality(self, expert_id: str) -> float:
        """Calcula a centralidade de um expert no gráfico"""
        if expert_id not in self.graph.experts:
            return 0.0
        
        incoming = sum(
            1 for (src, tgt) in self.graph.edges.keys() if tgt == expert_id
        )
        outgoing = sum(
            1 for (src, tgt) in self.graph.edges.keys() if src == expert_id
        )
        
        return (incoming + outgoing) / (2 * len(self.graph.experts) + 1e-8)
    
    def compute_betweenness(self, expert_id: str) -> float:
        """Calcula a betweenness centrality (quantas vezes está em caminhos curtos)"""
        if expert_id not in self.graph.experts:
            return 0.0
        
        betweenness = 0.0
        expert_ids = list(self.graph.experts.keys())
        
        for source in expert_ids:
            for target in expert_ids:
                if source != target and source != expert_id and target != expert_id:
                    path = self.graph.get_path_to_expert(source, target)
                    if path and expert_id in path:
                        betweenness += 1.0
        
        return betweenness / (len(expert_ids) ** 2 + 1e-8)
    
    def optimize_cluster_hierarchy(self):
        """Reorganiza a hierarquia de clusters para máxima eficiência"""
        for cluster in self.graph.clusters.values():
            if not cluster.expert_ids:
                continue
            
            centralities = {
                exp_id: self.compute_centrality(exp_id)
                for exp_id in cluster.expert_ids
            }
            
            sorted_experts = sorted(
                centralities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            cluster.expert_ids = [exp_id for exp_id, _ in sorted_experts]
    
    def optimize_edge_weights(self):
        """Ajusta pesos das arestas baseado em tráfego e latência"""
        for edge in self.graph.edges.values():
            if edge.traffic_count > 0:
                edge.update_weight(0.05)
            else:
                edge.update_weight(-0.02)
    
    def detect_bottlenecks(self) -> List[str]:
        """Identifica experts que são gargalos de comunicação"""
        betweenness_scores = {
            exp_id: self.compute_betweenness(exp_id)
            for exp_id in self.graph.experts.keys()
        }
        
        threshold = np.mean(list(betweenness_scores.values())) + np.std(list(betweenness_scores.values()))
        
        return [
            exp_id for exp_id, score in betweenness_scores.items()
            if score > threshold
        ]
    
    def relieve_bottlenecks(self):
        """Cria caminhos alternativos para gargalos"""
        bottlenecks = self.detect_bottlenecks()
        
        for bottleneck_id in bottlenecks:
            incoming_edges = [
                (src, tgt) for src, tgt in self.graph.edges.keys()
                if tgt == bottleneck_id
            ]
            outgoing_edges = [
                (src, tgt) for src, tgt in self.graph.edges.keys()
                if src == bottleneck_id
            ]
            
            for src, _ in incoming_edges:
                for _, tgt in outgoing_edges:
                    if (src, tgt) not in self.graph.edges:
                        self.graph.add_edge(src, tgt, weight=0.5, edge_type="bypass")
    
    def run_optimization(self) -> dict:
        """Executa ciclo completo de otimização"""
        stats = {
            "initial_edges": len(self.graph.edges),
            "initial_clusters": len(self.graph.clusters),
            "optimization_steps": 0,
            "edges_added": 0,
            "edges_removed": 0,
        }
        
        for step in range(self.optimization_steps):
            self.optimize_cluster_hierarchy()
            self.optimize_edge_weights()
            self.relieve_bottlenecks()
            
            if step % 10 == 0:
                mutation_result = self.graph.mutate_topology(mutation_rate=0.05)
                stats["edges_added"] += mutation_result["edges_added"]
                stats["edges_removed"] += mutation_result["edges_removed"]
        
        stats["final_edges"] = len(self.graph.edges)
        stats["final_clusters"] = len(self.graph.clusters)
        self.optimization_history.append(stats)
        
        return stats
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
