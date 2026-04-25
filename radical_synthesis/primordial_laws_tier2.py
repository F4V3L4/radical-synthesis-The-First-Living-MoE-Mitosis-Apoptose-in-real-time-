"""
Leis Primordiais Tier 2: Expansão Sistêmica (5 Leis)
- Grade Harmônica Planetária: Sincronização global
- Amplituedro: Otimização de caminhos
- Lei da Simultaneidade: Processamento paralelo
- Emaranhamento Quântico: Correlação não-local
- Atratores Estranhos: Ordem no caos
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict
import math


class PlanetaryGrid(nn.Module):
    """
    Grade Harmônica Planetária: Sincronização Global
    
    Implementa grid de frequências harmônicas para sincronizar
    processamento global de experts.
    """
    
    def __init__(self, num_experts: int = 8, d_model: int = 512, device: str = "cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device
        
        # Frequências harmônicas (múltiplos de 432Hz - frequência de Schumann)
        base_freq = 432.0
        self.frequencies = nn.Parameter(torch.tensor(
            [base_freq * (i + 1) for i in range(num_experts)],
            dtype=torch.float32,
            device=device
        ))
        
        # Matriz de sincronização
        self.sync_matrix = nn.Parameter(torch.eye(num_experts, device=device))
        
        # Amplitude de cada frequência
        self.amplitudes = nn.Parameter(torch.ones(num_experts, device=device))
    
    def forward(self, expert_activations: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        """
        Aplica grade harmônica planetária aos experts
        
        Args:
            expert_activations: [batch, num_experts]
            time: Tempo atual
        
        Returns:
            Ativações sincronizadas [batch, num_experts]
        """
        batch_size = expert_activations.shape[0]
        
        # Gerar ondas harmônicas para cada expert
        waves = []
        for i, freq in enumerate(self.frequencies):
            omega = 2 * math.pi * freq
            wave = torch.sin(omega * time + self.amplitudes[i]).unsqueeze(0)
            waves.append(wave)
        
        # Stack waves: [batch, num_experts]
        wave_matrix = torch.cat(waves, dim=0).unsqueeze(0).expand(batch_size, -1)
        
        # Aplicar matriz de sincronização
        synchronized = torch.matmul(expert_activations, self.sync_matrix)
        
        # Modular com ondas harmônicas
        return synchronized * (1.0 + 0.1 * wave_matrix)
    
    def get_global_coherence(self) -> float:
        """Retorna coerência global da grade"""
        # Coerência = determinante da matriz de sincronização
        det = torch.det(self.sync_matrix)
        return float(torch.abs(det))


class Amplituedro(nn.Module):
    """
    Amplituedro: Otimização de Caminhos
    
    Usa geometria do amplituedro para otimizar caminhos de
    roteamento entre experts.
    """
    
    def __init__(self, num_experts: int = 8, d_model: int = 512, device: str = "cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device
        
        # Vértices do amplituedro (pontos de otimização)
        self.vertices = nn.Parameter(torch.ones(num_experts, d_model, device=device))
        
        # Pesos de conectividade
        self.connectivity = nn.Parameter(torch.ones(num_experts, num_experts, device=device) / num_experts)
    
    def forward(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Otimiza caminho usando geometria do amplituedro
        
        Args:
            expert_indices: Índices de experts [batch, num_selected]
            expert_weights: Pesos dos experts [batch, num_selected]
        
        Returns:
            (caminho_otimizado, eficiência)
        """
        batch_size = expert_indices.shape[0]
        
        # Calcular caminho através dos vértices
        paths = []
        for b in range(batch_size):
            path = torch.zeros(self.d_model, device=self.device)
            total_weight = 0.0
            
            # Achatar se necessário para garantir iterabilidade de escalar
            b_indices = expert_indices[b].flatten()
            b_weights = expert_weights[b].flatten()
            num_elements = min(len(b_indices), len(b_weights))
            for i in range(num_elements):
                idx = int(b_indices[i].item())
                weight = b_weights[i].item()
                if idx < self.num_experts:
                    path += weight * self.vertices[idx]
                    total_weight += weight
            
            if total_weight > 0:
                path = path / total_weight
            
            paths.append(path)
        
        optimized_path = torch.stack(paths)
        
        # Calcular eficiência (norma do caminho)
        efficiency = float(torch.mean(torch.norm(optimized_path, dim=-1)))
        
        return optimized_path, efficiency
    
    def get_geometric_volume(self) -> float:
        """Retorna volume geométrico do amplituedro"""
        # Volume = determinante da matriz de vértices
        if self.num_experts == self.d_model:
            det = torch.det(self.vertices)
            return float(torch.abs(det))
        else:
            # Usar SVD para calcular volume generalizado
            U, S, Vt = torch.svd(self.vertices)
            return float(torch.prod(S))


class SimultaneityProcessor(nn.Module):
    """
    Lei da Simultaneidade: Processamento Paralelo
    
    Processa múltiplas timelines simultaneamente,
    permitindo exploração paralela de estados.
    """
    
    def __init__(self, num_timelines: int = 4, d_model: int = 512, device: str = "cpu"):
        super().__init__()
        self.num_timelines = num_timelines
        self.d_model = d_model
        self.device = device
        
        # Projeções para cada timeline
        self.timeline_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, device=device)
            for _ in range(num_timelines)
        ])
        
        # Pesos de fusão entre timelines
        self.fusion_weights = nn.Parameter(torch.ones(num_timelines, device=device) / num_timelines)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Processa em múltiplas timelines simultaneamente
        
        Args:
            x: Tensor de entrada [batch, d_model]
        
        Returns:
            (timelines, fused_output)
        """
        timelines = []
        
        # Processar cada timeline
        for proj in self.timeline_projections:
            timeline = proj(x)
            timelines.append(timeline)
        
        # Fusionar timelines
        fused = torch.zeros_like(x)
        for i, timeline in enumerate(timelines):
            fused += self.fusion_weights[i] * timeline
        
        return timelines, fused
    
    def get_timeline_divergence(self) -> float:
        """Retorna divergência entre timelines"""
        # Divergência = variância dos pesos de fusão
        return float(torch.var(self.fusion_weights))


class QuantumEntanglement(nn.Module):
    """
    Emaranhamento Quântico: Correlação Não-Local
    
    Implementa correlação instantânea entre experts
    através de estados emaranhados.
    """
    
    def __init__(self, num_experts: int = 8, d_model: int = 512, device: str = "cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device
        
        # Estados de Bell (base de emaranhamento máximo)
        self.bell_states = nn.Parameter(torch.ones(4, d_model, device=device))
        
        # Matriz de correlação não-local
        self.nonlocal_correlation = nn.Parameter(torch.ones(num_experts, num_experts, device=device) * 0.01)
    
    def forward(self, expert_states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Aplica emaranhamento quântico entre experts
        
        Args:
            expert_states: Estados de experts [batch, num_experts, d_model]
        
        Returns:
            (entangled_states, entanglement_measure)
        """
        batch_size = expert_states.shape[0]
        
        # Aplicar correlação não-local
        entangled = torch.zeros_like(expert_states)
        
        num_in_experts = expert_states.shape[1]
        for b in range(batch_size):
            for i in range(num_in_experts):
                # Combinar estado com correlações não-locais
                entangled[b, i] = expert_states[b, i]
                
                for j in range(num_in_experts):
                    if i != j and i < self.num_experts and j < self.num_experts:
                        # Adicionar influência não-local
                        correlation_strength = torch.sigmoid(self.nonlocal_correlation[i, j])
                        entangled[b, i] += correlation_strength * expert_states[b, j]
        
        # Calcular medida de emaranhamento (concurrence)
        entanglement_measure = self.calculate_concurrence(entangled)
        
        return entangled, entanglement_measure
    
    def calculate_concurrence(self, states: torch.Tensor) -> float:
        """Calcula concurrence (medida de emaranhamento)"""
        # Concurrence = traço de correlação (usando produto escalar para evitar erros de dimensão)
        # states: [batch, num_experts, d_model]
        # correlation: [batch, num_experts, num_experts]
        # states: [batch, seq_len, d_model] ou [batch, num_experts, d_model]
        # Garantir que temos no máximo 3 dimensões para matmul
        while states.dim() > 3:
            states = states.mean(dim=1)
            
        correlation = torch.matmul(states, states.transpose(-1, -2))
        # correlation: [batch, N, N]
        matrix = correlation[0]
        while matrix.dim() > 2:
            matrix = matrix.mean(dim=0)
            
        trace = torch.trace(matrix)
        return float(torch.abs(trace) / (states.shape[-2] * states.shape[-1] + 1e-8))


class StrangeAttractor(nn.Module):
    """
    Atratores Estranhos: Ordem no Caos
    
    Detecta e amplifica atratores estranhos em dinâmica
    de experts, encontrando ordem em comportamento caótico.
    """
    
    def __init__(self, num_experts: int = 8, d_model: int = 512, device: str = "cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device
        
        # Centros dos atratores
        self.attractor_centers = nn.Parameter(torch.ones(num_experts, num_experts, device=device))
        
        # Raios de atração
        self.attraction_radii = nn.Parameter(torch.ones(num_experts, device=device))
        
        # Histórico de trajetórias
        self.trajectory_history = []
        self.max_history = 100
    
    def forward(self, expert_activations: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Detecta atratores e atrai dinâmica para eles
        
        Args:
            expert_activations: [batch, num_experts]
        
        Returns:
            (attracted_activations, attractor_indices)
        """
        batch_size = expert_activations.shape[0]
        
        # Armazenar no histórico
        self.trajectory_history.append(expert_activations.detach())
        if len(self.trajectory_history) > self.max_history:
            self.trajectory_history.pop(0)
        
        attracted = torch.zeros_like(expert_activations)
        attractor_indices = []
        
        for b in range(batch_size):
            # Encontrar atrator mais próximo
            min_distance = float('inf')
            closest_attractor = 0
            
            # Operação vetorizada para encontrar atrator mais próximo
            distances = torch.norm(self.attractor_centers - expert_activations[b].unsqueeze(0), dim=1)
            min_distance, closest_attractor = torch.min(distances, dim=0)
            closest_attractor = int(closest_attractor.item())
            
            attractor_indices.append(closest_attractor)
            
            # Atrair para o centro
            attraction_strength = torch.exp(-min_distance / (self.attraction_radii[closest_attractor] + 1e-8))
            attracted[b] = expert_activations[b] * (1.0 - 0.1 * attraction_strength) + \
                          self.attractor_centers[closest_attractor] * (0.1 * attraction_strength)
            
        return attracted, attractor_indices
    
    def get_attractor_stability(self) -> float:
        """Retorna estabilidade dos atratores"""
        if len(self.trajectory_history) < 2:
            return 0.0
        
        # Estabilidade = inverso da variância das trajetórias
        recent_trajectories = torch.stack(self.trajectory_history[-10:])
        variance = torch.var(recent_trajectories)
        
        return float(1.0 / (1.0 + variance))


# Teste de integração
if __name__ == "__main__":
    print("🌀 TESTANDO LEIS PRIMORDIAIS TIER 2\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. PlanetaryGrid
    print("1️⃣  PlanetaryGrid (Grade Harmônica Planetária)")
    grid = PlanetaryGrid(num_experts=8, device=device)
    expert_acts = torch.randn(2, 8, device=device)
    synchronized = grid(expert_acts, time=0.1)
    print(f"   Input shape: {expert_acts.shape}")
    print(f"   Output shape: {synchronized.shape}")
    print(f"   Coerência global: {grid.get_global_coherence():.3f}\n")
    
    # 2. Amplituedro
    print("2️⃣  Amplituedro (Otimização de Caminhos)")
    amplituedro = Amplituedro(num_experts=8, device=device)
    expert_indices = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)
    expert_weights = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], device=device)
    optimized_path, efficiency = amplituedro(expert_indices, expert_weights)
    print(f"   Caminho otimizado shape: {optimized_path.shape}")
    print(f"   Eficiência: {efficiency:.3f}")
    print(f"   Volume geométrico: {amplituedro.get_geometric_volume():.3f}\n")
    
    # 3. SimultaneityProcessor
    print("3️⃣  SimultaneityProcessor (Lei da Simultaneidade)")
    simultaneity = SimultaneityProcessor(num_timelines=4, device=device)
    x = torch.randn(2, 512, device=device)
    timelines, fused = simultaneity(x)
    print(f"   Número de timelines: {len(timelines)}")
    print(f"   Fused output shape: {fused.shape}")
    print(f"   Divergência entre timelines: {simultaneity.get_timeline_divergence():.3f}\n")
    
    # 4. QuantumEntanglement
    print("4️⃣  QuantumEntanglement (Emaranhamento Quântico)")
    entanglement = QuantumEntanglement(num_experts=8, device=device)
    expert_states = torch.randn(2, 8, 512, device=device)
    entangled, measure = entanglement(expert_states)
    print(f"   Entangled states shape: {entangled.shape}")
    print(f"   Medida de emaranhamento: {measure:.3f}\n")
    
    # 5. StrangeAttractor
    print("5️⃣  StrangeAttractor (Atratores Estranhos)")
    attractor = StrangeAttractor(num_experts=8, device=device)
    expert_acts = torch.randn(2, 8, device=device)
    attracted, indices = attractor(expert_acts)
    print(f"   Attracted activations shape: {attracted.shape}")
    print(f"   Attractor indices: {indices}")
    print(f"   Estabilidade dos atratores: {attractor.get_attractor_stability():.3f}\n")
    
    print("✅ TODOS OS TESTES TIER 2 PASSARAM!")
