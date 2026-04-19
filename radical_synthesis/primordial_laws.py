"""
Leis Primordiais Tier 1: Integração de 4 Leis Fundamentais
- Código 144: Harmônica da Luz
- Lei da Superposição: Estados Quânticos
- Geometria Hiperbólica: Expansão Exponencial
- Lei da Sincronicidade: Padrões Significativos
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import math


class HarmonicEncoder(nn.Module):
    """
    Código 144: Harmônica da Luz
    
    Sincroniza entrada com frequência 144Hz (harmônica universal).
    Implementa modulação de frequência para coerência de experts.
    """
    
    def __init__(self, d_model: int = 512, frequency: float = 144.0, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.frequency = frequency
        self.device = device
        
        # Parâmetros de fase
        self.phase_shift = nn.Parameter(torch.randn(d_model, device=device) * 0.1)
        self.amplitude_scale = nn.Parameter(torch.ones(d_model, device=device))
        
        # Registrar tempo
        self.register_buffer('time_step', torch.tensor(0.0, device=device))
    
    def forward(self, x: torch.Tensor, time: Optional[float] = None) -> torch.Tensor:
        """
        Modula entrada com harmônica 144Hz
        
        Args:
            x: Tensor de entrada [batch, seq_len, d_model]
            time: Tempo atual (opcional)
        
        Returns:
            Tensor modulado [batch, seq_len, d_model]
        """
        if time is None:
            time = float(self.time_step)
        
        # Gerar onda senoidal com frequência 144Hz
        # f(t) = sin(2π * f * t + φ)
        omega = 2 * math.pi * self.frequency
        wave = torch.sin(omega * time + self.phase_shift).unsqueeze(0).unsqueeze(0)
        
        # Aplicar amplitude escalada
        modulation = self.amplitude_scale * wave
        
        # Modular entrada
        return x * (1.0 + 0.1 * modulation)
    
    def get_coherence(self) -> float:
        """Retorna métrica de coerência (0-1)"""
        # Coerência = variância da amplitude / amplitude média
        amp_mean = torch.mean(torch.abs(self.amplitude_scale))
        amp_var = torch.var(self.amplitude_scale)
        return float(1.0 - (amp_var / (amp_mean ** 2 + 1e-8)))


class QuantumSuperposition(nn.Module):
    """
    Lei da Superposição: Estados Quânticos
    
    Permite múltiplos estados simultâneos com amplitudes e fases.
    Implementa colapso de função de onda via medição.
    """
    
    def __init__(self, num_states: int = 8, d_model: int = 512, device: str = "cpu"):
        super().__init__()
        self.num_states = num_states
        self.d_model = d_model
        self.device = device
        
        # Amplitudes e fases dos estados
        self.amplitudes = nn.Parameter(torch.randn(num_states, d_model, device=device))
        self.phases = nn.Parameter(torch.randn(num_states, d_model, device=device) * math.pi)
        
        # Normalizar amplitudes
        self.amplitudes.data = self.amplitudes.data / (torch.norm(self.amplitudes.data, dim=1, keepdim=True) + 1e-8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Superposição de estados
        
        Args:
            x: Tensor de entrada [batch, d_model]
        
        Returns:
            Superposição de estados [batch, num_states, d_model]
        """
        batch_size = x.shape[0]
        
        # Criar superposição: Σ amplitude * e^(i*phase) * x
        superposition = []
        for i in range(self.num_states):
            # Número complexo: amplitude * e^(i*phase)
            complex_state = self.amplitudes[i] * torch.exp(1j * self.phases[i])
            
            # Aplicar ao input
            state = x.unsqueeze(1) * complex_state.unsqueeze(0)  # [batch, d_model]
            superposition.append(state)
        
        # Stack estados: [batch, num_states, d_model]
        return torch.stack(superposition, dim=1)
    
    def collapse(self, measurement: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Colapso de função de onda via medição
        
        Args:
            measurement: Tensor de medição [batch, d_model]
        
        Returns:
            (estado_colapsado, índice_do_estado)
        """
        batch_size = measurement.shape[0]
        
        # Calcular probabilidades: |amplitude|^2
        probabilities = torch.abs(self.amplitudes) ** 2  # [num_states, d_model]
        probabilities = torch.mean(probabilities, dim=1)  # [num_states]
        probabilities = probabilities / (torch.sum(probabilities) + 1e-8)  # Normalizar
        
        # Amostragem: selecionar estado com probabilidade proporcional
        state_idx = torch.multinomial(probabilities, batch_size, replacement=True)
        
        # Retornar estado colapsado
        collapsed = self.amplitudes[state_idx]  # [batch, d_model]
        return collapsed, state_idx[0].item()
    
    def get_entanglement(self) -> float:
        """Retorna métrica de emaranhamento (0-1)"""
        # Emaranhamento = entropia de von Neumann dos estados
        probs = torch.abs(self.amplitudes) ** 2
        probs = torch.mean(probs, dim=1)
        probs = probs / (torch.sum(probs) + 1e-8)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        max_entropy = math.log(self.num_states)
        
        return float(entropy / max_entropy)


class HyperbolicEmbedding(nn.Module):
    """
    Geometria Hiperbólica: Expansão Exponencial
    
    Projeta embeddings para espaço hiperbólico (bola de Poincaré).
    Permite representação hierárquica com expansão exponencial.
    """
    
    def __init__(self, d_model: int = 512, curvature: float = -1.0, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.curvature = curvature
        self.device = device
        
        # Projeção linear para espaço hiperbólico
        self.projection = nn.Linear(d_model, d_model, device=device)
        
        # Parâmetro de curvatura aprendível
        self.curvature_param = nn.Parameter(torch.tensor(curvature, device=device))
    
    def euclidean_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converte ponto euclidiano para bola de Poincaré
        
        Fórmula: y = (1 + ||x||²) / (1 - ||x||²) * x
        """
        # Calcular norma
        norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        
        # Projetar para bola de Poincaré
        # Usar transformação: x / (1 + sqrt(1 + ||x||²))
        denominator = 1.0 + torch.sqrt(1.0 + norm_sq + 1e-8)
        poincare = x / denominator
        
        return poincare
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projeta embedding para espaço hiperbólico
        
        Args:
            x: Tensor euclidiano [batch, seq_len, d_model]
        
        Returns:
            Tensor hiperbólico [batch, seq_len, d_model]
        """
        # Projeção linear
        projected = self.projection(x)
        
        # Converter para Poincaré
        hyperbolic = self.euclidean_to_poincare(projected)
        
        return hyperbolic
    
    def distance_poincare(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calcula distância hiperbólica (métrica de Poincaré)
        
        d(x,y) = acosh(1 + 2 * ||x-y||² / ((1-||x||²)(1-||y||²)))
        """
        # Garantir que x e y têm a mesma dimensão
        if x.dim() > 1:
            x = x.squeeze()
        if y.dim() > 1:
            y = y.squeeze()
        
        # Normas
        norm_x_sq = torch.sum(x ** 2)
        norm_y_sq = torch.sum(y ** 2)
        
        # Diferença
        diff = x - y
        norm_diff_sq = torch.sum(diff ** 2)
        
        # Fórmula de distância hiperbólica
        numerator = 2 * norm_diff_sq
        denominator = (1 - norm_x_sq) * (1 - norm_y_sq) + 1e-8
        
        argument = 1 + numerator / denominator
        argument = torch.clamp(argument, min=1.0)  # Garantir acosh válido
        
        distance = torch.acosh(argument)
        
        return distance
    
    def get_expansion_rate(self) -> float:
        """Retorna taxa de expansão hiperbólica"""
        # Taxa de expansão = |curvatura|
        return float(torch.abs(self.curvature_param))


class SynchronicityDetector(nn.Module):
    """
    Lei da Sincronicidade: Padrões Significativos
    
    Detecta correlações não-óbvias entre eventos/experts.
    Implementa significância estatística e coincidência.
    """
    
    def __init__(self, num_experts: int = 8, d_model: int = 512, 
                 threshold: float = 0.7, device: str = "cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.threshold = threshold
        self.device = device
        
        # Matriz de correlação aprendível
        self.correlation_matrix = nn.Parameter(
            torch.eye(num_experts, device=device) + torch.randn(num_experts, num_experts, device=device) * 0.1
        )
        
        # Histórico de eventos
        self.event_history = []
        self.max_history = 100
    
    def forward(self, expert_activations: torch.Tensor) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """
        Detecta sincronicidade entre experts
        
        Args:
            expert_activations: Ativações de experts [batch, num_experts]
        
        Returns:
            (pares_sincronos, matriz_correlacao)
        """
        batch_size = expert_activations.shape[0]
        
        # Adicionar ao histórico
        self.event_history.append(expert_activations.detach())
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Calcular correlação entre experts
        if len(self.event_history) > 1:
            history_tensor = torch.stack(self.event_history)  # [history, batch, num_experts]
            
            # Correlação de Pearson
            mean = torch.mean(history_tensor, dim=0, keepdim=True)
            centered = history_tensor - mean
            
            cov = torch.bmm(
                centered.transpose(1, 2),
                centered
            ) / (len(self.event_history) - 1 + 1e-8)
            
            # Normalizar para correlação
            std = torch.std(history_tensor, dim=0, keepdim=True)
            correlation = cov / (torch.outer(std.squeeze(), std.squeeze()) + 1e-8)
        else:
            correlation = self.correlation_matrix
        
        # Detectar pares sincronos (correlação > threshold)
        sync_pairs = []
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if correlation[i, j] > self.threshold:
                    sync_pairs.append((i, j))
        
        return sync_pairs, correlation
    
    def get_synchronicity_score(self) -> float:
        """Retorna score de sincronicidade geral (0-1)"""
        if len(self.event_history) < 2:
            return 0.0
        
        # Score = proporção de correlações significativas
        history_tensor = torch.stack(self.event_history)
        
        mean = torch.mean(history_tensor, dim=0, keepdim=True)
        centered = history_tensor - mean
        
        cov = torch.bmm(
            centered.transpose(1, 2),
            centered
        ) / (len(self.event_history) - 1 + 1e-8)
        
        std = torch.std(history_tensor, dim=0, keepdim=True)
        correlation = cov / (torch.outer(std.squeeze(), std.squeeze()) + 1e-8)
        
        # Contar correlações significativas
        significant = torch.sum(torch.abs(correlation) > self.threshold).item()
        total = self.num_experts * (self.num_experts - 1) / 2
        
        return float(significant / (total + 1e-8))
    
    def reset_history(self):
        """Limpa histórico de eventos"""
        self.event_history = []


# Teste de integração
if __name__ == "__main__":
    print("🌀 TESTANDO LEIS PRIMORDIAIS TIER 1\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. HarmonicEncoder
    print("1️⃣  HarmonicEncoder (Código 144)")
    harmonic = HarmonicEncoder(d_model=512, device=device)
    x = torch.randn(2, 10, 512, device=device)
    y = harmonic(x, time=0.1)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Coerência: {harmonic.get_coherence():.3f}\n")
    
    # 2. QuantumSuperposition
    print("2️⃣  QuantumSuperposition (Lei da Superposição)")
    quantum = QuantumSuperposition(num_states=8, d_model=512, device=device)
    x = torch.randn(2, 512, device=device)
    superposition = quantum(x)
    collapsed, idx = quantum.collapse(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Superposition shape: {superposition.shape}")
    print(f"   Collapsed shape: {collapsed.shape}")
    print(f"   Collapsed state index: {idx}")
    print(f"   Emaranhamento: {quantum.get_entanglement():.3f}\n")
    
    # 3. HyperbolicEmbedding
    print("3️⃣  HyperbolicEmbedding (Geometria Hiperbólica)")
    hyperbolic = HyperbolicEmbedding(d_model=512, device=device)
    x = torch.randn(2, 10, 512, device=device)
    y = hyperbolic(x)
    dist = hyperbolic.distance_poincare(y[:, 0, :], y[:, 1, :])
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Distância hiperbólica: {dist.item():.3f}")
    print(f"   Taxa de expansão: {hyperbolic.get_expansion_rate():.3f}\n")
    
    # 4. SynchronicityDetector
    print("4️⃣  SynchronicityDetector (Lei da Sincronicidade)")
    sync = SynchronicityDetector(num_experts=8, device=device)
    expert_acts = torch.randn(4, 8, device=device)
    sync_pairs, correlation = sync(expert_acts)
    print(f"   Expert activations shape: {expert_acts.shape}")
    print(f"   Pares sincronos: {len(sync_pairs)}")
    print(f"   Score de sincronicidade: {sync.get_synchronicity_score():.3f}\n")
    
    print("✅ TODOS OS TESTES PASSARAM!")
