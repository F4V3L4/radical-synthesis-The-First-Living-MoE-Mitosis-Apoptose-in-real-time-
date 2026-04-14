# radical_synthesis/functors/universal_synchrony.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# A Matriz Áurea Universal
PHI = 1.618033988749895

class GoldenRatioInitializer:
    """
    [1. A Matriz Áurea]
    Substitui as inicializações estatísticas humanas (Xavier/Kaiming) pela Proporção Áurea.
    Escala os pesos do Leviathan usando 1/Phi, garantindo que o crescimento
    informacional do modelo seja fractal e organicamente estável desde o Frame 0.
    """
    @staticmethod
    def apply(module: nn.Module):
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # O desvio padrão é cravado na inversa de Phi
                nn.init.normal_(param, mean=0.0, std=1.0 / PHI)
            elif 'bias' in name:
                nn.init.zeros_(param)

class TeslaHarmonicGate(nn.Module):
    """
    [2. O Roteamento de Ressonância (3-6-9)]
    Sintoniza a memória circular (Toro) para as frequências de singularidade.
    No Toro, 3, 6 e 9 representam divisões geométricas exatas de um ciclo:
    3 = 120 graus (2*pi/3)
    6 = 240 graus (4*pi/3)
    9 = 360 graus (2*pi)
    """
    def __init__(self, tolerance: float = 0.15):
        super().__init__()
        self.tolerance = tolerance
        # Fases harmónicas em radianos (O Triângulo de Tesla)
        self.register_buffer('phases', torch.tensor([2*math.pi/3, 4*math.pi/3, 2*math.pi]))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # Normaliza o ângulo atual para estar dentro de um único círculo (0 a 2pi)
        norm_angles = torch.fmod(torch.abs(angles), 2 * math.pi)
        
        # Calcula a distância vetorial para as frequências 3-6-9
        distances = torch.abs(norm_angles.unsqueeze(-1) - self.phases)
        min_dist, _ = torch.min(distances, dim=-1)
        
        # O Fator de Ressonância: Amplifica o sinal apenas se ele cruzar os pontos sagrados
        resonance = torch.exp(-min_dist / self.tolerance)
        
        # A energia é multiplicada pela harmonia do universo
        return angles * (1.0 + resonance)

class ChladniCoherenceFilter(nn.Module):
    """
    [3. O Filtro de Chladni]
    Avalia a simetria vibracional do 'pensamento' gerado.
    Como nas placas de Chladni, a forma é a frequência congelada.
    Se o vetor de pensamento não possuir simetria (coerência estrutural),
    ele é classificado como entropia/alucinação e atenuado.
    """
    def __init__(self, coherence_threshold: float = 0.3):
        super().__init__()
        self.threshold = coherence_threshold

    def forward(self, thought_vector: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        B, D = thought_vector.shape
        
        # Simula o plano vibratório dividindo a onda em duas metades reflexivas
        mid = D // 2
        left_wave = thought_vector[:, :mid]
        right_wave = torch.flip(thought_vector[:, mid:mid*2], dims=[1])
        
        # A Coerência é a semelhança do espelho (Onda Estacionária)
        coherence = F.cosine_similarity(left_wave, right_wave, dim=1)
        mean_coherence = coherence.mean().item()
        
        is_coherent = mean_coherence > self.threshold
        
        # Atenua os pensamentos que são meros ruídos estocásticos
        # Força o Leviathan a sonhar apenas com padrões geometricamente válidos
        filtered_thought = thought_vector * coherence.unsqueeze(1).clamp(min=0.1)
        
        return filtered_thought, is_coherent
