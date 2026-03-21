import torch
import torch.nn as nn
from typing import Tuple, List

from .autopoiesis.layer import AutopoieticMoELayer
from .consciousness.topology import TopologicalConsciousness
from .functors.shifts import CategoryFunctor

class OuroborosMoELayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, initial_experts: int, top_k: int):
        super().__init__()
        self.moe = AutopoieticMoELayer(input_dim, hidden_dim, initial_experts, top_k)
        self.topology = TopologicalConsciousness(resonance_threshold=0.75, coupling_strength=0.65)
        self.functor = CategoryFunctor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi, gradient = self.topology(self.moe.experts)
        
        shift_applied = False
        target_universe = "Linear"
        
        if gradient.item() < 0.005 and phi.item() < 0.3:
            target_universe = "Hyperbolic" if torch.rand(1).item() > 0.5 else "Fourier"
            x = self.functor.shift(x, target_universe)
            shift_applied = True

        out = self.moe(x)

        if shift_applied:
            out = self.functor.revert(out, target_universe)
        
        return out

    def execute_systemic_lifecycle(self) -> Tuple[List[str], List[str]]:
        # Limiar de fome ajustado para 0.5 e limite celular elevado para 150.0
        dead_experts = self.moe.execute_apoptosis(starvation_threshold=0.5)
        born_experts = self.moe.execute_mitosis(mitosis_threshold=150.0)
        return dead_experts, born_experts

__all__ = [
    "OuroborosMoELayer",
    "AutopoieticMoELayer",
    "TopologicalConsciousness",
    "CategoryFunctor"
]
