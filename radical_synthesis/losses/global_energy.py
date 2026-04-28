import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalEnergyFunction(nn.Module):
    """
    O 9 Central: Função de Energia Global que unifica a métrica de verdade do sistema.
    Unifica Loss (Erro), Entropia (Caos) e Estabilidade (Conatus).
    """
    def __init__(self, lambda_loss=1.0, lambda_entropy=0.1, lambda_stability=0.05):
        super().__init__()
        self.lambda_loss = lambda_loss
        self.lambda_entropy = lambda_entropy
        self.lambda_stability = lambda_stability

    def forward(self, model_loss, expert_weights, conatus_levels):
        """
        Calcula a Energia Global do Sistema.
        E = λ1*Loss + λ2*Entropy - λ3*Stability
        """
        # 1. Entropia do Roteamento (Mede o caos na decisão)
        # expert_weights: [batch, seq, num_experts]
        prob_dist = expert_weights.mean(dim=(0, 1))
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10))
        
        # 2. Estabilidade (Mede a saúde do Conatus)
        stability = torch.mean(conatus_levels)
        
        # 3. Energia Total (O objetivo é minimizar a energia)
        # Queremos baixa loss, baixa entropia e alta estabilidade
        total_energy = (self.lambda_loss * model_loss + 
                        self.lambda_entropy * entropy - 
                        self.lambda_stability * stability)
        
        return {
            "total_energy": total_energy,
            "entropy": entropy,
            "stability": stability,
            "model_loss": model_loss
        }


# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 5. Heisenberg Regularization (Uncertainty Principle)
#
# If the optimizer tries to fix a weight (position) too precisely, we inject a
# dynamic reverse adjustment on the gradient (momentum).  The machine never
# crystallizes into overfitting — geometric fluidity is always preserved.
# ─────────────────────────────────────────────────────────────────────────────
import math


class HeisenbergRegularization(nn.Module):
    """
    Uncertainty Principle applied to neural weight optimization.

    Delta_position * Delta_momentum >= hbar / 2

    When a weight's gradient norm (momentum) is too large (over-certainty in
    direction), we inject a counter-perturbation proportional to the weight
    magnitude (position certainty).  This prevents the optimizer from locking
    any weight into a rigid local minimum — the Radical Synthesis always has
    room to operate.
    """

    HBAR = 1.0545718e-34
    HBAR_SCALED = 1e-3

    def __init__(self, uncertainty_scale: float = 1.0):
        super().__init__()
        self.uncertainty_scale = uncertainty_scale
        self._violations_corrected: int = 0

    def apply(self, model: nn.Module) -> None:
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                delta_pos = p.data.norm().item()
                delta_mom = p.grad.norm().item()
                product = delta_pos * delta_mom
                min_product = self.HBAR_SCALED * self.uncertainty_scale
                if product < min_product and delta_pos > 1e-8:
                    correction = min_product / (product + 1e-12)
                    p.grad.mul_(correction)
                    self._violations_corrected += 1

    def forward(self, model: nn.Module) -> None:
        self.apply(model)

    @property
    def violations_corrected(self) -> int:
        return self._violations_corrected
