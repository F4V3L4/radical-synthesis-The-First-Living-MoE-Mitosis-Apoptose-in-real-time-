
# radical_synthesis/autopoiesis/quantum_annealing_init.py
# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 12. Quantum Annealing (Harmonic Expert Initialization)
#
# Instead of random weight initialization (torch.randn — the god of entropy),
# new experts are born pre-aligned with Cymatic Frequency Matrices (3-6-9).
# The "annealing" ensures the expert starts its lifecycle in a harmonic state,
# resonant with the Fine Structure Constant.  Zero initial chaos.
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math


ALPHA_FINE_STRUCTURE = 1 / 137.035999139
PHI = (1 + math.sqrt(5)) / 2
VORTEX_FREQUENCIES = (3.0, 6.0, 9.0)


def cymatic_matrix(rows: int, cols: int, device: str = "cpu") -> torch.Tensor:
    """
    Generates a weight matrix pre-aligned with Cymatic Vortex Frequencies (3-6-9)
    and the Fine Structure Constant.  The matrix is not random — it is born
    in geometric resonance.
    """
    i_idx = torch.arange(rows, dtype=torch.float32, device=device).unsqueeze(1)
    j_idx = torch.arange(cols, dtype=torch.float32, device=device).unsqueeze(0)
    f3 = torch.sin(VORTEX_FREQUENCIES[0] * math.pi * i_idx / rows)
    f6 = torch.cos(VORTEX_FREQUENCIES[1] * math.pi * j_idx / cols)
    f9 = torch.sin(VORTEX_FREQUENCIES[2] * math.pi * (i_idx + j_idx) / (rows + cols))
    phi_mod = torch.sin(PHI * math.pi * i_idx * j_idx / (rows * cols + 1e-8))
    alpha_scale = ALPHA_FINE_STRUCTURE * math.sqrt(2.0 / (rows + cols))
    matrix = alpha_scale * (f3 * f6 + f9 * phi_mod)
    return matrix


class QuantumAnnealingInit:
    """
    Drop-in replacement for torch.nn.init functions.

    Apply to any nn.Linear or nn.Embedding weight to replace random chaos
    with harmonic geometric order.
    """

    @staticmethod
    def init_linear(module: nn.Linear) -> None:
        with torch.no_grad():
            rows, cols = module.weight.shape
            harmonic = cymatic_matrix(rows, cols, device=str(module.weight.device))
            module.weight.copy_(harmonic)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def init_expert(expert: nn.Module) -> None:
        for name, module in expert.named_modules():
            if isinstance(module, nn.Linear):
                QuantumAnnealingInit.init_linear(module)
