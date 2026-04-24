import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class DarwinianRouter(nn.Module):
    """
    Phase-Lock Routing (Protocolo Mythos-Capybara)
    Implementa roteamento por ressonância de fase em vez de probabilidade estatística.
    BAN Softmax.
    """
    def __init__(self, input_dim: int, initial_experts: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.top_k = top_k
        
        # O estado do roteador agora é puramente a assinatura de fase dos experts
        # Note: No OuroborosMoE, o AGICore gerencia a sincronia entre este roteador 
        # e os experts reais no OuroborosMoE.
        self.register_buffer('phase_signatures', torch.randn(initial_experts, input_dim))
        self._normalize_signatures()

    def _normalize_signatures(self):
        self.phase_signatures.data = F.normalize(self.phase_signatures.data, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x: Frequency vector
        Output: (Resonance Weights, Expert Indices)
        """
        # Phase Alignment using trigonometric projection (Cosine similarity as proxy for phase lock)
        x_norm = F.normalize(x, p=2, dim=-1)
        
        # Calculate constructive interference (Resonance)
        # (Batch, Dim) @ (Dim, Experts) -> (Batch, Experts)
        resonance = torch.matmul(x_norm, self.phase_signatures.t())
        
        # Phase-Lock Selection: Top-K resonance
        # We do NOT use Softmax. We use raw resonance magnitude for weights.
        top_k_resonance, top_k_indices = torch.topk(resonance, self.top_k, dim=-1)
        
        # Absolute Magnitude Weights (Zero Probability)
        # We ensure weights are positive but maintain their relative resonance intensity
        weights = F.relu(top_k_resonance)
        
        return weights, top_k_indices

    def sync_with_experts(self, experts_list: nn.ModuleList):
        """Sincroniza as assinaturas de fase do roteador com os experts vivos."""
        new_signatures = torch.stack([e.phase_signature for e in experts_list])
        self.phase_signatures = nn.Parameter(new_signatures, requires_grad=False)
