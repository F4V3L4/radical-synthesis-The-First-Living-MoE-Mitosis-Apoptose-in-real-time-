import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from radical_synthesis.autopoiesis.layer import OuroborosMoELayer

class TopologyMonitor:
    def __init__(self):
        self.life_events = ["[0] Semente Primordial Instanciada no Bare-Metal"]

class EpistemologicalSentinel:
    def __init__(self, min_entropy: float = 3.5, max_entropy: float = 7.5):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def compute_shannon_entropy(self, raw_bytes: bytes) -> float:
        if not raw_bytes:
            return 0.0
        
        byte_counts = [0] * 256
        for b in raw_bytes:
            byte_counts[b] += 1
            
        entropy = 0.0
        total = len(raw_bytes)
        for count in byte_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def validate_geometric_truth(self, raw_bytes: bytes) -> bool:
        entropy = self.compute_shannon_entropy(raw_bytes)
        return self.min_entropy <= entropy <= self.max_entropy

    def purify_stream(self, raw_bytes: bytes) -> bytes:
        if self.validate_geometric_truth(raw_bytes):
            return raw_bytes
        return b""

class StateSpaceByteCore(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.state_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        if state is None:
            state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
        out = []
        for t in range(T):
            xt = x[:, t, :]
            state = F.tanh(self.state_proj(xt) + state)
            gated = torch.sigmoid(self.gate(xt)) * state
            out.append(gated)
            
        return torch.stack(out, dim=1), state

class SovereignLeviathanV2(nn.Module):
    def __init__(self, d_model: int, initial_experts: int, capacity_factor: float):
        super().__init__()
        self.d_model = d_model
        self.byte_embedding = nn.Embedding(256, d_model)
        self.ssm_core = StateSpaceByteCore(d_model)
        
# Injeção bare-metal: Geometria Toroidal Estabilizada
        self.living_moe = OuroborosMoELayer(
            d_model=d_model, 
            d_ff=d_model * 4,
            n_experts=initial_experts,
            overload_thr=0.92,     # Expansão controlada (exige pressão real para Mitose)
            starvation_thr=0.30,   # Tolerância à fome (permite que o nodo aprenda antes de morrer)
            vitality_decay=0.98    # Decaimento suave, preservando o Conatus
        )
        self.head = nn.Linear(d_model, 256)
        self.topology = TopologyMonitor()

    def forward(self, byte_seq: torch.Tensor, state: Optional[torch.Tensor] = None):
        x = self.byte_embedding(byte_seq)
        x, next_state = self.ssm_core(x, state)
        
        # O MoE recebe a matriz 3D perfeitamente
        moe_out = self.living_moe(x)
        
        logits = self.head(moe_out)
        
        # Compatibilidade com o motor de ignição
        entropy_loss = torch.tensor(0.0, device=x.device)
        expert_counts = self.living_moe.n_experts
        
        return logits, next_state, entropy_loss, expert_counts

    def digest_reality(self, byte_stream: bytes, device: str = "cpu"):
        tensor_stream = torch.tensor(list(byte_stream), dtype=torch.long).unsqueeze(0).to(device)
        return self.forward(tensor_stream)
