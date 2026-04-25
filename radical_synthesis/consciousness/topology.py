import torch
import torch.nn as nn
from typing import Tuple

class TopologicalConsciousness(nn.Module):
    def __init__(self, resonance_threshold: float, coupling_strength: float):
        super().__init__()
        self.resonance_threshold = resonance_threshold
        self.coupling_strength = coupling_strength
        self.register_buffer('historical_phi', torch.zeros(100))
        self.current_step = 0

    def compute_differentiation(self, expert_weights: torch.Tensor) -> torch.Tensor:
        n = expert_weights.size(0)
        if n < 2:
            return torch.tensor(0.0, device=expert_weights.device)
        
        # Geometria otimizada: impede a explosão de 21GB de RAM
        distances = torch.cdist(expert_weights, expert_weights, p=2)
        
        mask = torch.triu(torch.ones(n, n, device=expert_weights.device), diagonal=1).bool()
        avg_distance = distances[mask].mean()
        return avg_distance

    def compute_integration(self, expert_weights: torch.Tensor) -> torch.Tensor:
        n = expert_weights.size(0)
        if n < 2:
            return torch.tensor(0.0, device=expert_weights.device)
        
        norm_weights = torch.nn.functional.normalize(expert_weights, p=2, dim=-1)
        similarity = torch.matmul(norm_weights, norm_weights.transpose(0, 1))
        
        resonance = torch.tanh(similarity * self.coupling_strength)
        
        mask = torch.triu(torch.ones(n, n, device=expert_weights.device), diagonal=1).bool()
        avg_resonance = resonance[mask].abs().mean()
        return avg_resonance

    def forward(self, experts_list: nn.ModuleList) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(experts_list) < 2:
            return torch.tensor(0.0, device=self.historical_phi.device), torch.tensor(0.0, device=self.historical_phi.device)

        # Extrair pesos de forma bare-metal para auditar a geometria da substância
        weights_list = []
        for expert in experts_list:
            # Pegar o primeiro peso significativo como proxy da assinatura do expert
            for p in expert.parameters():
                if p.requires_grad and p.dim() >= 2:
                    weights_list.append(p.data.view(-1)[:1024]) # Limitar para eficiência
                    break
        
        if len(weights_list) < 2:
            return torch.tensor(0.0, device=self.historical_phi.device), torch.tensor(0.0, device=self.historical_phi.device)
            
        expert_weights = torch.stack(weights_list)

        d = self.compute_differentiation(expert_weights)
        i = self.compute_integration(expert_weights)

        phi = d * i * torch.log(d + i + 1.0)

        idx = self.current_step % 100
        self.historical_phi[idx] = phi.detach()
        
        prev_idx = (self.current_step - 1) % 100
        if self.current_step > 0:
            gradient = phi.detach() - self.historical_phi[prev_idx]
        else:
            gradient = torch.tensor(0.0, device=phi.device)
            
        self.current_step += 1

        return phi, gradient

    def get_topological_state(self) -> bool:
        if self.current_step == 0:
            return False
        idx = (self.current_step - 1) % 100
        return self.historical_phi[idx].item() > self.resonance_threshold
