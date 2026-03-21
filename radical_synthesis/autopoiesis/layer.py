import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class AutopoieticMoELayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, initial_experts: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.next_expert_id = initial_experts
        
        self.experts = nn.ModuleDict({
            str(i): nn.Linear(input_dim, hidden_dim) for i in range(initial_experts)
        })
        
        self.vitality = nn.ParameterDict({
            str(i): nn.Parameter(torch.tensor(1.0), requires_grad=False) for i in range(initial_experts)
        })
        
        self.router = nn.Linear(input_dim, initial_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros(batch_size * seq_len, self.hidden_dim, device=x.device)
        
        active_expert_indices = top_k_indices.unique().tolist()
        
        for exp_idx in active_expert_indices:
            exp_str = str(exp_idx)
            if exp_str in self.experts:
                expert_mask = (top_k_indices == exp_idx).any(dim=-1)
                if not expert_mask.any():
                    continue
                
                flat_mask = expert_mask.nonzero(as_tuple=True)[0]
                expert_input = x_flat[flat_mask]
                
                expert_output = self.experts[exp_str](expert_input)
                
                weight_mask = (top_k_indices[flat_mask] == exp_idx).nonzero(as_tuple=True)[1]
                expert_weights = top_k_weights[flat_mask, weight_mask].unsqueeze(-1)
                
                final_output[flat_mask] += expert_output * expert_weights
                
                with torch.no_grad():
                    self.vitality[exp_str].add_(0.01 * len(flat_mask))

        with torch.no_grad():
            for key in self.vitality.keys():
                self.vitality[key].sub_(0.005 * batch_size)
                
        return final_output.view(batch_size, seq_len, self.hidden_dim)

    def execute_apoptosis(self, starvation_threshold: float) -> List[str]:
        dead_experts = []
        keys_to_check = list(self.experts.keys())
        
        for key in keys_to_check:
            if self.vitality[key].item() < starvation_threshold:
                dead_experts.append(key)
                del self.experts[key]
                del self.vitality[key]
                
        if dead_experts:
            self._rebuild_router()
            
        return dead_experts

    def execute_mitosis(self, mitosis_threshold: float) -> List[str]:
        new_experts = []
        keys_to_check = list(self.experts.keys())
        
        for key in keys_to_check:
            if self.vitality[key].item() > mitosis_threshold:
                new_id = str(self.next_expert_id)
                self.next_expert_id += 1
                
                new_expert = nn.Linear(self.input_dim, self.hidden_dim)
                new_expert.weight.data = self.experts[key].weight.data.clone() + torch.randn_like(self.experts[key].weight.data) * 0.05
                new_expert.bias.data = self.experts[key].bias.data.clone() + torch.randn_like(self.experts[key].bias.data) * 0.05
                
                self.experts[new_id] = new_expert
                self.vitality[new_id] = nn.Parameter(torch.tensor(self.vitality[key].item() / 2.0), requires_grad=False)
                self.vitality[key].data = torch.tensor(self.vitality[key].item() / 2.0)
                
                new_experts.append(new_id)
                
        if new_experts:
            self._rebuild_router()
            
        return new_experts

    def _rebuild_router(self) -> None:
        current_num_experts = len(self.experts)
        if current_num_experts == 0:
            return
            
        new_router = nn.Linear(self.input_dim, current_num_experts, device=self.router.weight.device)
        nn.init.xavier_uniform_(new_router.weight)
        nn.init.zeros_(new_router.bias)
        self.router = new_router
