import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class DarwinianRouter(nn.Module):
    def __init__(self, input_dim: int, initial_experts: int, top_k: int, noise_scale: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.top_k = top_k
        self.noise_scale = noise_scale
        
        self.latent_genomes = nn.Parameter(torch.randn(initial_experts, input_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = F.normalize(x, p=2, dim=-1)
        genomes_norm = F.normalize(self.latent_genomes, p=2, dim=-1)
        
        genetic_affinity = torch.matmul(x_norm, genomes_norm.t())
        
        if self.training:
            thermodynamic_noise = torch.randn_like(genetic_affinity) * self.noise_scale
            genetic_affinity = genetic_affinity + thermodynamic_noise
            
        top_k_logits, top_k_indices = torch.topk(genetic_affinity, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        return top_k_weights, top_k_indices

    def execute_genome_mitosis(self, parent_indices: List[int], mutation_rate: float) -> None:
        if not parent_indices:
            return
            
        with torch.no_grad():
            parents = self.latent_genomes[parent_indices]
            mutations = torch.randn_like(parents) * mutation_rate
            children_genomes = parents + mutations
            
            self.latent_genomes = nn.Parameter(
                torch.cat([self.latent_genomes, children_genomes], dim=0)
            )

    def execute_genome_apoptosis(self, dead_indices: List[int]) -> None:
        if not dead_indices:
            return
            
        with torch.no_grad():
            num_experts = self.latent_genomes.size(0)
            keep_mask = torch.ones(num_experts, dtype=torch.bool, device=self.latent_genomes.device)
            keep_mask[dead_indices] = False
            
            self.latent_genomes = nn.Parameter(self.latent_genomes[keep_mask])
