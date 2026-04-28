
# radical_synthesis/losses/pauli_exclusion_loss.py
# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 9. Pauli Exclusion Principle (Anti-Redundancy)
#
# Two fermions cannot occupy the same quantum state.  If two experts become
# mathematically identical (high cosine similarity), an Orthogonal Repulsion
# Loss forces their gradients apart — one must specialize differently or face
# Apoptosis.  Zero redundancy of memory.
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PauliExclusionLoss(nn.Module):
    """
    Orthogonal Repulsion Loss between expert weight matrices.

    Computes pairwise cosine similarity between all expert centroids.
    When two experts exceed the similarity threshold, a repulsion penalty
    is added to the total loss, forcing gradient divergence.
    """

    def __init__(self, similarity_threshold: float = 0.85, repulsion_scale: float = 1.0):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.repulsion_scale = repulsion_scale
        self._repulsion_events: int = 0

    def forward(self, expert_centroids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_centroids: (N, D) tensor of expert weight centroids
        Returns:
            scalar repulsion loss
        """
        if expert_centroids.shape[0] < 2:
            return torch.tensor(0.0, requires_grad=True)
        normed = F.normalize(expert_centroids, dim=-1)
        sim_matrix = normed @ normed.T
        mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
        upper_sim = sim_matrix[mask]
        violations = upper_sim[upper_sim > self.similarity_threshold]
        if violations.numel() == 0:
            return torch.tensor(0.0, requires_grad=True)
        self._repulsion_events += violations.numel()
        repulsion_loss = self.repulsion_scale * violations.mean()
        return repulsion_loss

    @property
    def repulsion_events(self) -> int:
        return self._repulsion_events
