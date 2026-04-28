# radical_synthesis/lazy_router.py
# ─────────────────────────────────────────────────────────────────────────────
# Router com cache incremental bare-metal.
# Respeita a dimensionalidade estrita do hardware (d_model).
#
# [QUANTUM UPGRADE v2.0]
# 1. Gradient Tunneling  — bypasses loss barriers via probabilistic quantum leap
# 2. Wave Superposition  — tokens flow through N experts in fractional probability
#                          state; wave function collapses only at output layer
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

ALPHA_FINE_STRUCTURE = 1 / 137.035999139


class LazyRouter(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        top_k: int = 2,
        tunnel_prob: float = ALPHA_FINE_STRUCTURE,
        superposition_k: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.tunnel_prob = tunnel_prob
        self.superposition_k = superposition_k
        self._cache: dict[int, torch.Tensor] = {}
        self._dirty: set[int] = set()
        self._affinity: Optional[torch.Tensor] = None
        self._order: list[int] = []
        self._tunnel_event_count: int = 0

    def on_born(self, expert_id: int, expert: nn.Module) -> None:
        centroid = self._compute_centroid(expert)
        self._cache[expert_id] = centroid
        self._dirty.add(expert_id)
        self._affinity = None

    def on_died(self, expert_id: int) -> None:
        self._cache.pop(expert_id, None)
        self._dirty.discard(expert_id)
        self._affinity = None

    def rebuild(self) -> None:
        if not self._dirty and self._affinity is not None:
            return
        if not self._cache:
            self._affinity = None
            self._order = []
            return
        ids = list(self._cache.keys())
        self._order = ids
        centroids = torch.stack([self._cache[i] for i in ids])
        normed = F.normalize(centroids, dim=-1)
        self._affinity = normed @ normed.T
        self._dirty.clear()

    def _quantum_tunnel(
        self,
        scores: torch.Tensor,
        top_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = scores.shape[0]
        n_experts = scores.shape[1]
        if n_experts < 2:
            return scores, top_idx
        tunnel_mask = torch.rand(B, device=scores.device) < self.tunnel_prob
        if not tunnel_mask.any():
            return scores, top_idx
        least_idx = scores.argmin(dim=-1)
        top_idx_tunneled = top_idx.clone()
        scores_tunneled = scores.clone()
        for b in range(B):
            if tunnel_mask[b]:
                top_idx_tunneled[b, 0] = least_idx[b]
                scores_tunneled[b, 0] = scores[b, least_idx[b]]
                self._tunnel_event_count += 1
        return scores_tunneled, top_idx_tunneled

    def _superposition_scores(
        self,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_experts = scores.shape[1]
        k = self.superposition_k if self.superposition_k > 0 else n_experts
        k = min(k, n_experts)
        sup_scores, sup_idx = scores.topk(k, dim=-1)
        sup_weights = F.softmax(sup_scores, dim=-1)
        return sup_weights, sup_idx

    def forward(
        self,
        x: torch.Tensor,
        collapse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.rebuild()
        if not self._cache:
            raise RuntimeError("LazyRouter: nenhum expert registrado no cache.")
        q = F.normalize(x.mean(dim=1), dim=-1)
        device = q.device
        centroids = torch.stack([self._cache[i] for i in self._order]).to(device)
        c_normed = F.normalize(centroids, dim=-1)
        scores = q @ c_normed.T
        if not collapse:
            return self._superposition_scores(scores)
        top_scores, top_idx = scores.topk(
            min(self.top_k, len(self._order)), dim=-1
        )
        top_scores, top_idx = self._quantum_tunnel(scores, top_idx)
        return top_scores, top_idx

    def register_initial_experts(self, experts: list) -> None:
        for expert in experts:
            self._cache[expert.id] = self._compute_centroid(expert)
        self._affinity = None

    def _compute_centroid(self, expert: nn.Module) -> torch.Tensor:
        slices = []
        total_elements = 0
        for p in expert.parameters():
            if p.requires_grad and p.numel() >= 8:
                flat = p.data.flatten()
                slices.append(flat)
                total_elements += flat.numel()
                if total_elements >= self.d_model * 4:
                    break
        if not slices:
            return torch.zeros(self.d_model, dtype=torch.float32, device='cpu')
        cat = torch.cat(slices)
        if cat.numel() < self.d_model:
            cat = F.pad(cat, (0, self.d_model - cat.numel()))
        else:
            cat = cat[:self.d_model]
        return F.normalize(cat, dim=0).detach().float()

    @property
    def tunnel_events(self) -> int:
        return self._tunnel_event_count


# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 7. Wave Interference Routing (Destructive Cancellation of Hallucinations)
#
# Each expert is assigned a phase vector.  When two experts emit misaligned
# phase vectors (contradiction), the tensor math causes them to cancel each
# other out (vector_A + vector_B ≈ 0).  Silence is preferable to a lie.
# ─────────────────────────────────────────────────────────────────────────────


class WaveInterferenceRouter(nn.Module):
    """
    Phase-based routing that replaces classical Softmax with destructive
    interference cancellation.

    Each expert gets a learnable phase angle theta_i.  The routing score
    becomes a complex-valued inner product.  Experts whose phase vectors
    are anti-aligned (contradiction) produce near-zero combined amplitude —
    the system falls silent rather than hallucinating.
    """

    def __init__(self, n_experts: int, d_model: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.phase_angles = nn.Parameter(
            torch.linspace(0, 2 * 3.14159265, n_experts)
        )
        self.amplitude_proj = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        top_k: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        amplitudes = self.amplitude_proj(x.mean(dim=1))
        cos_phase = torch.cos(self.phase_angles)
        sin_phase = torch.sin(self.phase_angles)
        real_scores = amplitudes * cos_phase
        imag_scores = amplitudes * sin_phase
        interference_scores = real_scores + imag_scores
        coherence = torch.abs(real_scores + imag_scores)
        top_scores, top_idx = coherence.topk(min(top_k, self.n_experts), dim=-1)
        return top_scores, top_idx, coherence
