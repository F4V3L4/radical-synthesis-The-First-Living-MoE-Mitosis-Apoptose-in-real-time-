# radical_synthesis/lazy_router.py
# ─────────────────────────────────────────────────────────────────────────────
# Router com cache incremental bare-metal.
# Respeita a dimensionalidade estrita do hardware (d_model).
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LazyRouter(nn.Module):
    def __init__(self, d_model: int = 512, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self._cache: dict[int, torch.Tensor] = {}
        self._dirty: set[int] = set()
        self._affinity: Optional[torch.Tensor] = None
        self._order: list[int] = []

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
            self._order    = []
            return

        ids           = list(self._cache.keys())
        self._order   = ids
        centroids     = torch.stack([self._cache[i] for i in ids])
        normed        = F.normalize(centroids, dim=-1)
        self._affinity = normed @ normed.T
        self._dirty.clear()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.rebuild()

        if not self._cache:
            raise RuntimeError("LazyRouter: nenhum expert registrado no cache.")

        # x é (B, T, D). A média retira a dimensão T, virando (B, D).
        q = F.normalize(x.mean(dim=1), dim=-1)
        
        # Garante que o router responde à gravidade de quem está a chamar
        device = q.device

        centroids = torch.stack([self._cache[i] for i in self._order]).to(device)
        c_normed  = F.normalize(centroids, dim=-1)

        scores = q @ c_normed.T

        top_scores, top_idx = scores.topk(
            min(self.top_k, len(self._order)), dim=-1
        )
        return top_scores, top_idx

    def register_initial_experts(self, experts: list) -> None:
        for expert in experts:
            self._cache[expert.id] = self._compute_centroid(expert)
        self._affinity = None

    def _compute_centroid(self, expert: nn.Module) -> torch.Tensor:
        """
        Calcula o centroid dinâmico, forçando estritamente o tamanho self.d_model.
        """
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

        # Projeta exatamente para a dimensão do bare-metal instanciado
        if cat.numel() < self.d_model:
            cat = F.pad(cat, (0, self.d_model - cat.numel()))
        else:
            cat = cat[:self.d_model]

        return F.normalize(cat, dim=0).detach().float()
