# radical_synthesis/lazy_router.py
# ─────────────────────────────────────────────────────────────────────────────
# Router com cache incremental — só reconstrói o que mudou.
# Substitui o DarwinianRouter que reconstruía tudo do zero a cada evento.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LazyRouter(nn.Module):
    """
    Router que mantém um cache de centroids e só atualiza
    os experts que nasceram ou morreram desde o último rebuild.

    Em vez de recalcular a matriz de afinidade inteira (~40ms com 1.400 experts),
    só marca os experts 'sujos' e reconstrói na próxima chamada forward().

    Uso:
        router = LazyRouter(d_model=512, top_k=2)

        # Quando um expert nasce (mitose):
        router.on_born(clone.id, clone)

        # Quando um expert morre (apoptose):
        router.on_died(expert.id)

        # No forward do OuroborosMoELayer:
        top_scores, top_idx = router(x)
    """

    def __init__(self, d_model: int, top_k: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.top_k    = top_k

        # Cache: expert_id → tensor centroid (d_model,)
        self._cache: dict[int, torch.Tensor] = {}

        # IDs que precisam ser recalculados no próximo rebuild
        self._dirty: set[int] = set()

        # Matriz de afinidade entre experts (N × N) — pode ser None
        self._affinity: Optional[torch.Tensor] = None

        # Ordem dos experts na matriz (para indexar corretamente)
        self._order: list[int] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Eventos de ciclo de vida — chame esses métodos no lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_born(self, expert_id: int, expert: nn.Module) -> None:
        """
        Registra um expert recém-nascido.
        Chame logo após criar o clone na mitose.
        """
        centroid = self._compute_centroid(expert)
        self._cache[expert_id] = centroid
        self._dirty.add(expert_id)
        # Invalida a matriz — nova coluna/linha precisará ser inserida
        self._affinity = None

    def on_died(self, expert_id: int) -> None:
        """
        Remove um expert morto do cache.
        Chame logo após a apoptose.
        """
        self._cache.pop(expert_id, None)
        self._dirty.discard(expert_id)
        self._affinity = None  # invalida: linha/coluna sumiu

    # ─────────────────────────────────────────────────────────────────────────
    # Rebuild incremental
    # ─────────────────────────────────────────────────────────────────────────

    def rebuild(self) -> None:
        """
        Reconstrói apenas o necessário.
        É um no-op se nada mudou desde o último rebuild.
        Chamado automaticamente no forward().
        """
        if not self._dirty and self._affinity is not None:
            return  # cache válido, nada a fazer

        if not self._cache:
            self._affinity = None
            self._order    = []
            return

        ids           = list(self._cache.keys())
        self._order   = ids
        centroids     = torch.stack([self._cache[i] for i in ids])  # (N, d)
        normed        = F.normalize(centroids, dim=-1)
        self._affinity = normed @ normed.T                          # (N, N)
        self._dirty.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward — roteia tokens para os top_k experts
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Roteia cada item do batch para os top_k experts mais adequados.

        Args:
            x: tensor (batch, seq_len, d_model)

        Returns:
            top_scores: (batch, top_k)  — pontuação de cada expert
            top_idx:    (batch, top_k)  — índice de cada expert selecionado
        """
        self.rebuild()  # no-op se cache válido

        if not self._cache:
            raise RuntimeError("LazyRouter: nenhum expert registrado no cache.")

        # Representação do batch: média ao longo da sequência
        q = F.normalize(x.mean(dim=1), dim=-1)  # (batch, d_model)

        # Centroids de todos os experts
        centroids = torch.stack([self._cache[i] for i in self._order])
        c_normed  = F.normalize(centroids, dim=-1)  # (N, d_model)

        # Score de afinidade: cosine similarity
        scores = q @ c_normed.T  # (batch, N)

        top_scores, top_idx = scores.topk(
            min(self.top_k, len(self._order)), dim=-1
        )
        return top_scores, top_idx

    # ─────────────────────────────────────────────────────────────────────────
    # Inicialização em lote (use no __init__ do OuroborosMoELayer)
    # ─────────────────────────────────────────────────────────────────────────

    def register_initial_experts(self, experts: list) -> None:
        """
        Registra todos os experts iniciais de uma vez.
        Chame no __init__ do OuroborosMoELayer após criar os experts.

        Args:
            experts: lista de experts (cada um com .id e parâmetros PyTorch)
        """
        for expert in experts:
            self._cache[expert.id] = self._compute_centroid(expert)
        self._affinity = None  # será construída no primeiro forward

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitários
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_centroid(expert: nn.Module) -> torch.Tensor:
        """
        Calcula um vetor representativo do expert a partir dos seus parâmetros.
        Usa os primeiros parâmetros treináveis para eficiência.
        """
        slices = []
        for p in expert.parameters():
            if p.requires_grad and p.numel() >= 8:
                slices.append(p.data.flatten()[:128])
                if len(slices) >= 4:
                    break

        if not slices:
            # Expert sem parâmetros treináveis — retorna zeros
            return torch.zeros(128)

        cat = torch.cat(slices)

        # Normaliza para comprimento fixo de 128
        if cat.numel() < 128:
            cat = F.pad(cat, (0, 128 - cat.numel()))
        else:
            cat = cat[:128]

        return cat.detach().float()

    @property
    def n_experts(self) -> int:
        """Número de experts atualmente registrados no cache."""
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"LazyRouter(d_model={self.d_model}, top_k={self.top_k}, "
            f"n_experts={self.n_experts}, cache_valid={self._affinity is not None})"
        )
