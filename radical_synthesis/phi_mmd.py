# radical_synthesis/phi_mmd.py
# ─────────────────────────────────────────────────────────────────────────────
# Métrica Φ baseada em Maximum Mean Discrepancy (MMD).
# Substitui a métrica original que ficava instável com 1.000+ experts.
# ─────────────────────────────────────────────────────────────────────────────

import torch
from torch import Tensor


def phi_mmd(
    centroids: Tensor,
    subsample: int   = 64,
    bandwidth: float = 1.0,
) -> float:
    """
    Mede o quão diferentes os experts são entre si.

    Retorna um número entre 0.0 e 1.0:
      - Próximo de 0.0 → experts muito parecidos (colapso de diversidade → ruim)
      - Próximo de 1.0 → experts bem diferentes (diversidade saudável → bom)

    Args:
        centroids:  tensor (N, d_model) com um vetor representativo por expert.
                    Obtenha com: torch.stack([e.get_centroid() for e in experts])
        subsample:  quantos experts amostrar por cálculo.
                    64 é suficiente mesmo com 1.000+ experts no pool.
        bandwidth:  largura do kernel RBF. 1.0 funciona bem na maioria dos casos.

    Exemplo de uso:
        centroids = torch.stack([e.get_centroid() for e in self.experts])
        phi = phi_mmd(centroids, subsample=64)
        if phi < 0.15:
            self._trigger_categorical_shift()  # colapso detectado
    """
    N = centroids.shape[0]

    if N <= 1:
        return 0.0  # só um expert = sem diversidade possível

    # ── Amostragem aleatória para escalar com pools grandes ──────────────────
    k   = min(subsample, N)
    idx = torch.randperm(N, device=centroids.device)[:k]
    c   = centroids[idx]  # (k, d_model)

    # ── Kernel RBF entre todos os pares de experts ───────────────────────────
    # K(i,j) = exp( -||ci - cj||² / (2σ²) )
    # K próximo de 1 → experts muito similares
    # K próximo de 0 → experts muito diferentes
    dists = torch.cdist(c, c, p=2).pow(2)                 # (k, k)
    sigma = bandwidth * (c.shape[1] ** 0.5)               # heurística de Scott
    K     = torch.exp(-dists / (2.0 * sigma ** 2))        # (k, k)

    # ── Similaridade média entre pares distintos (off-diagonal) ─────────────
    mascara  = ~torch.eye(k, dtype=torch.bool, device=K.device)
    sim_media = K[mascara].mean().item()

    # ── Diversidade = 1 - similaridade ──────────────────────────────────────
    diversidade = 1.0 - sim_media

    return float(max(0.0, min(1.0, diversidade)))


# ─────────────────────────────────────────────────────────────────────────────
# Threshold padrão para detecção de colapso
# ─────────────────────────────────────────────────────────────────────────────
COLAPSO_THR = 0.15
"""
Valor de Φ abaixo do qual considera-se que os experts colapsaram.
Ajuste entre 0.10 (mais tolerante) e 0.25 (mais sensível) conforme a tarefa.
"""
