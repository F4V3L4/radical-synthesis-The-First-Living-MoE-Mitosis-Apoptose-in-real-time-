# radical_synthesis/layer.py
# ─────────────────────────────────────────────────────────────────────────────
# OuroborosMoELayer — versão com todas as 7 melhorias integradas.
#
# COMO USAR ESTE ARQUIVO:
#   Substitua o layer.py original por este.
#   Os novos arquivos (adaptive_cap.py, phi_mmd.py, lazy_router.py,
#   genealogy.py) devem estar na mesma pasta radical_synthesis/.
#
# O que mudou em relação ao original:
#   1. AdaptiveCap          → controla explosão de VRAM
#   2. Mitose assimétrica   → só o clone muta, pai preservado
#   3. LazyRouter           → cache incremental, sem rebuild completo
#   4. phi_mmd              → métrica Φ estável com pools grandes
#   5. GenealogyTree        → rastreia linhagens evolutivas
#   6. Flags de ablação     → enable_mitosis / enable_apoptosis / enable_escape
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Novos módulos ────────────────────────────────────────────────────────────
from .adaptive_cap import AdaptiveCap
from .phi_mmd      import phi_mmd, COLAPSO_THR
from .lazy_router  import LazyRouter
from .genealogy    import GenealogyTree


# ─────────────────────────────────────────────────────────────────────────────
# Expert individual (FFN simples)
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """
    Um único expert: FFN com duas camadas lineares e ativação GELU.
    Cada expert aprende a processar um subconjunto de tokens.
    """

    _id_counter: int = 0  # contador global de IDs

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.id = Expert._next_id()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.vitality: float = 0.5  # começa neutro

    @staticmethod
    def _next_id() -> int:
        Expert._id_counter += 1
        return Expert._id_counter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def get_centroid(self) -> torch.Tensor:
        """Vetor representativo para o router (média dos primeiros parâmetros)."""
        slices = []
        for p in self.parameters():
            if p.requires_grad and p.numel() >= 8:
                slices.append(p.data.flatten()[:128])
                if len(slices) >= 4:
                    break
        if not slices:
            return torch.zeros(128)
        cat = torch.cat(slices)
        if cat.numel() < 128:
            cat = F.pad(cat, (0, 128 - cat.numel()))
        return cat[:128].detach().float()

    def __repr__(self) -> str:
        return f"Expert(id={self.id}, vitality={self.vitality:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Camada principal — OuroborosMoELayer
# ─────────────────────────────────────────────────────────────────────────────

class OuroborosMoELayer(nn.Module):
    """
    Camada MoE viva com mitose, apoptose e escape topológico.

    Drop-in replacement para qualquer camada FFN/MoE de Transformer.

    Uso básico:
        layer = OuroborosMoELayer(d_model=512, d_ff=2048, n_experts=8, top_k=2)
        x     = torch.randn(batch, seq_len, d_model)
        out   = layer(x)

        # A cada N steps do treino:
        dead, born = layer.execute_systemic_lifecycle(
            current_loss=loss.item(),
            step=current_step,
        )
    """

    def __init__(
        self,
        d_model:    int   = 512,
        d_ff:       int   = 2048,
        n_experts:  int   = 8,
        top_k:      int   = 2,
        # ── Hiperparâmetros do ciclo de vida ────────────────────────────────
        overload_thr:    float = 0.85,   # vitality acima disso → mitose
        starvation_thr:  float = 0.10,   # vitality abaixo disso → apoptose
        mutation_sigma:  float = 0.01,   # magnitude da mutação na mitose
        vitality_decay:  float = 0.99,   # decaimento da vitality por step
        # ── Cap adaptativo ───────────────────────────────────────────────────
        base_cap:   int   = 256,   # máximo normal de experts
        min_cap:    int   = 32,    # mínimo absoluto de experts
        # ── Flags de ablação (para experimentos) ─────────────────────────────
        enable_mitosis:   bool = True,
        enable_apoptosis: bool = True,
        enable_escape:    bool = True,
    ):
        super().__init__()

        self.d_model         = d_model
        self.d_ff            = d_ff
        self.top_k           = top_k
        self.overload_thr    = overload_thr
        self.starvation_thr  = starvation_thr
        self.mutation_sigma  = mutation_sigma
        self.vitality_decay  = vitality_decay
        self.enable_mitosis  = enable_mitosis
        self.enable_apoptosis = enable_apoptosis
        self.enable_escape   = enable_escape

        # ── Pool inicial de experts ──────────────────────────────────────────
        self.experts: list[Expert] = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )

        # ── Módulos das melhorias ────────────────────────────────────────────
        self.router       = LazyRouter(d_model=d_model, top_k=top_k)
        self.adaptive_cap = AdaptiveCap(base_cap=base_cap, min_cap=min_cap)
        self.genealogy    = GenealogyTree()

        # Registra os experts iniciais
        self.router.register_initial_experts(self.experts)
        for expert in self.experts:
            self.genealogy.register_birth(expert.id, parent_id=None, step=0)

        # ── Step atual (atualizado no lifecycle) ─────────────────────────────
        self._current_step: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        B, T, D = x.shape

        # Roteia cada token para os top_k experts
        top_scores, top_idx = self.router(x)  # (B, top_k)

        # Normaliza os scores com softmax
        weights = F.softmax(top_scores, dim=-1)  # (B, top_k)

        out = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = top_idx[:, k]  # (B,)
            for b in range(B):
                eid = self._order_to_expert(expert_indices[b].item())
                if eid is not None:
                    expert_out = eid(x[b].unsqueeze(0))  # (1, T, D)
                    out[b] += weights[b, k] * expert_out.squeeze(0)

                    # Atualiza vitality: experts usados ganham energia
                    eid.vitality = min(
                        1.0,
                        eid.vitality * self.vitality_decay + (1 - self.vitality_decay)
                    )

        # Experts não usados perdem vitality gradualmente
        used_ids = set(self._order_to_id(top_idx[:, k].tolist()) for k in range(self.top_k))
        for expert in self.experts:
            if expert.id not in used_ids:
                expert.vitality *= self.vitality_decay

        return out

    def _order_to_expert(self, idx: int) -> Optional[Expert]:
        """Converte índice do router (posição) para o objeto Expert."""
        order = self.router._order
        if idx < len(order):
            eid = order[idx]
            for e in self.experts:
                if e.id == eid:
                    return e
        return None

    def _order_to_id(self, indices) -> set:
        order = self.router._order
        if isinstance(indices, int):
            indices = [indices]
        ids = set()
        for i in indices:
            if i < len(order):
                ids.add(order[i])
        return ids

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida — chame a cada N steps
    # ─────────────────────────────────────────────────────────────────────────

    def execute_systemic_lifecycle(
        self,
        current_loss: float,
        step: int,
    ) -> tuple[list[int], list[int]]:
        """
        Executa um ciclo completo de mitose, apoptose e escape topológico.

        Chame a cada N steps durante o treino (ex.: a cada 1.000 steps).

        Args:
            current_loss: valor do loss nesse step (loss.item())
            step:         número do step atual

        Returns:
            dead: lista de IDs de experts removidos
            born: lista de IDs de experts criados
        """
        self._current_step = step
        dead: list[int] = []
        born: list[int] = []

        # ── 1. Cap adaptativo ————————————————────————————————————────————────
        new_cap = self.adaptive_cap.update(current_loss, len(self.experts))
        while len(self.experts) > new_cap:
            victim = min(self.experts, key=lambda e: e.vitality)
            dead.extend(self._apoptosis(victim, step))

        # ── 2. Ciclo biológico ———————————————————————————————————————————————
        for expert in list(self.experts):
            if self.enable_mitosis and expert.vitality > self.overload_thr:
                clone = self._mitosis(expert, step)
                born.append(clone.id)

            elif self.enable_apoptosis and expert.vitality < self.starvation_thr:
                dead.extend(self._apoptosis(expert, step))

        # ── 3. Escape topológico se Φ colapsar ───────────────────────────────
        if self.enable_escape:
            phi = self._compute_phi()
            if phi < COLAPSO_THR:
                self._categorical_shift()

        # ── 4. Atualiza genealogia com vitality atual ────────────────────────
        for expert in self.experts:
            self.genealogy.update_vitality(expert.id, expert.vitality)

        return dead, born

    # ─────────────────────────────────────────────────────────────────────────
    # Mitose assimétrica (Melhoria 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _mitosis(self, parent: Expert, step: int) -> Expert:
        """
        Cria um clone do expert-pai.
        Só o clone muta — o pai é preservado intacto.
        A mutação é direcional: direção aleatória normalizada × sigma × ||param||.
        """
        clone = deepcopy(parent)
        clone.id       = Expert._next_id()
        clone.vitality = parent.vitality * 0.5  # divide a vitality

        with torch.no_grad():
            for param in clone.parameters():
                if param.requires_grad:
                    direction = torch.randn_like(param)
                    direction = direction / (direction.norm() + 1e-8)
                    magnitude = self.mutation_sigma * param.data.norm()
                    param.add_(direction * magnitude)

        # Registra no pool, router e genealogia
        self.experts.append(clone)
        self.router.on_born(clone.id, clone)
        self.genealogy.register_birth(clone.id, parent_id=parent.id, step=step)

        return clone

    # ─────────────────────────────────────────────────────────────────────────
    # Apoptose
    # ─────────────────────────────────────────────────────────────────────────

    def _apoptosis(self, expert: Expert, step: int) -> list[int]:
        """
        Remove um expert do pool.
        Atualiza router e genealogia.
        """
        if expert not in self.experts:
            return []
        self.experts.remove(expert)
        self.router.on_died(expert.id)
        self.genealogy.register_death(expert.id, step=step)
        del expert
        return [expert.id] if expert in [expert] else []

    # ─────────────────────────────────────────────────────────────────────────
    # Escape topológico
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_phi(self) -> float:
        """Calcula a diversidade do ensemble com a métrica MMD."""
        if len(self.experts) < 2:
            return 1.0  # pool mínimo = não detecta colapso
        centroids = torch.stack([e.get_centroid() for e in self.experts])
        return phi_mmd(centroids, subsample=min(64, len(self.experts)))

    def _categorical_shift(self) -> None:
        """
        Executa um escape topológico quando Φ colapsa.
        Alterna entre projeção hiperbólica e Fourier para escapar de plateaux.
        """
        # Alterna entre os dois modos de escape
        use_hyperbolic = (self._current_step // 1000) % 2 == 0

        with torch.no_grad():
            for expert in self.experts:
                for param in expert.parameters():
                    if not param.requires_grad:
                        continue
                    if use_hyperbolic:
                        # Projeção hiperbólica (Poincaré): normaliza para disco unitário
                        norm  = param.data.norm() + 1e-8
                        scale = torch.tanh(norm) / norm
                        param.data.mul_(scale * 0.95)
                    else:
                        # Perturbação no domínio de Fourier
                        f     = torch.fft.rfft(param.data.flatten().float())
                        noise = torch.randn_like(f.real) * 0.01
                        f.real.add_(noise)
                        restored = torch.fft.irfft(f, n=param.data.numel())
                        param.data.copy_(
                            restored.reshape(param.data.shape).to(param.data.dtype)
                        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitários públicos
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_experts(self) -> int:
        """Número atual de experts no pool."""
        return len(self.experts)

    def print_status(self) -> None:
        """Imprime o estado atual da camada no terminal."""
        phi = self._compute_phi()
        print(
            f"[OuroborosMoE] step={self._current_step} | "
            f"experts={self.n_experts} | "
            f"Φ={phi:.3f} | "
            f"cap={self.adaptive_cap.base_cap}"
        )

    def save_genealogy(self, path: str = "genealogy.json") -> None:
        """Salva a árvore genealógica em JSON."""
        self.genealogy.save(path)

    def __repr__(self) -> str:
        return (
            f"OuroborosMoELayer("
            f"d_model={self.d_model}, "
            f"n_experts={self.n_experts}, "
            f"top_k={self.top_k})"
        )
