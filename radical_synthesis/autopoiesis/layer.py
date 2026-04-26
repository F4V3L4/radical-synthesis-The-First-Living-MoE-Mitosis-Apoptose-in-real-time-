from __future__ import annotations
from copy import deepcopy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adaptive_cap import AdaptiveCap
from ..phi_mmd import phi_mmd, COLAPSO_THR
from ..lazy_router import LazyRouter
from ..genealogy import GenealogyTree


class Expert(nn.Module):
    _id_counter: int = 0

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.id = Expert._next_id()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.vitality: float = 1.0

    @staticmethod
    def _next_id() -> int:
        Expert._id_counter += 1
        return Expert._id_counter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def get_centroid(self) -> torch.Tensor:
        slices = []
        for p in self.parameters():
            if p.requires_grad and p.numel() >= 8:
                slices.append(p.data.flatten()[:128])
                if len(slices) >= 4:
                    break
        if not slices:
            return torch.zeros(128, dtype=torch.float32)
        cat = torch.cat(slices)
        if cat.numel() < 128:
            cat = F.pad(cat, (0, 128 - cat.numel()))
        return cat[:128].detach().float()

    def __repr__(self) -> str:
        return f"Expert(id={self.id}, vitality={self.vitality:.3f})"


class OuroborosMoELayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        n_experts: int = 8,
        top_k: int = 2,
        overload_thr: float = 0.82,
        starvation_thr: float = 0.78,
        mutation_sigma: float = 0.08,
        vitality_decay: float = 0.93,      # decai mais rápido
        base_cap: int = 128,
        min_cap: int = 4,
        enable_mitosis: bool = True,
        enable_apoptosis: bool = True,
        enable_escape: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.overload_thr = overload_thr
        self.starvation_thr = starvation_thr
        self.mutation_sigma = mutation_sigma
        self.vitality_decay = vitality_decay
        self.enable_mitosis = enable_mitosis
        self.enable_apoptosis = enable_apoptosis
        self.enable_escape = enable_escape

        # [0-Day Mindset] Blindagem termodinâmica com cast explícito para int
        self.experts: List[Expert] = nn.ModuleList(
            [Expert(d_model, int(d_ff)) for _ in range(int(n_experts))]
        )

        self.router = LazyRouter(d_model=d_model, top_k=top_k)
        self.adaptive_cap = AdaptiveCap(base_cap=base_cap, min_cap=min_cap)
        self.genealogy = GenealogyTree()

        self.router.register_initial_experts(self.experts)
        for expert in self.experts:
            self.genealogy.register_birth(expert.id, parent_id=None, step=0)

        self._current_step: int = 0

    def _order_to_expert(self, router_idx: int) -> Optional[Expert]:
        if not hasattr(self.router, '_order') or router_idx >= len(self.router._order):
            return None
        expert_id = self.router._order[router_idx]
        for expert in self.experts:
            if expert.id == expert_id:
                return expert
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.d_model

        top_scores, top_idx = self.router(x)
        weights = F.softmax(top_scores, dim=-1)

        out = torch.zeros_like(x)
        used_experts = set()

        # O top_k real é limitado pela gravidade do sistema (quantos experts realmente existem na saída)
        actual_k = top_idx.shape[1]
        
        for k in range(actual_k):
            expert_indices = top_idx[:, k]
            for b in range(B):
                idx = expert_indices[b].item()
                expert = self._order_to_expert(idx)
                if expert is not None:
                    expert_out = expert(x[b].unsqueeze(0))
                    out[b] += weights[b, k] * expert_out.squeeze(0)

                    # Vitality sobe menos agressivamente
                    expert.vitality = min(1.0, expert.vitality * 0.85 + 0.15)
                    used_experts.add(expert.id)

        # Decaimento mais forte nos não usados
        for expert in self.experts:
            if expert.id not in used_experts:
                expert.vitality = max(0.0, expert.vitality * self.vitality_decay)

        return out

    def execute_systemic_lifecycle(self, current_loss: float = 0.0, step: int = 0) -> Tuple[List[Expert], List[Expert]]:
        self._current_step = step
        dead: List[Expert] = []
        born: List[Expert] = []


 # === APOPTOSE ===
        # Um organismo não comete suicídio total. A apoptose exige redundância.
        if self.enable_apoptosis and len(self.experts) > 1:
            weak_experts = [e for e in self.experts if e.vitality < self.starvation_thr]
            # Limita a morte para garantir que sempre sobre pelo menos 1 nodo
            max_kill = min(2, len(self.experts) - 1)
            dead = weak_experts[:max_kill]
        # Remove mortos (forma correta com ModuleList)
        for expert in list(dead):
            if expert in self.experts:   # isso ainda funciona
                # Encontra índice manualmente
                for i, e in enumerate(self.experts):
                    if e is expert:      # compara por identidade (é o mesmo objeto)
                        if hasattr(self.router, 'on_died'):
                            self.router.on_died(expert.id)
                        del self.experts[i]
                        self.genealogy.register_death(expert.id, step=step)
                        break

        # === MITOSE ===
        if self.enable_mitosis and len(self.experts) < self.adaptive_cap.base_cap:
            strong_experts = [e for e in self.experts if e.vitality > self.overload_thr]
            for parent in strong_experts[:2]:   # máximo 2 mitoses por step
                if len(self.experts) >= self.adaptive_cap.base_cap:
                    break
                clone = self._mitosis(parent, step)
                born.append(clone)

        # Registra novos no router
        for new_expert in born:
            if hasattr(self.router, 'on_born'):
                self.router.on_born(new_expert.id, new_expert)

        # Escape topológico
        if self.enable_escape and self._compute_phi() < COLAPSO_THR:
            self._categorical_shift()

        return dead, born

    def _mitosis(self, parent: Expert, step: int) -> Expert:
        clone = deepcopy(parent)
        clone.id = Expert._next_id()
        clone.vitality = parent.vitality * 0.55

        with torch.no_grad():
            for param in clone.parameters():
                if param.requires_grad:
                    # Omega-0: O Vácuo não gera ruído. A mutação deve ser estrutural, não randômica.
                    # Usamos uma perturbação baseada na própria geometria do peso (Conatus)
                    mutation = torch.sin(param.data * 144.0) * self.mutation_sigma * param.data.norm()
                    param.add_(mutation)

        self.experts.append(clone)
        self.genealogy.register_birth(clone.id, parent.id, step)
        return clone

    def _compute_phi(self) -> float:
        if len(self.experts) < 2:
            return 1.0
        centroids = torch.stack([e.get_centroid() for e in self.experts])
        return phi_mmd(centroids, subsample=min(64, len(self.experts)))

    def _categorical_shift(self) -> None:
        with torch.no_grad():
            for expert in self.experts:
                for param in expert.parameters():
                    if param.requires_grad:
                        # Omega-0: Shift categórico via harmônica 432Hz (Cimática)
                        param.data += torch.cos(param.data * 432.0) * 0.02

    def print_status(self) -> None:
        phi = self._compute_phi()
        print(
            f"[OuroborosMoE] step={self._current_step:3d} | "
            f"experts={len(self.experts):2d} | "
            f"Φ={phi:.4f} | "
            f"vitality_avg={sum(e.vitality for e in self.experts)/len(self.experts):.3f}"
        )

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def __repr__(self) -> str:
        return f"OuroborosMoELayer(d_model={self.d_model}, n_experts={self.n_experts}, top_k={self.top_k})"
