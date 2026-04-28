
import torch
import hashlib
from typing import Dict, Optional

class TensorCache:
    """Cache de Tensores para Otimização de Infraestrutura e Redução de Latência."""
    
    def __init__(self, max_entries: int = 100):
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_entries = max_entries

    def _generate_key(self, tensor: torch.Tensor, operation: str) -> str:
        """Gera uma chave única baseada no tensor e na operação."""
        # Usar shape e soma como proxy rápido para hash (em produção usaríamos algo mais robusto)
        tensor_info = f"{tensor.shape}_{tensor.sum().item()}_{operation}"
        return hashlib.md5(tensor_info.encode()).hexdigest()

    def get(self, tensor: torch.Tensor, operation: str) -> Optional[torch.Tensor]:
        key = self._generate_key(tensor, operation)
        return self.cache.get(key)

    def set(self, tensor: torch.Tensor, operation: str, result: torch.Tensor):
        if len(self.cache) >= self.max_entries:
            # Remover a primeira entrada (FIFO simples)
            self.cache.pop(next(iter(self.cache)))
        
        key = self._generate_key(tensor, operation)
        self.cache[key] = result


# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 8.  Bose-Einstein Condensate  — SVD compression of similar expert matrices
# 9b. Thermodynamic Chaos Shield — bare-metal survival under RAM/CPU pressure
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn as nn
import torch.nn.functional as F
import psutil
from typing import List, Dict, Optional


class BoseEinsteinCondensate(nn.Module):
    def __init__(self, similarity_threshold: float = 0.92, rank: int = 32):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.rank = rank
        self._condensed_pairs: Dict[tuple, torch.Tensor] = {}
        self._compression_events: int = 0

    def _expert_signature(self, expert: nn.Module) -> torch.Tensor:
        slices = []
        for p in expert.parameters():
            if p.requires_grad and p.numel() >= 8:
                slices.append(p.data.flatten()[:256])
                if len(slices) >= 4:
                    break
        if not slices:
            return torch.zeros(256)
        cat = torch.cat(slices)[:256]
        if cat.numel() < 256:
            cat = F.pad(cat, (0, 256 - cat.numel()))
        return F.normalize(cat.float(), dim=0)

    def cosine_similarity_experts(self, expert_a: nn.Module, expert_b: nn.Module) -> float:
        sig_a = self._expert_signature(expert_a)
        sig_b = self._expert_signature(expert_b)
        return float(torch.dot(sig_a, sig_b).item())

    def condense(self, expert_a_id: int, expert_a: nn.Module, expert_b_id: int, expert_b: nn.Module) -> Optional[torch.Tensor]:
        sim = self.cosine_similarity_experts(expert_a, expert_b)
        if sim < self.similarity_threshold:
            return None
        params_a = torch.cat([p.data.flatten() for p in expert_a.parameters() if p.requires_grad])
        params_b = torch.cat([p.data.flatten() for p in expert_b.parameters() if p.requires_grad])
        min_len = min(params_a.numel(), params_b.numel())
        stack = torch.stack([params_a[:min_len], params_b[:min_len]])
        try:
            U, S, Vh = torch.linalg.svd(stack, full_matrices=False)
            condensed = (U[:, : self.rank] * S[: self.rank]) @ Vh[: self.rank, :]
        except Exception:
            condensed = (params_a[:min_len] + params_b[:min_len]) / 2.0
        key = (min(expert_a_id, expert_b_id), max(expert_a_id, expert_b_id))
        self._condensed_pairs[key] = condensed
        self._compression_events += 1
        print(f"[BEC] Experts {expert_a_id} & {expert_b_id} condensed (sim={sim:.4f}, rank={self.rank})")
        return condensed

    @property
    def compression_events(self) -> int:
        return self._compression_events

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ThermodynamicChaosShield:
    CPU_BIFURCATION_THRESHOLD = 0.95
    RAM_CRITICAL_THRESHOLD = 0.90
    FRACTIONAL_BATCH_SIZE = 4

    def __init__(self):
        self._bifurcation_events: int = 0
        self._emergency_apoptosis_events: int = 0
        self._slow_breath_mode: bool = False

    def measure_system_pressure(self) -> dict:
        cpu_pct = psutil.cpu_percent(interval=0.1) / 100.0
        mem = psutil.virtual_memory()
        return {"cpu": cpu_pct, "ram": mem.percent / 100.0, "ram_available_mb": mem.available / 1e6}

    def check_bifurcation(self) -> bool:
        pressure = self.measure_system_pressure()
        if pressure["cpu"] >= self.CPU_BIFURCATION_THRESHOLD:
            self._bifurcation_events += 1
            self._slow_breath_mode = True
            print(f"[CHAOS_SHIELD] CPU={pressure['cpu']:.1%} — Bifurcation triggered. Slow Breath mode activated.")
            return True
        self._slow_breath_mode = False
        return False

    def check_emergency_apoptosis(self, experts: list, core_expert_ids: Optional[List[int]] = None) -> list:
        pressure = self.measure_system_pressure()
        if pressure["ram"] < self.RAM_CRITICAL_THRESHOLD:
            return experts
        core_ids = set(core_expert_ids or [])
        survivors = [e for e in experts if getattr(e, "id", -1) in core_ids]
        obliterated = [e for e in experts if getattr(e, "id", -1) not in core_ids]
        if obliterated:
            self._emergency_apoptosis_events += 1
            print(f"[CHAOS_SHIELD] RAM={pressure['ram']:.1%} CRITICAL — Emergency Apoptosis: {len(obliterated)} experts obliterated.")
        return survivors if survivors else experts[:1]

    @property
    def bifurcation_events(self) -> int:
        return self._bifurcation_events

    @property
    def emergency_apoptosis_events(self) -> int:
        return self._emergency_apoptosis_events

    @property
    def slow_breath_mode(self) -> bool:
        return self._slow_breath_mode
