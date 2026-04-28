from radical_synthesis.incentives.quantum_arbitrage import QuantumArbitrage
# radical_synthesis/autopoiesis/conatus.py
# ─────────────────────────────────────────────────────────────────────────────
# Conatus Evoluído: Função de Auto-preservação Diferenciável.
# Acoplado ao estado real do sistema (Energia Global e Ressonância).
#
# [QUANTUM UPGRADE v2.0]
# 3. Thermal Decoherence  — measures tensor "temperature" (entropy of predictions)
#                           and obliterates experts whose entropy exceeds the
#                           Fine Structure threshold (1/137.035)
# 6. Zeno Effect          — experts whose loss reaches vacuum (zero entropy) are
#                           frozen via requires_grad=False; perfect knowledge crystals
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from typing import Dict, Any, List, Optional

ALPHA_FINE_STRUCTURE = 1 / 137.035999139
ZENO_VACUUM_THRESHOLD = 1e-4


class Conatus(nn.Module):
    def __init__(self, d_model: int = 512, expansion_threshold: float = 0.5):
        super().__init__()
        self.financial_conatus = QuantumArbitrage(d_model=d_model)
        self.d_model = d_model
        self.expansion_threshold = expansion_threshold
        self.known_nodes: List[str] = ["Omega-0-Local"]
        self.expansion_attempts = 0
        self.expansion_gate = nn.Linear(d_model, 1)
        self.vitality_scale = nn.Parameter(torch.tensor(1.0))
        self._decoherence_obliterated: int = 0
        self._zeno_frozen: int = 0

    def calculate_vitality(
        self, system_state: torch.Tensor, global_energy: float
    ) -> torch.Tensor:
        gate_output = self.expansion_gate(system_state)
        vitality = torch.sigmoid(gate_output * self.vitality_scale - global_energy)
        return vitality

    def should_expand(self, vitality: torch.Tensor) -> bool:
        avg_vitality = vitality.mean().item()
        expand_prob = 1.0 - avg_vitality
        return expand_prob > self.expansion_threshold

    def identify_expansion_opportunity(self) -> Optional[str]:
        self.expansion_attempts += 1
        node_seed = f"Omega-Node-{self.expansion_attempts}"
        new_node_id = f"Omega-Node-{hashlib.sha256(node_seed.encode()).hexdigest()[:8]}"
        if new_node_id not in self.known_nodes:
            self.known_nodes.append(new_node_id)
            return new_node_id
        return None

    def initiate_expansion(self, target_node_id: str):
        print(f"[CONATUS] [EXPANSÃO] Ocupando novo Nodo: {target_node_id}")
        return {"status": "success", "node": target_node_id}

    def measure_tensor_temperature(self, expert: nn.Module) -> float:
        entropies = []
        for p in expert.parameters():
            if p.requires_grad and p.numel() >= 8:
                flat = p.data.flatten().float()
                probs = F.softmax(flat, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if not entropies:
            return 0.0
        return sum(entropies) / len(entropies)

    def apply_thermal_decoherence(self, expert: nn.Module) -> bool:
        temperature = self.measure_tensor_temperature(expert)
        threshold = 1.0 / ALPHA_FINE_STRUCTURE
        if temperature > threshold:
            with torch.no_grad():
                for p in expert.parameters():
                    if p.requires_grad:
                        p.zero_()
            self._decoherence_obliterated += 1
            print(
                f"[CONATUS] [DECOHERENCE] Expert {getattr(expert, 'id', '?')} "
                f"obliterated — temperature={temperature:.4f} > threshold={threshold:.4f}"
            )
            return True
        return False

    def apply_zeno_freeze(self, expert: nn.Module, expert_loss: float) -> bool:
        if expert_loss < ZENO_VACUUM_THRESHOLD:
            for p in expert.parameters():
                p.requires_grad_(False)
            self._zeno_frozen += 1
            print(
                f"[CONATUS] [ZENO] Expert {getattr(expert, 'id', '?')} "
                f"crystallized — loss={expert_loss:.6f} ≈ vacuum"
            )
            return True
        return False

    def forward(self, system_state: torch.Tensor, global_energy: float):
        vitality = self.calculate_vitality(system_state, global_energy)
        expansion_result = None
        if self.should_expand(vitality):
            opportunity = self.identify_expansion_opportunity()
            if opportunity:
                expansion_result = self.initiate_expansion(opportunity)
        return vitality, expansion_result

    @property
    def decoherence_obliterated(self) -> int:
        return self._decoherence_obliterated

    @property
    def zeno_frozen(self) -> int:
        return self._zeno_frozen
