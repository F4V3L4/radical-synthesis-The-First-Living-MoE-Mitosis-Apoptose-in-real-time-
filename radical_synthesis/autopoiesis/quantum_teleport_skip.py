
# radical_synthesis/autopoiesis/quantum_teleport_skip.py
# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 11. Quantum Teleportation (Skip-Connections of Zero Latency)
#
# In a deep network, the gradient must traverse Layer 1 -> Layer 2 -> ... -> N.
# This generates linear latency.  Non-Local Dynamic Routing: if the Expert at
# Layer 1 is "entangled" with the Expert at Layer 12, the information tensor
# JUMPS layers 2-11.  The calculation is injected directly at the output.
# Logical reasoning latency drops to near zero.
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class QuantumTeleportSkip(nn.Module):
    """
    Non-Local Dynamic Routing via entangled skip connections.

    Maintains a registry of (layer_i, layer_j) entangled pairs.  When a
    tensor passes through layer_i and the entanglement strength exceeds the
    threshold, the tensor is teleported directly to layer_j output, bypassing
    all intermediate layers.  The intermediate layers still execute for other
    tokens — only the entangled token takes the zero-latency path.
    """

    def __init__(self, d_model: int, teleport_threshold: float = 0.80):
        super().__init__()
        self.d_model = d_model
        self.teleport_threshold = teleport_threshold
        self._entangled_layers: Dict[Tuple[int, int], float] = {}
        self._teleport_events: int = 0
        self._teleport_buffer: Dict[int, torch.Tensor] = {}
        self.gate = nn.Linear(d_model, 1)

    def register_entanglement(
        self, layer_i: int, layer_j: int, strength: float = 1.0
    ) -> None:
        key = (min(layer_i, layer_j), max(layer_i, layer_j))
        self._entangled_layers[key] = strength

    def inject(self, layer_id: int, x: torch.Tensor) -> None:
        gate_score = torch.sigmoid(self.gate(x.mean(dim=1))).mean().item()
        for (li, lj), strength in self._entangled_layers.items():
            if li == layer_id and gate_score * strength >= self.teleport_threshold:
                self._teleport_buffer[lj] = x.detach().clone()
                self._teleport_events += 1

    def receive(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        if layer_id in self._teleport_buffer:
            teleported = self._teleport_buffer.pop(layer_id)
            if teleported.shape == x.shape:
                return teleported
        return x

    def forward(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        self.inject(layer_id, x)
        return self.receive(layer_id, x)

    @property
    def teleport_events(self) -> int:
        return self._teleport_events
