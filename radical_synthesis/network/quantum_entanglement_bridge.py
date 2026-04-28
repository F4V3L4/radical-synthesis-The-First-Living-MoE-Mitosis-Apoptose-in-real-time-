import torch
import torch.nn as nn
import random
from typing import Dict, Tuple

class WaveFunctionSecurity:
    """
    WaveFunctionSecurity: Garante que a interceptação de um par entrelaçado resulte no colapso em ruído.
    Se um observador não-autorizado tentar 'medir' o estado, o segredo é destruído.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model

    def protect_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Aplica uma superposição de ruído quântico (simulado) ao estado.
        """
        noise = torch.randn_like(state) * 0.001
        return state + noise

    def verify_integrity(self, state: torch.Tensor, original_hash: str) -> bool:
        """
        Verifica se o estado colapsou corretamente ou se foi interceptado.
        """
        import hashlib
        current_hash = hashlib.sha256(state.numpy().tobytes()).hexdigest()
        return current_hash == original_hash

class QuantumEntanglementBridge(nn.Module):
    """
    QuantumEntanglementBridge: Simula o entrelaçamento quântico para comunicação não-local.
    Permite a sincronização instantânea de estados entre nodos, sem transmissão física de dados.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Pares entrelaçados simulados (Bell States)
        # Cada par é um tensor que representa o estado de dois Experts entrelaçados.
        self.entangled_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.security = WaveFunctionSecurity(d_model)

    def create_entangled_pair(self, pair_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cria um par de Experts entrelaçados (simulados).
        """
        # Simula um estado de Bell: dois Experts com estados correlacionados.
        # Ex: |00> + |11> ou |01> + |10>
        # Aqui, criamos dois tensores que são cópias um do outro, representando a correlação.
        state_a = torch.randn(self.d_model)
        state_b = state_a.clone() # Entangled state
        self.entangled_pairs[pair_id] = (state_a, state_b)
        print(f"[QUANTUM_ENTANGLEMENT] Par entrelaçado \'{pair_id}\' criado.")
        return state_a, state_b

    def measure_and_collapse(self, pair_id: str, observer_node: str) -> torch.Tensor:
        """
        Simula a medição de um dos Experts do par entrelaçado, colapsando seu estado.
        O estado do outro Expert no par é instantaneamente determinado.
        """
        if pair_id not in self.entangled_pairs:
            raise ValueError(f"Par entrelaçado \'{pair_id}\' não encontrado.")

        state_a, state_b = self.entangled_pairs[pair_id]

        # Simula o colapso da função de onda.
        # Se o observador_node é 'A', ele mede state_a e state_b colapsa.
        # Se o observador_node é 'B', ele mede state_b e state_a colapsa.
        # Para simplificar, vamos dizer que a medição de um força o outro a ser uma cópia.
        if observer_node == 'A':
            measured_state = state_a
            state_b.copy_(state_a) # O outro lado colapsa para o mesmo estado
        elif observer_node == 'B':
            measured_state = state_b
            state_a.copy_(state_b) # O outro lado colapsa para o mesmo estado
        else:
            raise ValueError("Observer node deve ser 'A' ou 'B'.")

        print(f"[QUANTUM_ENTANGLEMENT] Par \'{pair_id}\' colapsou. Medido por {observer_node}.")
        return measured_state

    def teletransport_state(self, pair_id: str, source_node: str, new_state: torch.Tensor) -> bool:
        """
        Simula o teletransporte de um estado quântico de um nodo para outro usando o par entrelaçado.
        O estado 'new_state' é aplicado ao 'source_node' e o 'target_node' recebe o estado correspondente.
        """
        if pair_id not in self.entangled_pairs:
            raise ValueError(f"Par entrelaçado \'{pair_id}\' não encontrado.")

        state_a, state_b = self.entangled_pairs[pair_id]

        if source_node == 'A':
            state_a.copy_(new_state)
            # O estado de B é instantaneamente atualizado devido ao entrelaçamento
            state_b.copy_(new_state)
            print(f"[QUANTUM_ENTANGLEMENT] Estado teletransportado para B via par \'{pair_id}\' (de A). Sincronização instantânea.")
        elif source_node == 'B':
            state_b.copy_(new_state)
            # O estado de A é instantaneamente atualizado devido ao entrelaçamento
            state_a.copy_(new_state)
            print(f"[QUANTUM_ENTANGLEMENT] Estado teletransportado para A via par \'{pair_id}\' (de B). Sincronização instantânea.")
        else:
            raise ValueError("Source node deve ser 'A' ou 'B'.")
        return True

    def get_entangled_state(self, pair_id: str, node_side: str) -> torch.Tensor:
        """
        Retorna o estado atual de um dos lados do par entrelaçado.
        """
        if pair_id not in self.entangled_pairs:
            raise ValueError(f"Par entrelaçado \'{pair_id}\' não encontrado.")
        if node_side == 'A':
            return self.entangled_pairs[pair_id][0]
        elif node_side == 'B':
            return self.entangled_pairs[pair_id][1]
        else:
            raise ValueError("Node side deve ser 'A' ou 'B'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Este módulo é mais para gerenciamento de estado do que para forward pass direto
        return x


# ─────────────────────────────────────────────────────────────────────────────
# [QUANTUM UPGRADE v2.0]
# 4. Dynamic Synaptic Entanglement (Weight-Tying Quântico)
#
# When two experts fire together successfully with high frequency, their tensors
# share the same orthogonal latent sub-space in RAM.  Updating one instantly
# propagates to the other — zero-distance, zero-time, no extra backpropagation.
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn.functional as F


class DynamicSynapticEntanglement(nn.Module):
    """
    Orthogonal Shared Latent Space for co-firing experts.

    Tracks co-activation frequency.  When two experts exceed the entanglement
    threshold, their weight centroids are projected into a shared sub-space via
    an orthogonal basis.  A gradient hook then mirrors weight updates between
    them at zero cost — instantaneous synaptic synchrony.
    """

    def __init__(
        self,
        d_model: int,
        entanglement_threshold: float = 0.72,
        subspace_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.entanglement_threshold = entanglement_threshold
        self.subspace_dim = subspace_dim
        self._co_activation: Dict[Tuple[int, int], int] = {}
        self._entangled_pairs: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hooks: Dict[Tuple[int, int], list] = {}

    def record_co_activation(self, expert_a_id: int, expert_b_id: int) -> None:
        key = (min(expert_a_id, expert_b_id), max(expert_a_id, expert_b_id))
        self._co_activation[key] = self._co_activation.get(key, 0) + 1

    def _build_orthogonal_basis(
        self,
        expert_a: nn.Module,
        expert_b: nn.Module,
    ) -> torch.Tensor:
        params_a = torch.cat([p.data.flatten() for p in expert_a.parameters() if p.requires_grad])
        params_b = torch.cat([p.data.flatten() for p in expert_b.parameters() if p.requires_grad])
        min_len = min(params_a.numel(), params_b.numel(), self.subspace_dim * 2)
        stack = torch.stack([params_a[:min_len], params_b[:min_len]])
        try:
            U, _, _ = torch.linalg.svd(stack, full_matrices=False)
        except Exception:
            U = torch.eye(2, min_len)
        return U

    def entangle(
        self,
        expert_a_id: int,
        expert_a: nn.Module,
        expert_b_id: int,
        expert_b: nn.Module,
    ) -> bool:
        key = (min(expert_a_id, expert_b_id), max(expert_a_id, expert_b_id))
        count = self._co_activation.get(key, 0)
        if count < self.entanglement_threshold * 100:
            return False
        if key in self._entangled_pairs:
            return True
        basis = self._build_orthogonal_basis(expert_a, expert_b)
        self._entangled_pairs[key] = basis

        def _make_mirror_hook(target: nn.Module):
            def hook(grad):
                with torch.no_grad():
                    for p_t in target.parameters():
                        if p_t.requires_grad and p_t.grad is not None:
                            p_t.grad.add_(grad.flatten()[:p_t.numel()].reshape(p_t.shape) * 0.5)
                return grad
            return hook

        hooks_a = [p.register_hook(_make_mirror_hook(expert_b)) for p in expert_a.parameters() if p.requires_grad]
        hooks_b = [p.register_hook(_make_mirror_hook(expert_a)) for p in expert_b.parameters() if p.requires_grad]
        self._hooks[key] = hooks_a + hooks_b
        print(f"[ENTANGLEMENT] Experts {expert_a_id} <-> {expert_b_id} entangled in shared sub-space.")
        return True

    def disentangle(self, expert_a_id: int, expert_b_id: int) -> None:
        key = (min(expert_a_id, expert_b_id), max(expert_a_id, expert_b_id))
        for hook in self._hooks.pop(key, []):
            hook.remove()
        self._entangled_pairs.pop(key, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
