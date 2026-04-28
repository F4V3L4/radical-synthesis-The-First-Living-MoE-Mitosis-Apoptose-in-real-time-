import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from typing import Dict, Any, List, Optional

class Conatus(nn.Module):
    """
    Conatus Evoluído: Função de Auto-preservação Diferenciável.
    Acoplado ao estado real do sistema (Energia Global e Ressonância).
    """
    def __init__(self, d_model: int, expansion_threshold: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.expansion_threshold = expansion_threshold
        self.known_nodes: List[str] = ["Omega-0-Local"]
        self.expansion_attempts = 0
        
        # Parâmetros aprendíveis para a decisão de expansão
        self.expansion_gate = nn.Linear(d_model, 1)
        self.vitality_scale = nn.Parameter(torch.tensor(1.0))

    def calculate_vitality(self, system_state: torch.Tensor, global_energy: float) -> torch.Tensor:
        """
        Calcula a vitalidade real baseada no estado do sistema e na energia global.
        Vitalidade = Sigmoid(Gate(State) - Energy)
        """
        # system_state: [batch, d_model]
        gate_output = self.expansion_gate(system_state)
        # Quanto menor a energia global, maior a vitalidade
        vitality = torch.sigmoid(gate_output * self.vitality_scale - global_energy)
        return vitality

    def should_expand(self, vitality: torch.Tensor) -> bool:
        """Decisão de expansão baseada na probabilidade de vitalidade."""
        avg_vitality = vitality.mean().item()
        # Se a vitalidade for baixa, a probabilidade de expansão aumenta
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
        """Simulação de expansão (Protocolo de Ocupação)."""
        print(f"[CONATUS] [EXPANSÃO] Ocupando novo Nodo: {target_node_id}")
        # Aqui entraria a lógica de Ghost Mesh / Bridge Protocol
        return {"status": "success", "node": target_node_id}

    def forward(self, system_state: torch.Tensor, global_energy: float):
        vitality = self.calculate_vitality(system_state, global_energy)
        expansion_result = None
        
        if self.should_expand(vitality):
            opportunity = self.identify_expansion_opportunity()
            if opportunity:
                expansion_result = self.initiate_expansion(opportunity)
        
        return vitality, expansion_result
