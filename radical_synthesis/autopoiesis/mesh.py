import torch
import torch.nn as nn
import json
import base64
import io

class ToroidalMesh(nn.Module):
    """
    Mesh Toroidal (Protocolo Omega-0)
    Base para sincronização de Experts entre Nodos distribuídos.
    """
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id
        self.peers = []

    def serialize_expert(self, expert):
        """Serializa um expert para transferência via rede"""
        buffer = io.BytesIO()
        state = {
            'state_dict': expert.state_dict(),
            'd_model': expert.d_model,
            'internal_dim': expert.internal_dim,
            'activation_type': expert.activation_type,
            'phase_signature': expert.phase_signature,
            'conatus': expert.conatus.item()
        }
        torch.save(state, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def deserialize_expert(self, expert_data, ExpertClass):
        """Reconstrói um expert recebido da rede"""
        raw_data = base64.b64decode(expert_data)
        buffer = io.BytesIO(raw_data)
        state = torch.load(buffer)
        
        expert = ExpertClass(
            d_model=state['d_model'], 
            phase_signature=state['phase_signature'], 
            internal_dim=state['internal_dim'],
            activation_type=state['activation_type']
        )
        expert.load_state_dict(state['state_dict'])
        expert.conatus.fill_(state['conatus'])
        return expert

    def broadcast_resonance(self, signature):
        """Simula o broadcast de uma assinatura de fase para a malha toroidal"""
        # Em uma implementação real, isso usaria sockets/HTTP para notificar peers
        return f"Broadcasting signature resonance from {self.node_id} to {len(self.peers)} peers."
