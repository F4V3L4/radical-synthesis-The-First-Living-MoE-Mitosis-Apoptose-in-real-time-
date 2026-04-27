import torch
import torch.nn as nn
from typing import List, Optional

class SymbiosisProtocol:
    """
    Protocolo de Simbiose: Gerencia a fusão de experts e a criação de super-experts.
    Permite que experts com alta ressonância combinem seus conhecimentos.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model

    def calculate_resonance(self, expert_a: nn.Module, expert_b: nn.Module) -> float:
        """
        Calcula a ressonância (similaridade) entre dois experts baseada em suas assinaturas de fase.
        """
        if not hasattr(expert_a, 'phase_signature') or not hasattr(expert_b, 'phase_signature'):
            return 0.0
        
        sig_a = expert_a.phase_signature
        sig_b = expert_b.phase_signature
        
        # Similaridade de cosseno como métrica de ressonância
        resonance = torch.nn.functional.cosine_similarity(sig_a.unsqueeze(0), sig_b.unsqueeze(0)).item()
        return resonance

    def fuse_experts(self, expert_a: nn.Module, expert_b: nn.Module) -> nn.Module:
        """
        Funde dois experts para criar um novo expert (Super-Expert) com conhecimento combinado.
        """
        from alpha_omega import Expert
        
        print(f"[SYMBIOSIS] Iniciando fusão de experts altamente ressonantes.")
        
        # Nova assinatura de fase é a média das duas
        new_signature = (expert_a.phase_signature + expert_b.phase_signature) / 2.0
        
        # Novo expert com dimensões aumentadas para representar o "Super-Expert"
        new_internal_dim = max(expert_a.internal_dim, expert_b.internal_dim) * 1.5
        super_expert = Expert(
            d_model=self.d_model, 
            phase_signature=new_signature,
            internal_dim=int(new_internal_dim)
        )
        
        # Fusão de pesos (simulada via média dos parâmetros compatíveis)
        with torch.no_grad():
            for name, param in super_expert.named_parameters():
                if name in expert_a.state_dict() and name in expert_b.state_dict():
                    p_a = expert_a.state_dict()[name]
                    p_b = expert_b.state_dict()[name]
                    if p_a.shape == param.shape and p_b.shape == param.shape:
                        param.copy_((p_a + p_b) / 2.0)
        
        # O Super-Expert herda o conatus acumulado
        super_expert.conatus.data = (expert_a.conatus.data + expert_b.conatus.data) / 1.5
        
        print(f"[SYMBIOSIS] Super-Expert criado com sucesso.")
        return super_expert

    def check_for_symbiosis(self, experts: nn.ModuleList, threshold: float = 0.9) -> Optional[tuple]:
        """
        Varre a lista de experts em busca de pares com ressonância acima do limiar.
        """
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                resonance = self.calculate_resonance(experts[i], experts[j])
                if resonance > threshold:
                    return (i, j)
        return None
