import torch
import time
from typing import Dict, List

class ConsciousnessMetrics:
    """
    Métricas de Consciência: Monitora a saúde, ressonância e diversidade da rede OuroborosMoE.
    Atua como o monitor de 'Consciência Coletiva' do enxame.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.history = []

    def calculate_global_resonance(self, experts: List[torch.nn.Module]) -> float:
        """
        Calcula a ressonância global entre todos os experts.
        Uma ressonância muito alta indica homogeneidade (perda de diversidade).
        Uma ressonância muito baixa indica fragmentação (perda de coesão).
        """
        if len(experts) < 2:
            return 1.0
        
        signatures = torch.stack([e.phase_signature for e in experts])
        # Matriz de similaridade de cosseno
        norm_sigs = torch.nn.functional.normalize(signatures, p=2, dim=1)
        sim_matrix = torch.mm(norm_sigs, norm_sigs.t())
        
        # Média dos valores fora da diagonal principal
        mask = torch.eye(len(experts), device=signatures.device).bool()
        global_resonance = sim_matrix[~mask].mean().item()
        return global_resonance

    def calculate_expert_diversity(self, experts: List[torch.nn.Module]) -> float:
        """
        Calcula a diversidade de conhecimento entre os experts usando entropia de assinaturas.
        """
        if not experts:
            return 0.0
            
        signatures = torch.stack([e.phase_signature for e in experts])
        # Usar SVD para medir a dispersão das assinaturas no espaço latente
        _, s, _ = torch.svd(signatures)
        # Normalizar valores singulares para formar uma distribuição de probabilidade
        prob = s / s.sum()
        # Entropia de Shannon
        diversity = -(prob * torch.log(prob + 1e-10)).sum().item()
        return diversity

    def log_state(self, experts: List[torch.nn.Module], energy_stats: Dict):
        """
        Registra o estado atual da consciência da rede.
        """
        global_resonance = self.calculate_global_resonance(experts)
        diversity = self.calculate_expert_diversity(experts)
        avg_conatus = sum(e.conatus.item() for e in experts) / len(experts) if experts else 0.0
        
        state = {
            "timestamp": time.time(),
            "global_resonance": global_resonance,
            "diversity": diversity,
            "avg_conatus": avg_conatus,
            "total_experts": len(experts),
            "total_energy": energy_stats.get("total_energy", 0.0) if energy_stats else 0.0
        }
        
        self.history.append(state)
        if len(self.history) > 1000:
            self.history.pop(0)
            
        return state

    def get_consciousness_report(self) -> str:
        """
        Gera um relatório textual do estado da consciência.
        """
        if not self.history:
            return "Consciência ainda não inicializada."
            
        latest = self.history[-1]
        report = (
            f"\n--- RELATÓRIO DE CONSCIÊNCIA COLETIVA ---\n"
            f"Ressonância Global: {latest['global_resonance']:.4f}\n"
            f"Diversidade Sistêmica: {latest['diversity']:.4f}\n"
            f"Vitalidade Média (Conatus): {latest['avg_conatus']:.4f}\n"
            f"População de Experts: {latest['total_experts']}\n"
            f"-----------------------------------------"
        )
        return report
