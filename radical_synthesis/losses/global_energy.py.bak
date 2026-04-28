import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalEnergyFunction(nn.Module):
    """
    O 9 Central: Função de Energia Global que unifica a métrica de verdade do sistema.
    Unifica Loss (Erro), Entropia (Caos) e Estabilidade (Conatus).
    """
    def __init__(self, lambda_loss=1.0, lambda_entropy=0.1, lambda_stability=0.05):
        super().__init__()
        self.lambda_loss = lambda_loss
        self.lambda_entropy = lambda_entropy
        self.lambda_stability = lambda_stability

    def forward(self, model_loss, expert_weights, conatus_levels):
        """
        Calcula a Energia Global do Sistema.
        E = λ1*Loss + λ2*Entropy - λ3*Stability
        """
        # 1. Entropia do Roteamento (Mede o caos na decisão)
        # expert_weights: [batch, seq, num_experts]
        prob_dist = expert_weights.mean(dim=(0, 1))
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10))
        
        # 2. Estabilidade (Mede a saúde do Conatus)
        stability = torch.mean(conatus_levels)
        
        # 3. Energia Total (O objetivo é minimizar a energia)
        # Queremos baixa loss, baixa entropia e alta estabilidade
        total_energy = (self.lambda_loss * model_loss + 
                        self.lambda_entropy * entropy - 
                        self.lambda_stability * stability)
        
        return {
            "total_energy": total_energy,
            "entropy": entropy,
            "stability": stability,
            "model_loss": model_loss
        }
