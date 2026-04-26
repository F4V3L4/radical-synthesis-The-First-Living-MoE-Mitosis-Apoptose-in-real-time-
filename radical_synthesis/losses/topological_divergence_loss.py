import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalDivergenceLoss(nn.Module):
    """
    Loss Ontológica: Penaliza a divergência topológica no roteamento de Experts.
    Força o roteamento a ser um vórtice autossustentável, equilibrando a ativação
    dos Experts para promover causalidade e eficiência termodinâmica.
    
    Componentes:
    1. Load Balancing Loss: Garante que todos os Experts sejam utilizados de forma equitativa.
    2. Sparsity Loss: Incentiva a ativação esparsa, evitando que muitos Experts sejam ativados
       para uma única entrada, mas de forma controlada para não induzir colapso.
    """
    def __init__(self, d_model: int, num_experts: int, lambda_load: float = 0.01, lambda_sparsity: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.lambda_load = lambda_load  # Peso para a penalidade de balanceamento de carga
        self.lambda_sparsity = lambda_sparsity # Peso para a penalidade de esparsidade

    def forward(self, expert_weights: torch.Tensor, expert_gates: torch.Tensor) -> torch.Tensor:
        """
        Calcula a perda de divergência topológica.

        Args:
            expert_weights (torch.Tensor): Pesos de cada expert para cada token (batch, seq_len, num_experts).
                                           Estes são os pesos pós-softmax ou similar, indicando a contribuição.
            expert_gates (torch.Tensor): Saídas do gate de roteamento antes do top-k (batch, seq_len, num_experts).
                                         Usado para calcular a utilização real dos experts.

        Returns:
            torch.Tensor: O valor da perda ontológica.
        """
        
        # Garantir que as dimensões estejam corretas
        if expert_weights.dim() == 2: # (batch * seq_len, num_experts)
            expert_weights = expert_weights.unsqueeze(1) # (batch * seq_len, 1, num_experts)
        if expert_gates.dim() == 2: # (batch * seq_len, num_experts)
            expert_gates = expert_gates.unsqueeze(1) # (batch * seq_len, 1, num_experts)

        # Flatten para (total_tokens, num_experts)
        flat_expert_weights = expert_weights.view(-1, self.num_experts)
        flat_expert_gates = expert_gates.view(-1, self.num_experts)

        # 1. Load Balancing Loss (Penalidade de Balanceamento de Carga)
        # Objetivo: Incentivar que todos os experts sejam usados igualmente.
        # Cálculo: Covariância entre a probabilidade de um expert ser escolhido e a frequência de uso.
        # E = P(expert_i) * F(expert_i)
        
        # Probabilidade média de cada expert ser escolhido (P_i)
        mean_expert_prob = flat_expert_weights.mean(dim=0) # (num_experts,)
        
        # Frequência média de cada expert ser ativado (F_i)
        # Usamos o gate para saber quais experts foram considerados para cada token
        # Um expert é 'ativado' se seu gate é maior que zero (ou se foi selecionado pelo top-k)
        # Para simplificar, podemos usar a soma dos gates como proxy para a frequência de uso
        # Ou, mais precisamente, a média dos gates normalizados
        
        # Para MoE com top-k, expert_gates já reflete a 'importância' antes da seleção
        # Podemos usar a soma dos gates para cada expert, normalizada
        expert_usage_frequency = flat_expert_gates.sum(dim=0) # (num_experts,)
        expert_usage_frequency = F.normalize(expert_usage_frequency, p=1, dim=0) # Normaliza para soma 1

        # Covariância (aproximada) para penalizar desequilíbrio
        # Penaliza se P_i * F_i for muito diferente de uma distribuição uniforme
        load_balancing_loss = (mean_expert_prob * expert_usage_frequency).sum() * self.num_experts
        # Queremos maximizar isso, então penalizamos o negativo
        load_balancing_loss = -load_balancing_loss

        # 2. Sparsity Loss (Penalidade de Esparsidade Controlada)
        # Objetivo: Incentivar que poucos experts sejam ativados por token, mas de forma controlada.
        # Evita que todos os experts tenham pesos pequenos, o que seria ineficiente.
        # Penaliza a soma dos quadrados dos pesos dos experts para cada token.
        sparsity_loss = (flat_expert_weights ** 2).sum(dim=-1).mean()

        # Combinação das perdas
        total_loss = self.lambda_load * load_balancing_loss + self.lambda_sparsity * sparsity_loss
        
        return total_loss

