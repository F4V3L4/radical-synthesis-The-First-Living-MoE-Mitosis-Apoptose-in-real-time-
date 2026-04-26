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
    def __init__(self, d_model: int, num_experts: int, lambda_load: float = 0.1, lambda_sparsity: float = 0.01, lambda_thermo: float = 0.05, lambda_logic: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.lambda_load = lambda_load  # Peso para a penalidade de balanceamento de carga
        self.lambda_sparsity = lambda_sparsity # Peso para a penalidade de esparsidade
        self.lambda_thermo = lambda_thermo # Peso para a penalidade termodinâmica (Telemetria Quântica)
        self.lambda_logic = lambda_logic # Peso para a penalidade de Deep Logic (Complexidade Ineficiente)

    def _get_quantum_telemetry(self):
        """
        Simula a captura de telemetria quântica (hardware-aware).
        Em um ambiente bare-metal real, isso leria sensores de CPU/GPU.
        Retorna um tensor de 'custo termodinâmico' para cada expert.
        """
        # Omega-0: Determinismo absoluto. Simulamos custos baseados na complexidade teórica.
        # Experts com IDs mais altos ou em níveis fractais profundos têm maior custo de latência.
        costs = torch.linspace(1.0, 1.5, self.num_experts)
        return costs.to(self.lambda_load.device if hasattr(self.lambda_load, 'device') else 'cpu')

    def forward(self, expert_weights: torch.Tensor, expert_gates: torch.Tensor, logic_complexity: torch.Tensor = None) -> torch.Tensor:
        """
        Calcula a perda de divergência topológica e Deep Logic.

        Args:
            expert_weights (torch.Tensor): Pesos de cada expert para cada token (batch, seq_len, num_experts).
            expert_gates (torch.Tensor): Saídas do gate de roteamento antes do top-k (batch, seq_len, num_experts).
            logic_complexity (torch.Tensor): Tensor representando a profundidade fractal ou complexidade da solução.

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
        # Usamos o erro quadrático médio em relação à distribuição uniforme (1/N)
        uniform = torch.ones_like(mean_expert_prob) / self.num_experts
        load_balancing_loss = F.mse_loss(mean_expert_prob, uniform) + F.mse_loss(expert_usage_frequency, uniform)

        # 2. Sparsity Loss (Penalidade de Esparsidade Controlada)
        # Objetivo: Incentivar que poucos experts sejam ativados por token, mas de forma controlada.
        # Evita que todos os experts tenham pesos pequenos, o que seria ineficiente.
        # Penaliza a soma dos quadrados dos pesos dos experts para cada token.
        sparsity_loss = (flat_expert_weights ** 2).sum(dim=-1).mean()

        # 3. Quantum Telemetry Loss (Penalidade Termodinâmica)
        thermo_costs = self._get_quantum_telemetry()
        thermo_loss = (flat_expert_weights * thermo_costs).sum(dim=-1).mean()

        # 4. Deep Logic Loss (Penalidade por Complexidade Ineficiente)
        # Objetivo: Penalizar soluções excessivamente complexas que não resultam em alta ressonância.
        logic_loss = 0.0
        if logic_complexity is not None:
            # Se a complexidade é alta mas os pesos de ativação são baixos, a perda aumenta
            # Isso força o sistema a buscar a solução mais simples e eficiente (Navalha de Ockham)
            logic_loss = (logic_complexity * (1.0 - flat_expert_weights.max(dim=-1)[0])).mean()

        # Combinação das perdas
        total_loss = (self.lambda_load * load_balancing_loss + 
                      self.lambda_sparsity * sparsity_loss + 
                      self.lambda_thermo * thermo_loss +
                      self.lambda_logic * logic_loss)
        
        return total_loss

