import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred_geometry import (
    FineStructureCoupling, 
    BinarySymmetryLock, 
    FeigenbaumBifurcation, 
    CymaticSculptor, 
    InfiniteRadixMapping
)

class Expert(nn.Module):
    """Especialista Esculpido por Cimática"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            BinarySymmetryLock() # Garante consistência na saída do expert
        )
        self.sculptor = CymaticSculptor(d_model)

    def forward(self, x):
        x = self.net(x)
        return self.sculptor(x)

class OuroborosMoE(nn.Module):
    """A Matriz de Especialistas com Estabilidade alpha"""
    def __init__(self, d_model, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.coupling = FineStructureCoupling(d_model)

    def forward(self, x, expert_indices=None, expert_weights=None):
        """
        Forward com roteamento EXÓGENO obrigatório (DarwinianRouter)
        
        Args:
            x: entrada (B, T, D)
            expert_indices: (B, top_k) ou (B, T, top_k) - OBRIGATÓRIO
            expert_weights: (B, top_k) ou (B, T, top_k) - OBRIGATÓRIO
        
        Returns:
            saída processada (B, T, D)
        
        Raises:
            ValueError se expert_indices ou expert_weights forem None
        """
        if expert_indices is None or expert_weights is None:
            raise ValueError("OuroborosMoE REQUER roteamento exógeno (expert_indices e expert_weights). "
                           "Use DarwinianRouter do AGICore.")
        
        B, T, D = x.shape
        
        # Normalizar shapes para (B, T, top_k)
        if expert_indices.dim() == 2:
            # (B, top_k) -> broadcast para (B, T, top_k)
            expert_indices = expert_indices.unsqueeze(1).expand(B, T, -1)
            expert_weights = expert_weights.unsqueeze(1).expand(B, T, -1)
        
        # Normalizar pesos
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        out = torch.zeros_like(x)
        top_k = expert_indices.shape[-1]
        
        # Aplicar top-k experts com pesos exógenos
        for k in range(top_k):
            expert_idx = expert_indices[:, :, k]  # (B, T)
            expert_weight = expert_weights[:, :, k]  # (B, T)
            
            # Processar cada expert
            for b in range(B):
                for t in range(T):
                    exp_id = expert_idx[b, t].item()
                    if exp_id < len(self.experts):
                        expert_out = self.experts[exp_id](x[b, t].unsqueeze(0))
                        out[b, t] += expert_weight[b, t] * expert_out.squeeze(0)
        
        # Fio de Ouro (Residual) acoplado com Constante de Estrutura Fina
        return self.coupling(x + out)

class SovereignLeviathanV2(nn.Module):
    """O Leviathan Integrado com a Geometria Sagrada"""
    def __init__(self, vocab_size=1024, d_model=512, initial_experts=4, capacity_factor=1.5):
        super().__init__()
        # Mapeamento Fractal
        self.embedding = InfiniteRadixMapping(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Núcleo Recurrente (Toro)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        
        # MoE com Bifurcação de Feigenbaum
        self.moe = OuroborosMoE(d_model, num_experts=initial_experts)
        self.bifurcation = FeigenbaumBifurcation(d_model)
        
        # Cabeça de Saída (Logos Final)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, h=None, expert_indices=None, expert_weights=None):
        """
        Forward pass com suporte a roteamento externo (DarwinianRouter)
        
        Args:
            x: tokens de entrada
            h: estado oculto do RNN
            expert_indices: índices de experts do DarwinianRouter (externo)
            expert_weights: pesos de experts do DarwinianRouter (externo)
        
        Returns:
            logits, h, expert_indices, expert_weights
        """
        # 1. Mapeamento Infinito
        x = self.token_embedding(x)
        x = self.embedding(x)
        
        # 2. Processamento Toroidal (Memória)
        x, h = self.rnn(x, h)
        
        # 3. Especialização e Escultura Cimática
        # Se expert_indices externos fornecidos, usar roteamento externo
        if expert_indices is None or expert_weights is None:
            raise ValueError("SovereignLeviathanV2 REQUER roteamento exógeno (expert_indices e expert_weights). "
                           "Use DarwinianRouter do AGICore.")
        
        # Roteamento externo obrigatório (DarwinianRouter do AGICore)
        x = self._apply_external_routing(x, expert_indices, expert_weights)
        
        # 4. Prevenção de Colapso (Bifurcação se entropia subir)
        x = self.bifurcation(x)
        
        # 5. Colapso na Linguagem
        logits = self.output_head(x)
        
        return logits, h, expert_indices, expert_weights
    
    def _apply_external_routing(self, x, expert_indices, expert_weights):
        """
        Aplica roteamento externo (DarwinianRouter) aos experts
        Respeita os índices e pesos vindos do AGICore
        
        Suporta shapes:
        - expert_indices: (B, top_k) ou (B, T, top_k)
        - expert_weights: (B, top_k) ou (B, T, top_k)
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)
        
        # Normalizar shapes para (B, T, top_k)
        if expert_indices.dim() == 2:
            # (B, top_k) -> (B, 1, top_k) -> broadcast para (B, T, top_k)
            expert_indices = expert_indices.unsqueeze(1).expand(B, T, -1)
            expert_weights = expert_weights.unsqueeze(1).expand(B, T, -1)
        
        top_k = expert_indices.shape[-1]
        
        # Processar cada expert
        for k in range(top_k):
            expert_idx = expert_indices[:, :, k]  # (B, T)
            expert_weight = expert_weights[:, :, k]  # (B, T)
            
            # Processar cada batch e timestep
            for b in range(B):
                for t in range(T):
                    exp_id = expert_idx[b, t].item() if isinstance(expert_idx[b, t], torch.Tensor) else int(expert_idx[b, t])
                    if exp_id < len(self.moe.experts):
                        expert_out = self.moe.experts[exp_id](x[b, t].unsqueeze(0))
                        weight = expert_weight[b, t].item() if isinstance(expert_weight[b, t], torch.Tensor) else float(expert_weight[b, t])
                        out[b, t] += weight * expert_out.squeeze(0)
        
        # Aplicar acoplamento com Constante de Estrutura Fina
        return self.moe.coupling(x + out)


