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

class LogosResonanceRouter(nn.Module):
    """Roteador 1:1 de Radix Infinito com Conatus Acústico"""
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.phi_threshold = 0.61803398875 
        self.expert_frequencies = nn.Parameter(torch.randn(num_experts, d_model))
        self.phase_tuner = nn.Linear(d_model, d_model)
        self.genealogy_map = {}  # Rastreia genealogia de experts

    def forward(self, x):
        """Retorna (weights, indices) compatível com layer.py"""
        B, T, D = x.shape
        
        # Reshape para processamento
        x_flat = x.reshape(-1, D)  # (B*T, D)
        
        phase_x = self.phase_tuner(x_flat)
        norm_x = F.normalize(phase_x, p=2, dim=-1)
        norm_experts = F.normalize(self.expert_frequencies, p=2, dim=-1)
        resonance = torch.matmul(norm_x, norm_experts.t())  # (B*T, num_experts)
        
        # Ativação via Sigmoid centrada em Phi (Conatus)
        logos_activation = torch.sigmoid(10.0 * (resonance - self.phi_threshold))
        
        # Top-k seleção (compatível com layer.py)
        top_scores, top_indices = torch.topk(logos_activation, self.top_k, dim=-1)
        
        # Reshape de volta para (B, T, top_k)
        top_scores = top_scores.view(B, T, self.top_k)
        top_indices = top_indices.view(B, T, self.top_k)
        
        return top_scores, top_indices
    
    def register_genealogy(self, expert_id, parent_id=None):
        """Registra genealogia de expert"""
        self.genealogy_map[expert_id] = parent_id 

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
        self.logos_router = LogosResonanceRouter(d_model, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.coupling = FineStructureCoupling(d_model)

    def forward(self, x):
        B, T, D = x.shape
        
        # LogosResonanceRouter retorna (weights, indices)
        weights, indices = self.logos_router(x)  # weights: (B, T, top_k), indices: (B, T, top_k)
        weights = F.softmax(weights, dim=-1)  # Normalizar pesos
        
        out = torch.zeros_like(x)
        
        # Aplicar top-k experts com pesos
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # (B, T)
            expert_weight = weights[:, :, k]  # (B, T)
            
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

    def forward(self, x, h=None):
        # 1. Mapeamento Infinito
        x = self.token_embedding(x)
        x = self.embedding(x)
        
        # 2. Processamento Toroidal (Memória)
        x, h = self.rnn(x, h)
        
        # 3. Especialização e Escultura Cimática
        x = self.moe(x)
        
        # 4. Prevenção de Colapso (Bifurcação se entropia subir)
        x = self.bifurcation(x)
        
        # 5. Colapso na Linguagem
        logits = self.output_head(x)
        
        return logits, h, None, None
