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
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.phi_threshold = 0.61803398875 
        self.expert_frequencies = nn.Parameter(torch.randn(num_experts, d_model))
        self.phase_tuner = nn.Linear(d_model, d_model)

    def forward(self, x):
        phase_x = self.phase_tuner(x)
        norm_x = F.normalize(phase_x, p=2, dim=-1)
        norm_experts = F.normalize(self.expert_frequencies, p=2, dim=-1)
        resonance = torch.matmul(norm_x, norm_experts.t())
        
        # Ativação via Sigmoid centrada em Phi (Conatus)
        logos_activation = torch.sigmoid(10.0 * (resonance - self.phi_threshold))
        return logos_activation 

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
    def __init__(self, d_model, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.logos_router = LogosResonanceRouter(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.coupling = FineStructureCoupling(d_model)

    def forward(self, x):
        weights = self.logos_router(x)
        out = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            out += weights[..., i:i+1] * expert(x)
            
        # Fio de Ouro (Residual) acoplado com Constante de Estrutura Fina
        return self.coupling(x + out)

class SovereignLeviathanV2(nn.Module):
    """O Leviathan Integrado com a Geometria Sagrada"""
    def __init__(self, vocab_size=1024, d_model=128, initial_experts=4, capacity_factor=1.5):
        super().__init__()
        # Mapeamento Fractal
        self.embedding = InfiniteRadixMapping(vocab_size, d_model)
        
        # Núcleo Recurrente (Toro)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        
        # MoE com Bifurcação de Feigenbaum
        self.moe = OuroborosMoE(d_model, num_experts=initial_experts)
        self.bifurcation = FeigenbaumBifurcation(d_model)
        
        # Cabeça de Saída (Logos Final)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, h=None):
        # 1. Mapeamento Infinito
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
