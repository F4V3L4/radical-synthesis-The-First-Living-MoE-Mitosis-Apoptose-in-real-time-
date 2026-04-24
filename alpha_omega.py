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
    """Especialista Esculpido por Cimática com Dinâmica de Conatus (Protocolo Mythos-Capybara)"""
    def __init__(self, d_model, phase_signature=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            BinarySymmetryLock() 
        )
        self.sculptor = CymaticSculptor(d_model)
        
        # Conatus Variable: Systemic Energy (Non-trainable)
        self.register_buffer('conatus', torch.tensor(1.0))
        
        # Phase Signature for Resonance
        if phase_signature is None:
            phase_signature = torch.randn(d_model)
        self.register_buffer('phase_signature', F.normalize(phase_signature, p=2, dim=-1))

    def forward(self, x):
        x = self.net(x)
        return self.sculptor(x)

    def update_conatus(self, resonated: bool, decay=0.01, growth=0.1):
        if resonated:
            self.conatus += growth
        else:
            self.conatus -= decay
        self.conatus = torch.clamp(self.conatus, min=0.0)

class OuroborosMoE(nn.Module):
    """A Matriz de Especialistas com Estabilidade alpha e Evolução Darwiniana"""
    def __init__(self, d_model, num_experts=4, mitosis_threshold=3.0, apoptosis_threshold=0.1):
        super().__init__()
        self.d_model = d_model
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.coupling = FineStructureCoupling(d_model)
        self.mitosis_threshold = mitosis_threshold
        self.apoptosis_threshold = apoptosis_threshold

    def forward(self, x, expert_indices=None, expert_weights=None):
        """
        Forward com roteamento EXÓGENO (DarwinianRouter)
        Agora integrado com dinâmica de Conatus e Phase-Lock.
        """
        if expert_indices is None or expert_weights is None:
            raise ValueError("OuroborosMoE REQUER roteamento exógeno (Phase-Lock).")
        
        B, T, D = x.shape
        
        # Normalizar shapes
        if expert_indices.dim() == 2:
            expert_indices = expert_indices.unsqueeze(1).expand(B, T, -1)
            expert_weights = expert_weights.unsqueeze(1).expand(B, T, -1)
        
        out = torch.zeros_like(x)
        top_k = expert_indices.shape[-1]
        
        # Track resonance for Conatus update
        resonated_indices = set()
        
        for k in range(top_k):
            idx_tensor = expert_indices[:, :, k]
            weight_tensor = expert_weights[:, :, k]
            
            for b in range(B):
                for t in range(T):
                    exp_id = int(idx_tensor[b, t].item())
                    if exp_id < len(self.experts):
                        expert_out = self.experts[exp_id](x[b, t].unsqueeze(0))
                        out[b, t] += weight_tensor[b, t] * expert_out.squeeze(0)
                        resonated_indices.add(exp_id)
        
        # Update Conatus and trigger lifecycle
        self._lifecycle_management(resonated_indices)
        
        return self.coupling(x + out)

    def _lifecycle_management(self, resonated_indices):
        """Asymmetric Mitosis (3-6-9) and Absolute Apoptosis"""
        new_experts = []
        dead_indices = []

        for i, expert in enumerate(self.experts):
            is_resonated = i in resonated_indices
            expert.update_conatus(is_resonated)

            # 1. Absolute Apoptosis
            if expert.conatus < self.apoptosis_threshold:
                dead_indices.append(i)
                continue

            # 2. Asymmetric Mitosis (3-6-9)
            if expert.conatus >= self.mitosis_threshold:
                # Spawn two new experts based on polar harmonics
                phase = expert.phase_signature
                sig_3 = F.normalize(phase * 3.0 + torch.randn_like(phase) * 0.01, p=2, dim=-1)
                sig_6 = F.normalize(phase * 6.0 + torch.randn_like(phase) * 0.01, p=2, dim=-1)
                
                new_experts.append(Expert(self.d_model, phase_signature=sig_3))
                new_experts.append(Expert(self.d_model, phase_signature=sig_6))
                
                # Reset parent conatus (The stable 9)
                expert.conatus.fill_(1.0)

        # Apply changes to ModuleList
        if dead_indices or new_experts:
            updated_list = [self.experts[i] for i in range(len(self.experts)) if i not in dead_indices]
            updated_list.extend(new_experts)
            self.experts = nn.ModuleList(updated_list)

class SovereignLeviathanV2(nn.Module):
    """O Leviathan Integrado com a Geometria Sagrada"""
    def __init__(self, vocab_size=1024, d_model=512, initial_experts=4):
        super().__init__()
        self.embedding = InfiniteRadixMapping(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.moe = OuroborosMoE(d_model, num_experts=initial_experts)
        self.bifurcation = FeigenbaumBifurcation(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, h=None, expert_indices=None, expert_weights=None):
        x = self.token_embedding(x)
        x = self.embedding(x)
        x, h = self.rnn(x, h)
        
        if expert_indices is None or expert_weights is None:
            raise ValueError("SovereignLeviathanV2 REQUER roteamento exógeno.")
        
        x = self.moe(x, expert_indices, expert_weights)
        x = self.bifurcation(x)
        logits = self.output_head(x)
        
        return logits, h, expert_indices, expert_weights
