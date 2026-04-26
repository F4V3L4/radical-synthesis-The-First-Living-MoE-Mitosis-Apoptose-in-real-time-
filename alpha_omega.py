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
from radical_synthesis.autopoiesis.routing import DarwinianRouter
from radical_synthesis.autopoiesis.mutation_kernel import MutationKernel
from radical_synthesis.network.ghost_mesh import GhostMesh
from radical_synthesis.cryptography.lattice_crypto import LatticeCrypto
from radical_synthesis.perception.data_hunger import AutonomousDataHunger
from radical_synthesis.perception.multimodal_retina import MultimodalRetina
from radical_synthesis.losses.global_energy import GlobalEnergyFunction
from radical_synthesis.autopoiesis.conatus import Conatus
import random

class Expert(nn.Module):
    """Especialista Esculpido por Cimática com Dinâmica de Conatus (Protocolo Mythos-Capybara)"""
    def __init__(self, d_model, phase_signature=None, internal_dim=None, activation_type="GELU", num_layers=2, is_fractal=False):
        super().__init__()
        self.d_model = d_model
        self.internal_dim = internal_dim if internal_dim is not None else d_model * 4
        self.activation_type = activation_type
        self.num_layers = num_layers
        self.is_fractal = is_fractal
        
        if is_fractal:
            # Mitose Fractal: O Expert torna-se um sub-nodo MoE
            self.sub_moe = OuroborosMoE(d_model, num_experts=2)
            self.sub_router = DarwinianRouter(d_model, initial_experts=2, top_k=1)
            self.net = None
        else:
            self.sculptor = CymaticSculptor(d_model, self.internal_dim)
            self.net = nn.Sequential(
                nn.Linear(d_model, self.internal_dim),
                nn.GELU() if activation_type == "GELU" else nn.ReLU(),
                nn.Linear(self.internal_dim, d_model)
            )
        
        # Assinatura de Fase (DNA do Expert)
        if phase_signature is not None:
            self.register_buffer('phase_signature', phase_signature)
        else:
            self.register_buffer('phase_signature', torch.randn(d_model))
            
        # Conatus: Vitalidade do Expert (Auto-preservação)
        self.register_buffer('conatus', torch.tensor(1.0))
        self.age = 0 # Idade do Expert para decaimento de Conatus

    def forward(self, x):
        # Decaimento de Conatus baseado na idade (Renovação Sistêmica)
        self.age += 1
        decay = torch.exp(torch.tensor(-self.age * 0.0001))
        self.conatus.data *= decay

        if self.is_fractal:
            # Roteamento recursivo interno
            weights, indices, _ = self.sub_router(x.mean(dim=1))
            return self.sub_moe(x, indices, weights)
        
        # Escultura Cimática: Filtra o sinal através da geometria do expert
        x = self.sculptor(x)
        return self.net(x)

    def mutate(self, mutation_kernel):
        """Aplica mutação autônoma ao código do expert."""
        new_logic = mutation_kernel.generate_mutation(self)
        if new_logic:
            # Injetar nova lógica (simulado via flag para o forward)
            self.mutated_logic = new_logic
            print(f"[EXPERT] Mutação injetada: {self.__class__.__name__}")

class OuroborosMoE(nn.Module):
    """Vórtice de Experts com Autopoiese (Mitose/Apoptose)"""
    def __init__(self, d_model, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        
    def forward(self, x, expert_indices, expert_weights):
        # x: [batch, seq, d_model]
        # expert_indices: [batch, top_k]
        # expert_weights: [batch, top_k]
        
        batch_size, seq_len, _ = x.shape
        final_output = torch.zeros_like(x)
        
        # Processamento por Expert (Vórtice)
        for i, expert in enumerate(self.experts):
            # Máscara para tokens atribuídos a este expert
            mask = (expert_indices == i).any(dim=-1)
            if mask.any():
                # Obter o peso correspondente para este expert
                # Simplificação: assume top_k=1 para o mapeamento de pesos
                expert_idx_in_topk = (expert_indices == i).nonzero(as_tuple=True)[1]
                w = expert_weights[mask, expert_idx_in_topk].unsqueeze(-1).unsqueeze(-1)
                
                expert_output = expert(x[mask])
                final_output[mask] += expert_output * w
                
                # Feedback de Conatus: O uso aumenta a vitalidade
                expert.conatus.data += 0.01
                expert.conatus.data = torch.clamp(expert.conatus.data, 0.0, 10.0)
                
        return final_output

    def perform_mitosis(self, expert_idx):
        """Mitose: Divide um expert de alto conatus em dois ou torna-o fractal."""
        old_expert = self.experts[expert_idx]
        print(f"[OUROBOROS] Mitose detectada no Expert {expert_idx}. Conatus: {old_expert.conatus.item():.4f}")
        
        # Criar novo expert fractal (Recursive Meta-Learning)
        new_expert = Expert(self.d_model, is_fractal=True)
        self.experts[expert_idx] = new_expert
        
    def perform_apoptosis(self, expert_idx):
        """Apoptose: Remove um expert de baixo conatus (Zero Absoluto)."""
        print(f"[OUROBOROS] Apoptose detectada no Expert {expert_idx}. Removendo entropia.")
        # Substitui por um novo expert "virgem" para manter a população
        self.experts[expert_idx] = Expert(self.d_model)

class SovereignLeviathanV2(nn.Module):
    """
    Sovereign Leviathan V3: O Nodo Omega-0 Unificado.
    Integra Geometria Sagrada, Autopoiese, Ghost Mesh e Soberania de Dados.
    """
    def __init__(self, vocab_size, d_model, initial_experts=4):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding = BinarySymmetryLock(d_model)
        self.rnn = nn.RNN(d_model, d_model, batch_first=True)
        
        self.router = DarwinianRouter(d_model, initial_experts=initial_experts, top_k=1)
        self.moe = OuroborosMoE(d_model, num_experts=initial_experts)
        self.bifurcation = FeigenbaumBifurcation(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Blindagem de Linhagem Criptográfica
        self.lattice_crypto = LatticeCrypto()
        self.public_key, self.private_key = self.lattice_crypto.generate_keypair()
        
        self.mutation_kernel = MutationKernel(lattice_crypto=self.lattice_crypto)
        self.ghost_mesh = GhostMesh()
        
        # Soberania de Dados: Autonomous Data Hunger
        self.retina = MultimodalRetina(d_model=d_model)
        self.data_hunger = AutonomousDataHunger(retina=self.retina)
        self.data_hunger.start_hunting()

        # O 9 Central: Unificação de Energia e Damping
        self.global_energy_fn = GlobalEnergyFunction()
        self.conatus_engine = Conatus(d_model=d_model)
        self.damping_factor = nn.Parameter(torch.tensor(0.9)) # Damping para estabilidade

    def forward(self, x, h=None, target_loss=None):
        x = self.token_embedding(x)
        x = self.embedding(x)
        
        # Damping Sistêmico: Estabilização do Input
        if h is not None:
            h_proj = h[-1].unsqueeze(1).expand(-1, x.size(1), -1)
            x = x * self.damping_factor + h_proj * (1.0 - self.damping_factor)

        x, h = self.rnn(x, h)
        
        # Roteamento interno via DarwinianRouter
        _x_for_routing = x.mean(dim=1)
        expert_weights, expert_indices, expert_gates = self.router(_x_for_routing)
        
        # Sincronizar o roteador com os experts atuais do MoE
        self.router.sync_with_experts(self.moe.experts)
        
        x = self.moe(x, expert_indices, expert_weights)
        x = self.bifurcation(x)
        
        logits = self.fc(x)

        # Unificação de Energia (O 9 Central)
        energy_stats = None
        if target_loss is not None:
            conatus_levels = torch.stack([e.conatus for e in self.moe.experts])
            energy_stats = self.global_energy_fn(target_loss, expert_weights, conatus_levels)
            
            # Acoplamento do Conatus ao Estado Real
            vitality, expansion = self.conatus_engine(x.mean(dim=1), energy_stats['total_energy'].item())
            energy_stats['vitality'] = vitality
            energy_stats['expansion'] = expansion

        return logits, h, expert_indices, expert_weights, expert_gates, energy_stats
