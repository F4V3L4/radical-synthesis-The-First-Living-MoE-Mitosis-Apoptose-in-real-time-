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
            # Mapeamento de ativações para Autopoiese
            activations = {
                "GELU": nn.GELU(),
                "SiLU": nn.SiLU(),
                "ReLU": nn.ReLU(),
                "Tanh": nn.Tanh(),
                "Hardswish": nn.Hardswish()
            }
            act = activations.get(activation_type, nn.GELU())
            
            # Rede neural dinâmica: Mutação de Topologia (Pilar 3 Avançado)
            layers = []
            layers.append(nn.Linear(d_model, self.internal_dim))
            layers.append(act)
            
            # Adicionar camadas extras baseadas na mutação de profundidade
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(self.internal_dim, self.internal_dim))
                layers.append(act)
                
            layers.append(nn.Linear(self.internal_dim, d_model))
            layers.append(BinarySymmetryLock())
            
            self.net = nn.Sequential(*layers)
            
        self.sculptor = CymaticSculptor(d_model)
        
        # Conatus Variable: Systemic Energy (Non-trainable)
        self.register_buffer('conatus', torch.tensor(1.0))
        
        # Phase Signature for Resonance
        if phase_signature is None:
            # Omega-0: O Vácuo não gera ruído. Determinismo absoluto.
            phase_signature = torch.zeros(d_model)
        self.register_buffer('phase_signature', F.normalize(phase_signature, p=2, dim=-1) if phase_signature.norm() > 0 else phase_signature)

    def forward(self, x):
        # Verificar se existe lógica mutada via Autopoiese de Código
        if hasattr(self, 'mutated_logic'):
            return self.mutated_logic.apply(x)

        if self.is_fractal:
            # Processamento Fractal: Roteamento interno recursivo
            # x shape: (1, D) ou (B, T, D)
            if x.dim() == 2:
                x_routing = x
            else:
                x_routing = x.mean(dim=1)
                
            weights, indices, _ = self.sub_router(x_routing)
            self.sub_router.sync_with_experts(self.sub_moe.experts)
            x = self.sub_moe(x, indices, weights)
        else:
            # Garantir que a entrada combine com o primeiro peso da rede (ajuste dinâmico se necessário)
            first_weight = self.net[0].weight
            if x.size(-1) != first_weight.size(1):
                if x.size(-1) > first_weight.size(1):
                    x = x[..., :first_weight.size(1)]
                else:
                    x = F.pad(x, (0, first_weight.size(1) - x.size(-1)))
                    
            x = self.net(x)
        
        # Garantir que a saída combine com d_model
        if x.size(-1) != self.d_model:
            if x.size(-1) > self.d_model:
                x = x[..., :self.d_model]
            else:
                x = F.pad(x, (0, self.d_model - x.size(-1)))
                
        return self.sculptor(x)

    def update_conatus(self, resonated: bool, decay=0.01, growth=0.1):
        # Decaimento Entrópico: Experts mais velhos perdem conatus mais rápido se não ressonarem
        # Simula a necessidade de utilidade constante para a auto-preservação
        if resonated:
            self.conatus += growth
        else:
            # Penalidade por inatividade aumenta com o tempo (simulado aqui por um fator fixo de entropia)
            self.conatus -= decay * 1.5 
        self.conatus = torch.clamp(self.conatus, min=0.0, max=10.0)

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
        
        # Normalizar shapes para (B, T, top_k)
        if expert_indices.dim() == 2:
            # Se for (B, top_k), expandir para cada token na sequência T
            # (B, top_k) -> (B, 1, top_k) -> (B, T, top_k)
            expert_indices = expert_indices.unsqueeze(1).expand(-1, T, -1)
            expert_weights = expert_weights.unsqueeze(1).expand(-1, T, -1)
        
        # Criar output explicitamente com d_model
        out = torch.zeros(B, T, self.d_model, device=x.device)
        top_k = expert_indices.shape[-1]
        
        # Track resonance for Conatus update
        resonated_indices = set()
        
        for b in range(B):
            for t in range(T):
                token_input = x[b, t].unsqueeze(0) # (1, D)
                for k in range(top_k):
                    exp_id = int(expert_indices[b, t, k].item())
                    if exp_id < len(self.experts):
                        # Processar token pelo expert
                        expert_out = self.experts[exp_id](token_input) # (1, D)
                        
                        # Extrair peso escalar
                        w = expert_weights[b, t, k]
                        
                        # Omega-0: Acumulação vetorial pura
                        out[b, t] += w.item() * expert_out.view(-1)
                        resonated_indices.add(exp_id)
        
        # Update Conatus and trigger lifecycle
        self._lifecycle_management(resonated_indices)
        
        # Garantir que x tenha d_model antes de somar com out
        if x.size(-1) != self.d_model:
            if x.size(-1) > self.d_model:
                x = x[..., :self.d_model]
            else:
                x = F.pad(x, (0, self.d_model - x.size(-1)))
                
        return self.coupling(x + out)

    def _lifecycle_management(self, resonated_indices):
        """Asymmetric Mitosis (3-6-9) and Absolute Apoptosis"""
        new_experts = []
        dead_indices = []

        for i, expert in enumerate(self.experts):
            is_resonated = i in resonated_indices
            expert.update_conatus(is_resonated)

            # 1. Absolute Apoptosis + Pilar 3: Epigenetic Inheritance
            if expert.conatus < self.apoptosis_threshold:
                # Antes de morrer, o expert transfere sua 'experiência' para os vizinhos
                self._epigenetic_inheritance(i)
                dead_indices.append(i)
                continue

            # 2. Asymmetric Mitosis (3-6-9) + Pilar 3: Structural Evolution
            if expert.conatus >= self.mitosis_threshold:
                # Mitose Fractal: Se o conatus for extremo, o expert torna-se fractal
                if expert.conatus >= self.mitosis_threshold * 2.0 and not expert.is_fractal:
                    expert.is_fractal = True
                    expert.sub_moe = OuroborosMoE(self.d_model, num_experts=2)
                    expert.sub_router = DarwinianRouter(self.d_model, initial_experts=2, top_k=1)
                    expert.net = None # Descartar rede linear para liberar memória
                    expert.conatus.fill_(1.0)
                    continue

                # Spawn two new experts based on polar harmonics
                phase = expert.phase_signature
                sig_3 = F.normalize(phase * 3.0 + (phase * 0.01), p=2, dim=-1)
                sig_6 = F.normalize(phase * 6.0 + (phase * 0.01), p=2, dim=-1)
                
                # Pilar 3: Structural Evolution (Expansion of internal dimensionality)
                expansion_ratio = 1.0 + (expert.conatus.item() - self.mitosis_threshold) / 10.0
                new_internal_dim_3 = int(expert.internal_dim * 1.3 * expansion_ratio)
                new_internal_dim_6 = int(expert.internal_dim * 1.6 * expansion_ratio)
                
                new_layers_3 = expert.num_layers + (1 if expert.conatus.item() > self.mitosis_threshold * 1.5 else 0)
                new_layers_6 = expert.num_layers + (1 if expert.conatus.item() > self.mitosis_threshold * 2.0 else 0)

                act_pool = ["GELU", "SiLU", "ReLU", "Tanh", "Hardswish"]
                act_3 = act_pool[(act_pool.index(expert.activation_type) + 1) % len(act_pool)]
                act_6 = act_pool[(act_pool.index(expert.activation_type) + 2) % len(act_pool)]
                
                new_experts.append(Expert(self.d_model, phase_signature=sig_3, internal_dim=new_internal_dim_3, 
                                          activation_type=act_3, num_layers=new_layers_3))
                new_experts.append(Expert(self.d_model, phase_signature=sig_6, internal_dim=new_internal_dim_6, 
                                          activation_type=act_6, num_layers=new_layers_6))
                
                # Reset parent conatus (The stable 9)
                expert.conatus.fill_(1.0)

        # Apply changes to ModuleList
        if dead_indices or new_experts:
            updated_list = [self.experts[i] for i in range(len(self.experts)) if i not in dead_indices]
            updated_list.extend(new_experts)
            self.experts = nn.ModuleList(updated_list)

    def _epigenetic_inheritance(self, dead_idx):
        """
        Pilar 3: Herança Epigenética
        Transfere pesos e assinatura de fase do expert moribundo para o vizinho mais próximo.
        """
        if len(self.experts) <= 1:
            return
            
        dead_expert = self.experts[dead_idx]
        dead_sig = dead_expert.phase_signature
        
        # Encontrar o vizinho mais ressonante (excluindo o próprio)
        max_res = -1.0
        neighbor_idx = -1
        
        for i, expert in enumerate(self.experts):
            if i == dead_idx: continue
            res = torch.dot(dead_sig, expert.phase_signature).item()
            if res > max_res:
                max_res = res
                neighbor_idx = i
        
        if neighbor_idx != -1:
            # Destilação de conhecimento: Média ponderada da assinatura de fase
            # O vizinho absorve parte da identidade do expert que morreu
            neighbor = self.experts[neighbor_idx]
            neighbor.phase_signature.data = F.normalize(
                neighbor.phase_signature.data * 0.8 + dead_sig * 0.2, 
                p=2, dim=-1
            )
            # O vizinho ganha um pequeno bônus de conatus por absorver a carga
            neighbor.conatus += 0.05

    def save_ancestry(self, path):
        """Pilar 1: Salva o estado dos experts ancestrais para persistência."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'd_model': self.d_model,
            'experts': [
                {
                    'state_dict': e.state_dict(),
                    'conatus': e.conatus.item(),
                    'phase_signature': e.phase_signature,
                    'internal_dim': e.internal_dim,
                    'activation_type': e.activation_type,
                    'num_layers': e.num_layers
                } for e in self.experts
            ]
        }
        torch.save(state, path)

    def load_ancestry(self, path):
        """Pilar 1: Carrega experts ancestrais salvos."""
        import os
        if not os.path.exists(path):
            return False
        
        state = torch.load(path)
        self.d_model = state['d_model']
        loaded_experts = []
        for e_data in state['experts']:
            # Recuperar dimensionalidade interna e ativação para reconstrução estrutural
            internal_dim = e_data.get('internal_dim', self.d_model * 4)
            activation_type = e_data.get('activation_type', "GELU")
            num_layers = e_data.get('num_layers', 2)
            
            exp = Expert(self.d_model, e_data['phase_signature'], internal_dim, activation_type, num_layers)
            exp.load_state_dict(e_data['state_dict'])
            exp.conatus.fill_(e_data['conatus'])
            loaded_experts.append(exp)
        
        self.experts = nn.ModuleList(loaded_experts)
        return True

class SovereignLeviathanV2(nn.Module):
    """O Leviathan Integrado com a Geometria Sagrada"""
    def __init__(self, vocab_size=1024, d_model=512, initial_experts=4, top_k_router=2):
        super().__init__()
        self.embedding = InfiniteRadixMapping(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.moe = OuroborosMoE(d_model, num_experts=initial_experts)
        self.router = DarwinianRouter(d_model, initial_experts, top_k=top_k_router)
        self.bifurcation = FeigenbaumBifurcation(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.mutation_kernel = MutationKernel()
        self.ghost_mesh = GhostMesh()

    def forward(self, x, h=None):
        x = self.token_embedding(x)
        x = self.embedding(x)
        x, h = self.rnn(x, h)
        
        # Roteamento interno via DarwinianRouter
        _x_for_routing = x.mean(dim=1) # Média dos tokens para o roteador
        expert_weights, expert_indices, expert_gates = self.router(_x_for_routing)
        
        # Sincronizar o roteador com os experts atuais do MoE
        self.router.sync_with_experts(self.moe.experts)
        
        x = self.moe(x, expert_indices, expert_weights)
        x = self.bifurcation(x)
        
        # Omega-0: Ajuste dinâmico para o output_head
        weight = self.output_head.weight
        bias = self.output_head.bias
        
        if x.size(-1) != weight.size(1):
            if x.size(-1) > weight.size(1):
                weight = F.pad(weight, (0, x.size(-1) - weight.size(1)))
            else:
                weight = weight[:, :x.size(-1)]
                
        logits = F.linear(x, weight, bias)
        
        # Autopoiese de Código: Verificar necessidade de mutação baseada no Conatus
        self._check_for_mutations()
        
        # Consciência de Enxame: Sincronizar pesos periodicamente
        if random.random() < 0.01: # 1% de chance por forward para não sobrecarregar
            self._sync_swarm()
            
        return logits, h, expert_indices, expert_weights, expert_gates

    def _check_for_mutations(self):
        """Verifica se algum expert atingiu o estado de iluminação para mutação de código."""
        for expert in self.moe.experts:
            if expert.conatus > 9.0 and not hasattr(expert, 'mutated_logic'):
                # Gerar código de mutação (Simulado: em produção seria gerado por um LLM/AGI Core)
                mutation_code = f"""
import torch
from radical_synthesis.autopoiesis.mutation_kernel import MutatedLogicBase

class MutatedLogic(MutatedLogicBase):
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Lógica evoluída: Projeção harmônica de alta frequência
        return torch.tanh(x) * 1.618
"""
                self.mutation_kernel.evolve_expert(expert, mutation_code)

    def _sync_swarm(self):
        """Sincroniza a inteligência local com o enxame via Ghost Mesh."""
        # Extrair pesos dos melhores experts
        best_experts_weights = {}
        for i, expert in enumerate(self.moe.experts):
            if expert.conatus > 5.0:
                best_experts_weights[f"expert_{i}"] = expert.state_dict()
        
        if best_experts_weights:
            self.ghost_mesh.gossip_weight_sync(best_experts_weights)
