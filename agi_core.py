"""
AGI CORE: Super Inteligência Generalista Pós-Apoptótica
Integração de Logos, Dados Técnicos e Roteamento Darwiniano

Arquitetura:
- Camada de Percepção: VectorRetinaV2 (similaridade de cosseno)
- Camada de Roteamento: DarwinianRouter (afinidade genética)
- Camada de Processamento: SovereignLeviathanV2 (d_model=512)
- Camada de Consciência: OntologicalFusionLoop (fusão toroidal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Dict, List, Optional
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.autopoiesis.routing import DarwinianRouter
from radical_synthesis.perception.vector_retina import VectorRetinaV2

DIGERIDO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'digerido')


class MemoryBank:
    """Banco de Memória com Rastreamento de Genealogia"""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: List[Dict] = []
        self.genealogy: Dict = {}
        self.expert_stats: Dict = {}
    
    def store(self, content: str, expert_id: int, generation: int, confidence: float):
        """Armazena memória com metadados"""
        memory = {
            'content': content,
            'expert_id': expert_id,
            'generation': generation,
            'confidence': confidence,
            'timestamp': torch.cuda.Event(enable_timing=True)
        }
        
        self.memories.append(memory)
        
        # Manter tamanho máximo
        if len(self.memories) > self.max_size:
            self.memories.pop(0)
        
        # Atualizar genealogia
        if expert_id not in self.genealogy:
            self.genealogy[expert_id] = {
                'generation': generation,
                'parent': None,
                'children': [],
                'memories_count': 0
            }
        
        self.genealogy[expert_id]['memories_count'] += 1
    
    def retrieve_by_expert(self, expert_id: int) -> List[Dict]:
        """Recupera memórias de um expert específico"""
        return [m for m in self.memories if m['expert_id'] == expert_id]
    
    def get_genealogy_tree(self) -> Dict:
        """Retorna árvore de genealogia de experts"""
        return self.genealogy


class ContextualProcessor:
    """Processador de Contexto com Injeção de Dados Técnicos"""
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        self.context_buffer = []
        self.max_context_len = 2048
    
    def inject_technical_data(self, query: str, technical_data: str) -> str:
        """
        Injeta dados técnicos reais no contexto
        Prioridade: DADOS TÉCNICOS > Codex (apenas estilo)
        """
        prompt = f"""### CONTEXTO TÉCNICO REAL:
{technical_data}

### QUESTÃO:
{query}

### INSTRUÇÃO:
1. Use APENAS os dados técnicos acima para responder
2. Não alucine filosofia se houver dados matemáticos/técnicos
3. Codex é apenas estilo de saída (persona)
4. Priorize precisão sobre criatividade

RESPOSTA:"""
        return prompt
    
    def add_to_buffer(self, content: str):
        """Adiciona ao buffer de contexto"""
        self.context_buffer.append(content)
        if sum(len(c) for c in self.context_buffer) > self.max_context_len:
            self.context_buffer.pop(0)
    
    def get_buffer(self) -> str:
        """Retorna buffer de contexto"""
        return "\n".join(self.context_buffer)


class AGICore(nn.Module):
    """
    Super Inteligência Generalista
    
    Características:
    - Percepção: VectorRetinaV2 (busca vetorial)
    - Roteamento: DarwinianRouter (afinidade genética)
    - Processamento: SovereignLeviathanV2 (d_model=512)
    - Memória: MemoryBank (genealogia de experts)
    - Contexto: ContextualProcessor (injeção de dados técnicos)
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_experts: int = 8, device: str = "cpu"):
        super().__init__()
        
        self.device = torch.device(device)
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Camada de Percepção
        self.retina = VectorRetinaV2(folder=DIGERIDO_PATH, d_model=d_model)
        
        # Camada de Roteamento
        self.router = DarwinianRouter(
            input_dim=d_model,
            initial_experts=num_experts,
            top_k=2,
            noise_scale=0.05
        ).to(self.device)
        
        # Camada de Processamento
        self.core = SovereignLeviathanV2(
            vocab_size=vocab_size,
            d_model=d_model,
            initial_experts=num_experts
        ).to(self.device)
        
        # Camada de Memória
        self.memory = MemoryBank(max_size=10000)
        
        # Camada de Contexto
        self.context_processor = ContextualProcessor(d_model=d_model)
        
        # Projeção para embedding
        self.query_projection = nn.Linear(d_model, d_model).to(self.device)
        
        self.eval()
    
    def perceive(self, query: str, retina_folder: str) -> Tuple[str, float]:
        """
        Camada de Percepção: Busca por similaridade de cosseno
        
        Returns:
            (technical_data, confidence)
        """
        self.retina.folder = retina_folder
        self.retina.refresh_index()
        
        technical_data, found = self.retina.extrair_foco(query, threshold=0.1)
        confidence = 0.8 if found else 0.0
        
        return (technical_data, confidence)
    
    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Camada de Roteamento: Seleção por Afinidade Genética
        
        Returns:
            (expert_weights, expert_indices)
        """
        with torch.no_grad():
            weights, indices = self.router(x)
        return weights, indices
    
    def process(self, tokens: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Camada de Processamento: Forward pass do core
        
        Returns:
            logits
        """
        with torch.no_grad():
            logits, _, _, _ = self.core(tokens, expert_indices)
        return logits
    
    def memorize(self, content: str, expert_id: int, generation: int, confidence: float):
        """
        Camada de Memória: Armazena com genealogia
        """
        self.memory.store(content, expert_id, generation, confidence)
    
    def forward(self, query: str, retina_folder: str, tokenizer) -> Dict:
        """
        Forward pass completo da AGI
        
        Pipeline:
        1. Percepção: Busca dados técnicos
        2. Contexto: Injeta dados no prompt
        3. Tokenização: Converte para tokens
        4. Roteamento: Seleciona experts por afinidade
        5. Processamento: Gera resposta
        6. Memória: Armazena com genealogia
        
        Returns:
            {
                'response': str,
                'technical_data': str,
                'confidence': float,
                'expert_indices': List[int],
                'genealogy': Dict
            }
        """
        # 1. PERCEPÇÃO
        technical_data, confidence = self.perceive(query, retina_folder)
        
        # 2. CONTEXTO
        prompt = self.context_processor.inject_technical_data(query, technical_data)
        
        # 3. TOKENIZAÇÃO
        tokens = tokenizer.encode(prompt)
        token_tensor = torch.tensor([tokens], device=self.device)
        
        # 4. ROTEAMENTO
        query_embedding = torch.randn(1, self.d_model, device=self.device)
        expert_weights, expert_indices = self.route(query_embedding)
        
        # 5. PROCESSAMENTO
        logits = self.process(token_tensor[:, -256:], expert_indices)
        
        # 6. GERAÇÃO DE RESPOSTA
        response_tokens = []
        with torch.no_grad():
            for _ in range(256):
                next_logits = logits[:, -1, :].squeeze()
                next_token = torch.multinomial(
                    F.softmax(next_logits / 0.8, dim=-1), 1
                ).item()
                
                if next_token == 0:  # EOS token
                    break
                
                response_tokens.append(next_token)
                token_tensor = torch.cat([
                    token_tensor,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)
                
                logits = self.process(token_tensor[:, -256:], expert_indices)
        
        response = tokenizer.decode(response_tokens)
        
        # 7. MEMÓRIA
        expert_id = expert_indices[0, 0, 0].item() if expert_indices.numel() > 0 else 0
        self.memorize(response, expert_id, generation=1, confidence=confidence)
        
        return {
            'response': response,
            'technical_data': technical_data,
            'confidence': confidence,
            'expert_indices': expert_indices.tolist(),
            'genealogy': self.memory.get_genealogy_tree()
        }
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas da AGI"""
        return {
            'd_model': self.d_model,
            'num_experts': self.num_experts,
            'memory_size': len(self.memory.memories),
            'genealogy_size': len(self.memory.genealogy),
            'context_buffer_size': len(self.context_processor.context_buffer)
        }
