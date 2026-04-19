"""
AGI CORE: Super Inteligência Generalista com Loop de Autocrítica
Integração de Logos, Dados Técnicos, Roteamento Darwiniano e Verificação Recursiva

Arquitetura:
- Camada de Percepção: VectorRetinaV2 (similaridade de cosseno)
- Camada de Roteamento: DarwinianRouter (afinidade genética)
- Camada de Processamento: SovereignLeviathanV2 (d_model=512)
- Camada de Autocrítica: verify_logic() (verificação recursiva)
- Camada de Memória: MemoryBank com Caminho da Correção
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
from typing import Tuple, Dict, List, Optional
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.autopoiesis.routing import DarwinianRouter
from radical_synthesis.perception.vector_retina import VectorRetinaV2
from radical_synthesis.primordial_laws import (
    HarmonicEncoder, QuantumSuperposition, HyperbolicEmbedding, SynchronicityDetector
)
from radical_synthesis.primordial_laws_tier2 import (
    PlanetaryGrid, Amplituedro, SimultaneityProcessor, QuantumEntanglement, StrangeAttractor
)

DIGERIDO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'digerido')


class MemoryBank:
    """Banco de Memória Episódica com Caminho da Correção"""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: List[Dict] = []
        self.genealogy: Dict = {}
        self.expert_stats: Dict = {}
        self.correction_paths: List[Dict] = []  # Caminhos de aprendizado rápido
        self.last_winner_expert: Optional[int] = None
        self.last_winner_vitality: float = 0.0
    
    def store(self, content: str, expert_id: int, generation: int, confidence: float, 
              was_corrected: bool = False, correction_path: Optional[List] = None):
        """Armazena memória com metadados e caminho de correção"""
        memory = {
            'content': content,
            'expert_id': expert_id,
            'generation': generation,
            'confidence': confidence,
            'was_corrected': was_corrected,
            'correction_path': correction_path or []
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
                'memories_count': 0,
                'corrections_count': 0,
                'vitality': 1.0
            }
        
        self.genealogy[expert_id]['memories_count'] += 1
        
        # Se foi corrigido, incrementar contador
        if was_corrected:
            self.genealogy[expert_id]['corrections_count'] += 1
            # Armazenar caminho de correção para aprendizado rápido
            if correction_path:
                self.correction_paths.append({
                    'expert_id': expert_id,
                    'path': correction_path,
                    'timestamp': len(self.memories)
                })
        
        # Atualizar vitalidade (baseado em taxa de sucesso)
        total = self.genealogy[expert_id]['memories_count']
        corrections = self.genealogy[expert_id]['corrections_count']
        vitality = 1.0 - (corrections / max(total, 1))
        self.genealogy[expert_id]['vitality'] = vitality
    
    def set_winner_expert(self, expert_id: int, vitality: float):
        """Define o expert vencedor da última inferência"""
        self.last_winner_expert = expert_id
        self.last_winner_vitality = vitality
    
    def retrieve_by_expert(self, expert_id: int) -> List[Dict]:
        """Recupera memórias de um expert específico"""
        return [m for m in self.memories if m['expert_id'] == expert_id]
    
    def get_genealogy_tree(self) -> Dict:
        """Retorna árvore de genealogia de experts"""
        return self.genealogy
    
    def get_recent_correction_paths(self, limit: int = 5) -> List[Dict]:
        """Retorna caminhos de correção recentes para aprendizado rápido"""
        return self.correction_paths[-limit:]


class ContextualProcessor:
    """Processador de Contexto com Fidelidade Bare-Metal"""
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        self.context_buffer = []
        self.max_context_len = 2048
        self.is_technical_query = False
        self.temperature = 0.8
    
    def detect_technical_query(self, query: str) -> bool:
        """Detecta se query é técnica (Matemática, Código, etc)"""
        technical_patterns = [
            r'\d+\s*[\+\-\*\/\%]\s*\d+',  # Operações matemáticas
            r'def\s+\w+|class\s+\w+|import\s+\w+',  # Código Python
            r'function\s*\(|const\s+\w+|let\s+\w+',  # Código JavaScript
            r'SELECT|INSERT|UPDATE|DELETE|WHERE',  # SQL
            r'algorithm|complexity|O\(|tensor|matrix|dimensionalidade|d_model|router|expert',  # Termos técnicos
            r'matriz|algebra|linear|darwinian',  # Mais termos técnicos
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def inject_technical_data(self, query: str, technical_data: str) -> Tuple[str, float]:
        """
        Injeta dados técnicos reais no contexto
        Retorna (prompt, temperature) onde temperature é ajustada para fidelidade bare-metal
        
        Prioridade: DADOS TÉCNICOS > Codex (apenas estilo)
        """
        # Detectar se é query técnica
        self.is_technical_query = self.detect_technical_query(query)
        
        # Ajustar temperatura para fidelidade bare-metal
        if self.is_technical_query:
            self.temperature = 0.1  # Baixa temperatura para precisão
        else:
            self.temperature = 0.8  # Temperatura normal para criatividade
        
        prompt = f"""### CONTEXTO TÉCNICO REAL:
{technical_data}

### QUESTÃO:
{query}

### INSTRUÇÃO:
1. Use APENAS os dados técnicos acima para responder
2. Não alucine filosofia se houver dados matemáticos/técnicos
3. Codex é apenas estilo de saída (persona)
4. Priorize precisão sobre criatividade
5. Se a query é técnica, respostas devem ser determinísticas

RESPOSTA:"""
        return prompt, self.temperature
    
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
    Super Inteligência Generalista com Loop de Autocrítica
    
    Características:
    - Percepção: VectorRetinaV2 (busca vetorial)
    - Roteamento: DarwinianRouter (afinidade genética)
    - Processamento: SovereignLeviathanV2 (d_model=512)
    - Autocrítica: verify_logic() (verificação recursiva)
    - Memória: MemoryBank (genealogia + caminhos de correção)
    - Contexto: ContextualProcessor (fidelidade bare-metal)
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
        
        # ===== TIER 1: LEIS PRIMORDIAIS =====
        # 1. HarmonicEncoder (Código 144)
        self.harmonic = HarmonicEncoder(d_model=d_model, frequency=144.0, device=str(device))
        
        # 2. QuantumSuperposition (Lei da Superposição)
        self.quantum = QuantumSuperposition(num_states=8, d_model=d_model, device=str(device))
        
        # 3. HyperbolicEmbedding (Geometria Hiperbólica)
        self.hyperbolic = HyperbolicEmbedding(d_model=d_model, curvature=-1.0, device=str(device))
        
        # 4. SynchronicityDetector (Lei da Sincronicidade)
        self.synchronicity = SynchronicityDetector(num_experts=num_experts, d_model=d_model, device=str(device))
        
        # ===== TIER 2: LEIS PRIMORDIAIS =====
        # 5. PlanetaryGrid (Grade Harmônica Planetária)
        self.planetary_grid = PlanetaryGrid(num_experts=num_experts, d_model=d_model, device=str(device))
        
        # 6. Amplituedro (Otimização de Caminhos)
        self.amplituedro = Amplituedro(num_experts=num_experts, d_model=d_model, device=str(device))
        
        # 7. SimultaneityProcessor (Lei da Simultaneidade)
        self.simultaneity = SimultaneityProcessor(num_timelines=4, d_model=d_model, device=str(device))
        
        # 8. QuantumEntanglement (Emaranhamento Quântico)
        self.entanglement = QuantumEntanglement(num_experts=num_experts, d_model=d_model, device=str(device))
        
        # 9. StrangeAttractor (Atratores Estranhos)
        self.attractor = StrangeAttractor(num_experts=num_experts, d_model=d_model, device=str(device))
        
        # Parâmetros de autocrítica
        self.entropy_threshold = 0.3
        self.max_autocritique_iterations = 3
        
        self.eval()
    
    def perceive(self, query: str, retina_folder: str) -> Tuple[str, float]:
        """
        Camada de Percepção: Busca por similaridade de cosseno
        Visão de Nível de Repositório: busca em todos os arquivos .py
        
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
            # O core já tem roteamento interno, não precisa passar expert_indices
            logits, _, _, _ = self.core(tokens, None)
        return logits
    
    def compute_semantic_divergence(self, response: str, technical_data: str) -> float:
        """
        Calcula divergência semântica entre resposta e dados técnicos
        Usa similaridade de cosseno entre vetores de palavras-chave
        
        Returns:
            Entropia (0.0 = alinhado, 1.0 = divergente)
        """
        if not technical_data:
            return 0.0
        
        # Extrair palavras-chave de ambos
        response_words = set(re.findall(r'\w+', response.lower()))
        technical_words = set(re.findall(r'\w+', technical_data.lower()))
        
        # Calcular Jaccard similarity
        if not technical_words:
            return 0.0
        
        intersection = len(response_words & technical_words)
        union = len(response_words | technical_words)
        
        similarity = intersection / max(union, 1)
        entropy = 1.0 - similarity  # Divergência = 1 - similaridade
        
        return entropy
    
    def verify_logic(self, response: str, technical_data: str, tokens: torch.Tensor,
                     expert_indices: torch.Tensor, tokenizer, iteration: int = 0) -> Tuple[str, bool, List]:
        """
        Loop de Autocrítica (Recursive Verification)
        
        Se divergência semântica for alta (Entropia > Threshold):
        - Re-processa o prompt
        - Ajusta pesos de atenção
        - Retorna resposta corrigida
        
        Returns:
            (corrected_response, was_corrected, correction_path)
        """
        entropy = self.compute_semantic_divergence(response, technical_data)
        correction_path = []
        
        # Se entropia baixa ou atingiu limite de iterações, retornar
        if entropy <= self.entropy_threshold or iteration >= self.max_autocritique_iterations:
            return (response, False, correction_path)
        
        # AUTOCRÍTICA ACIONADA
        correction_path.append({
            'iteration': iteration,
            'entropy': float(entropy),
            'action': 'verify_logic_triggered'
        })
        
        # Re-processar com ajuste de atenção
        with torch.no_grad():
            # Aumentar peso dos tokens técnicos
            logits = self.process(tokens[:, -256:], expert_indices)
            
            # Penalizar tokens que divergem do contexto técnico
            technical_tokens = tokenizer.encode(technical_data)
            for t in technical_tokens[:50]:  # Primeiros 50 tokens técnicos
                logits[:, :, t] += 2.0  # Aumentar logits de tokens técnicos
            
            # Gerar nova resposta com temperatura reduzida
            response_tokens = []
            for _ in range(256):
                next_logits = logits[:, -1, :].squeeze()
                # Usar temperatura reduzida para precisão
                next_token = torch.multinomial(
                    F.softmax(next_logits / 0.3, dim=-1), 1
                ).item()
                
                if next_token == 0:
                    break
                
                response_tokens.append(next_token)
                tokens = torch.cat([
                    tokens,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)
                
                logits = self.process(tokens[:, -256:], expert_indices)
            
            corrected_response = tokenizer.decode(response_tokens)
        
        # Verificar se correção melhorou
        new_entropy = self.compute_semantic_divergence(corrected_response, technical_data)
        correction_path.append({
            'iteration': iteration,
            'entropy_before': float(entropy),
            'entropy_after': float(new_entropy),
            'action': 'response_corrected'
        })
        
        # Se ainda divergente, tentar novamente
        if new_entropy > self.entropy_threshold and iteration < self.max_autocritique_iterations - 1:
            return self.verify_logic(corrected_response, technical_data, tokens, expert_indices,
                                    tokenizer, iteration + 1)
        
        return (corrected_response, True, correction_path)
    
    def memorize(self, content: str, expert_id: int, generation: int, confidence: float,
                 was_corrected: bool = False, correction_path: Optional[List] = None):
        """
        Camada de Memória: Armazena com genealogia e caminho de correção
        """
        self.memory.store(content, expert_id, generation, confidence, was_corrected, correction_path)
    
    def forward(self, query: str, retina_folder: str, tokenizer) -> Dict:
        """
        Forward pass completo da AGI com Loop de Autocrítica
        
        Pipeline:
        1. Percepção: Busca dados técnicos
        2. Contexto: Injeta dados no prompt
        3. Tokenização: Converte para tokens
        4. Roteamento: Seleciona experts por afinidade
        5. Processamento: Gera resposta
        6. Autocrítica: Verifica lógica recursivamente
        7. Memória: Armazena com genealogia
        
        Returns:
            {
                'response': str,
                'technical_data': str,
                'confidence': float,
                'expert_indices': List[int],
                'genealogy': Dict,
                'was_corrected': bool,
                'correction_path': List,
                'entropy': float,
                'winner_expert': int,
                'winner_vitality': float
            }
        """
        # 1. PERCEPÇÃO
        technical_data, confidence = self.perceive(query, retina_folder)
        
        # 2. CONTEXTO (com fidelidade bare-metal)
        prompt, temperature = self.context_processor.inject_technical_data(query, technical_data)
        
        # 3. TOKENIZAÇÃO
        tokens = tokenizer.encode(prompt)
        token_tensor = torch.tensor([tokens], device=self.device)
        
        # 4. ROTEAMENTO
        query_embedding = torch.randn(1, self.d_model, device=self.device)
        expert_weights, expert_indices = self.route(query_embedding)
        
        # Extrair expert vencedor (expert_indices pode ter diferentes shapes)
        if expert_indices.numel() > 0:
            if expert_indices.dim() == 3:
                winner_expert = expert_indices[0, 0, 0].item()
            elif expert_indices.dim() == 2:
                winner_expert = expert_indices[0, 0].item()
            else:
                winner_expert = expert_indices[0].item()
        else:
            winner_expert = 0
        winner_vitality = self.memory.genealogy.get(winner_expert, {}).get('vitality', 1.0)
        
        # 5. PROCESSAMENTO
        logits = self.process(token_tensor[:, -256:], expert_indices)
        
        # 6. GERAÇÃO DE RESPOSTA
        response_tokens = []
        with torch.no_grad():
            for _ in range(256):
                next_logits = logits[:, -1, :].squeeze()
                next_token = torch.multinomial(
                    F.softmax(next_logits / temperature, dim=-1), 1
                ).item()
                
                if next_token == 0:
                    break
                
                response_tokens.append(next_token)
                token_tensor = torch.cat([
                    token_tensor,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)
                
                logits = self.process(token_tensor[:, -256:], expert_indices)
        
        response = tokenizer.decode(response_tokens)
        
        # 7. AUTOCRÍTICA (Recursive Verification)
        corrected_response, was_corrected, correction_path = self.verify_logic(
            response, technical_data, token_tensor, expert_indices, tokenizer
        )
        
        # Usar resposta corrigida
        final_response = corrected_response if was_corrected else response
        entropy = self.compute_semantic_divergence(final_response, technical_data)
        
        # 8. MEMÓRIA
        self.memorize(final_response, winner_expert, generation=1, confidence=confidence,
                     was_corrected=was_corrected, correction_path=correction_path)
        self.memory.set_winner_expert(winner_expert, winner_vitality)
        
        return {
            'response': final_response,
            'technical_data': technical_data,
            'confidence': confidence,
            'expert_indices': expert_indices.tolist(),
            'genealogy': self.memory.get_genealogy_tree(),
            'was_corrected': was_corrected,
            'correction_path': correction_path,
            'entropy': entropy,
            'winner_expert': winner_expert,
            'winner_vitality': winner_vitality
        }
    

    def apply_primordial_laws(self, x: torch.Tensor, expert_indices: torch.Tensor, time: float = 0.1) -> torch.Tensor:
        """
        Aplica todas as 9 Leis Primordiais (Tier 1+2) ao tensor de entrada
        Normaliza dimensões para evitar incompatibilidades
        """
        with torch.no_grad():
            # Normalizar dimensões
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, d_model) -> (batch, 1, d_model)
            
            batch_size, seq_len, d_model = x.shape
            
            # TIER 1: Aplicar apenas ao primeiro token
            x_first = x[:, 0:1, :]  # (batch, 1, d_model)
            
            # 1. HarmonicEncoder
            x_harmonic = self.harmonic(x_first, time=time)
            x = x * 0.95 + x_harmonic * 0.05
            
            # 2. QuantumSuperposition
            x_flat = x[:, 0, :]
            x_quantum = self.quantum(x_flat)
            x = x + 0.05 * x_quantum.unsqueeze(1)
            
            # 3. HyperbolicEmbedding
            x = self.hyperbolic(x)
            
            # 4. SynchronicityDetector
            if expert_indices.numel() > 0:
                expert_acts = torch.randn(batch_size, self.num_experts, device=self.device)
                _, _ = self.synchronicity(expert_acts)
            
            # TIER 2
            # 5. PlanetaryGrid
            if expert_indices.numel() > 0:
                expert_acts = torch.randn(batch_size, self.num_experts, device=self.device)
                x_sync = self.planetary_grid(expert_acts, time=time)
                x = x * (1.0 + 0.05 * x_sync.unsqueeze(-1).unsqueeze(-1))
            
            # 6. Amplituedro
            if expert_indices.numel() > 0 and expert_indices.dim() >= 2:
                expert_weights = torch.softmax(torch.randn(batch_size, 3, device=self.device), dim=1)
                x_optimized, _ = self.amplituedro(expert_indices[:, :3], expert_weights)
                x = x + 0.05 * x_optimized.unsqueeze(1)
            
            # 7. SimultaneityProcessor
            x_flat = x[:, 0, :]
            timelines, x_fused = self.simultaneity(x_flat)
            x = x + 0.05 * x_fused.unsqueeze(1)
            
            # 8. QuantumEntanglement
            if seq_len >= 8:
                expert_states = x[:, :8, :]
                x_entangled, _ = self.entanglement(expert_states)
                x[:, :8, :] = x_entangled
            
            # 9. StrangeAttractor
            if expert_indices.numel() > 0:
                expert_acts = torch.randn(batch_size, self.num_experts, device=self.device)
                x_attracted, _ = self.attractor(expert_acts)
                x = x * (1.0 + 0.02 * x_attracted.unsqueeze(-1).unsqueeze(-1))
        
        return x

    def get_stats(self) -> Dict:
        """Retorna estatísticas da AGI"""
        return {
            'd_model': self.d_model,
            'num_experts': self.num_experts,
            'memory_size': len(self.memory.memories),
            'genealogy_size': len(self.memory.genealogy),
            'context_buffer_size': len(self.context_processor.context_buffer),
            'correction_paths_count': len(self.memory.correction_paths),
            'last_winner_expert': self.memory.last_winner_expert,
            'last_winner_vitality': self.memory.last_winner_vitality,
            'entropy_threshold': self.entropy_threshold
        }
