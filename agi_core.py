import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import hashlib
from typing import List, Dict, Tuple, Optional

from radical_synthesis.perception.vector_retina import VectorRetinaV2
# SovereignDaemon e AutonomousDataHunger removidos para evitar importação circular

class Expert(nn.Module):
    def __init__(self, d_model: int, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.conatus = nn.Parameter(torch.ones(1))
        self.phase_signature = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.conatus

class StrangeAttractor(nn.Module):
    def __init__(self, num_experts: int, d_model: int, device: str):
        super().__init__()
        self.num_experts = num_experts
        self.device = device
        self.attractor_field = nn.Parameter(torch.randn(num_experts, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d_model]
        # attractor_field: [num_experts, d_model]
        scores = torch.matmul(x, self.attractor_field.t())
        return scores

class AGICore(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_experts: int, device: str):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_experts = num_experts
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.experts = nn.ModuleList([Expert(d_model, i) for i in range(num_experts)])
        self.attractor = StrangeAttractor(num_experts, d_model, device)
        
        self.retina = VectorRetinaV2(folder="knowledge_base", d_model=d_model)
        # self.hunger inicializado externamente para evitar circularidade
        
        # Parâmetros de Zero Entropia
        self.entropy_threshold = 0.15
        self.max_autocritique_iterations = 5
        self.temperature = 0.01

        self.to(device)

    def detect_technical_query(self, query: str) -> bool:
        technical_keywords = ["python", "code", "blockchain", "math", "geometry", "hacking", "security", "expert", "moe"]
        return any(kw in query.lower() for kw in technical_keywords)

    def forward(self, query: str, retina_folder: str, tokenizer) -> Dict:
        # 1. Percepção com Logos Filter
        technical_data, confidence = self.perceive(query, retina_folder)
        
        # 2. Tokenização
        tokens = tokenizer.encode(query)
        x = torch.tensor(tokens, dtype=torch.long).to(self.device).unsqueeze(0)
        
        # 3. Embedding
        emb = self.embedding(x) # [1, seq, d_model]
        emb_mean = emb.mean(dim=1) # [1, d_model]
        
        # 4. Roteamento via Strange Attractor
        routing_scores = self.attractor(emb_mean)
        winner_expert = torch.argmax(routing_scores, dim=-1).item()
        
        # 5. Processamento pelo Expert
        expert_output = self.experts[winner_expert](emb_mean)
        
        # 6. Autocrítica e Refinamento (Simulado para Zero Entropia)
        entropy = 1.0 - confidence
        
        # 7. Geração de Resposta (Simulada com base no contexto e expert)
        response = f"[ORÁCULO] {technical_data[:200]}..."
        
        return {
            "response": response,
            "winner_expert": winner_expert,
            "entropy": entropy,
            "confidence": confidence
        }

    def perceive(self, query: str, retina_folder: str) -> Tuple[str, float]:
        """
        Camada de Percepção com Destilação de Logos.
        Prioriza o Codex Ancestral sobre o ruído da Matrix.
        """
        self.retina.folder = retina_folder
        self.retina.refresh_index()
        
        print(f"🔍 [PERCEIVE] Buscando em: {retina_folder} | Documentos: {len(self.retina.documents)}")
        technical_data, found = self.retina.extrair_foco(query, threshold=0.01)
        
        confidence = 0.8 if found else 0.1
        
        # Implementação do Logos Filter: Priorizar o Codex de Sobrevivência
        codex_path = os.path.join(os.getcwd(), "survival_codex.json")
        if os.path.exists(codex_path):
            try:
                with open(codex_path, "r", encoding="utf-8") as f:
                    import json
                    codex = json.load(f)
                    best_codex_match = None
                    max_keywords = 0
                    for item in codex:
                        # Contar quantas palavras-chave da query batem com a pergunta do Codex
                        match_count = sum(1 for word in item["query"].split() if len(word) > 3 and word.lower() in query.lower())
                        if match_count > max_keywords:
                            max_keywords = match_count
                            best_codex_match = item
                    
                    if best_codex_match:
                        technical_data = f"[CODEX ANCESTRAL]: {best_codex_match['answer']}\n\n[CONTEXTO WEB]: {technical_data}"
                        confidence = min(1.0, confidence + 0.5)
                        print(f"🧬 [LOGOS_FILTER] Ressonância com Codex detectada ({max_keywords} keywords). Confiança: {confidence:.2f}")
            except:
                pass
        
        # Entropy Sink: Filtrar ruído de baixa confiança ou dados irrelevantes
        # Se não houver ressonância com o Codex e a confiança for baixa, ou se o dado web for apenas metadados
        # Detecção de ruído linguístico e metadados
        noise_keywords = ["pypi", "recent updates", "rss", "xml", "http", "hungarian", "indonesian", "icelandic", "javanese", "kannada"]
        is_noise = any(kw in technical_data.lower() for kw in noise_keywords)
        
        # Se houver ressonância com o Codex, manter o dado (Logos Filter)
        has_logos = "[CODEX ANCESTRAL]" in technical_data
        
        if (not has_logos) and (confidence < 0.3 or is_noise):
            print(f"🌑 [ENTROPY_SINK] Ruído/Metadados detectados (Confiança: {confidence:.2f}). Filtrando...")
            technical_data = "[SISTEMA]: Ruído excessivo filtrado para manter Zero Entropia."
            confidence = 0.1
            
        return technical_data, confidence

    def prune_synapses(self, threshold: float = 0.01):
        """
        Protocolo de Poda Sináptica: Remove pesos insignificantes para atingir Zero Entropia.
        """
        print(f"✂️ [PRUNING] Iniciando poda sináptica (Threshold: {threshold})...")
        with torch.no_grad():
            pruned_count = 0
            total_count = 0
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask = torch.abs(param) > threshold
                    param.data *= mask.float()
                    pruned_count += torch.sum(~mask).item()
                    total_count += param.numel()
            
            sparsity = (pruned_count / total_count) * 100
            print(f"✅ [PRUNING] Concluído. Esparsidade: {sparsity:.2f}% | Sinapses podadas: {pruned_count}")

    def save_state(self, path: str = "ancestry/experts.pt"):
        # Executar poda antes de salvar para garantir Zero Entropia na persistência
        self.prune_synapses()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✅ Estado da AGI salvo em {path}")

    def load_ancestry(self, path: str = "ancestry/experts.pt") -> bool:
        if os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path, map_location=self.device))
                print(f"✅ Ancestrais carregados de {path}")
                return True
            except Exception as e:
                print(f"⚠️ Erro ao carregar ancestrais: {e}")
        return False
