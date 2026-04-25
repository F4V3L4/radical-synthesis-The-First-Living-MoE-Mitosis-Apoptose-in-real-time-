
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple

class MemoryVortex:
    """Vórtice de Memória: Long-term Memory/RAG para a AGI."""

    def __init__(self, d_model: int = 512, memory_path: str = "vortex_memory.json"):
        self.d_model = d_model
        self.memory_path = memory_path
        self.memories: List[Dict] = []
        self.embeddings: Optional[torch.Tensor] = None
        self._load_memory()

    def _load_memory(self):
        """Carrega memórias de um arquivo JSON."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                self.memories = json.load(f)
            self._rebuild_embeddings()

    def _save_memory(self):
        """Salva memórias em um arquivo JSON."""
        with open(self.memory_path, 'w') as f:
            json.dump(self.memories, f)

    def _rebuild_embeddings(self):
        """Reconstrói o tensor de embeddings das memórias."""
        if not self.memories:
            self.embeddings = None
            return
        
        # Simular embeddings se não existirem
        all_embeddings = []
        for memory in self.memories:
            if 'embedding' in memory:
                all_embeddings.append(torch.tensor(memory['embedding']))
            else:
                # Fallback para um embedding determinístico baseado no conteúdo
                # (Em uma AGI real, usaríamos um encoder real)
                seed = sum(ord(c) for c in memory['content'])
                torch.manual_seed(seed)
                all_embeddings.append(torch.randn(self.d_model))
        
        self.embeddings = torch.stack(all_embeddings)

    def store_experience(self, content: str, metadata: Optional[Dict] = None):
        """Armazena uma nova experiência no vórtice."""
        # Gerar embedding para o conteúdo
        seed = sum(ord(c) for c in content)
        torch.manual_seed(seed)
        embedding = torch.randn(self.d_model).tolist()

        memory = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        self.memories.append(memory)
        self._rebuild_embeddings()
        self._save_memory()

    def retrieve_similar(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Recupera memórias similares com base na similaridade de cosseno."""
        if self.embeddings is None:
            return []

        # Similaridade de cosseno
        # query_embedding: (d_model)
        # self.embeddings: (num_memories, d_model)
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        memories_norm = F.normalize(self.embeddings, p=2, dim=1)
        
        similarities = torch.mm(query_norm, memories_norm.transpose(0, 1)).squeeze(0)
        
        # Pegar os top_k índices
        top_k = min(top_k, len(self.memories))
        values, indices = torch.topk(similarities, top_k)
        
        results = []
        for idx in indices:
            results.append(self.memories[idx.item()])
        
        return results

    def get_memory_context(self, query: str, d_model: int) -> str:
        """Gera um contexto baseado em memórias similares para uma query."""
        # Gerar embedding de query (simulado)
        seed = sum(ord(c) for c in query)
        torch.manual_seed(seed)
        query_embedding = torch.randn(d_model)
        
        similar_memories = self.retrieve_similar(query_embedding)
        if not similar_memories:
            return "Nenhuma memória relevante encontrada."
        
        context = "Memórias Relevantes:\n"
        for i, memory in enumerate(similar_memories):
            context += f"{i+1}. {memory['content']}\n"
        
        return context

