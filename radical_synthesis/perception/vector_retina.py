"""
VectorRetinaV2: Percepção Vetorial com Similaridade de Cosseno
Zero Entropia - Bare-metal numpy, sem dependências externas
Conatus de Precisão: Prioriza dados técnicos reais sobre alucinações
"""

import numpy as np
import os
from typing import Tuple, List, Dict


class VectorRetinaV2:
    """
    Retina Vetorial com Similaridade de Cosseno
    
    Características:
    - Busca por similaridade de cosseno (bare-metal numpy)
    - Índice vetorial em memória
    - Suporte a chunking com overlap
    - Cache de embeddings
    - Zero Entropia: sem bibliotecas desnecessárias
    """
    
    def __init__(self, folder: str, d_model: int = 512):
        """
        Inicializa a Retina Vetorial
        
        Args:
            folder: Caminho para pasta com arquivos .txt
            d_model: Dimensionalidade dos vetores (padrão: 512)
        """
        self.folder = folder
        self.d_model = d_model
        self.documents: List[Dict] = []
        self.vectors: np.ndarray = np.array([])
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        self._build_vector_index()
    
    def _build_vector_index(self):
        """Constrói índice vetorial de todos os arquivos em folder/"""
        if not os.path.exists(self.folder):
            return
        
        for arq in os.listdir(self.folder):
            if not arq.endswith('.txt'):
                continue
            
            filepath = os.path.join(self.folder, arq)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                continue
            
            # Dividir em chunks de 600 caracteres com overlap de 200
            # Bare-metal Fix: Suportar arquivos menores que 600 caracteres
            if len(content) <= 600:
                chunks = [content]
            else:
                chunks = [content[i:i+600] for i in range(0, len(content) - 400, 200)]
            
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    'file': arq,
                    'chunk': chunk,
                    'offset': i * 200
                })
                
                # Gerar vetor
                vec = self._text_to_vector(chunk)
                if isinstance(self.vectors, list):
                    self.vectors.append(vec)
                else:
                    self.vectors = [vec] if self.vectors.size == 0 else list(self.vectors) + [vec]
        
        if len(self.vectors) > 0:
            self.vectors = np.array(self.vectors)
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Converte texto em vetor de contexto
        
        Estratégia: Hash de palavras para criar vetor esparso normalizado
        Bare-metal: apenas numpy, sem ML libraries
        
        Args:
            text: Texto para vetorizar
            
        Returns:
            Vetor normalizado L2 de dimensão d_model
        """
        words = text.lower().split()
        vec = np.zeros(self.d_model, dtype=np.float32)
        
        # Mapear palavras para índices via hash
        for word in words:
            idx = hash(word) % self.d_model
            vec[idx] += 1.0
        
        # Normalizar L2
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def extrair_foco(self, query: str, threshold: float = 0.1) -> Tuple[str, bool]:
        """
        Busca por similaridade de cosseno
        
        Conatus de Precisão: Retorna apenas se similaridade > threshold
        
        Args:
            query: Query de busca
            threshold: Limiar mínimo de similaridade (padrão: 0.1)
            
        Returns:
            Tupla (chunk_encontrado, sucesso)
        """
        if len(self.documents) == 0 or len(self.vectors) == 0:
            return "", False
        
        query_vec = self._text_to_vector(query)
        
        # Similaridade de cosseno: dot product entre vetores normalizados
        similarities = np.dot(self.vectors, query_vec)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Threshold: apenas retornar se similaridade > limiar
        if best_score >= threshold:
            best_chunk = self.documents[best_idx]['chunk']
            return (best_chunk, True)
        
        return ("", False)
    
    def buscar_multiplos(self, query: str, top_k: int = 3, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Busca os top-k resultados mais similares
        
        Args:
            query: Query de busca
            top_k: Número de resultados (padrão: 3)
            threshold: Limiar mínimo de similaridade
            
        Returns:
            Lista de (chunk, score) ordenada por score descendente
        """
        if len(self.documents) == 0 or len(self.vectors) == 0:
            return []
        
        query_vec = self._text_to_vector(query)
        similarities = np.dot(self.vectors, query_vec)
        
        # Top-k índices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                chunk = self.documents[idx]['chunk']
                results.append((chunk, float(score)))
        
        return results
    
    def refresh_index(self):
        """Reconstrói o índice vetorial (útil após adicionar novos arquivos)"""
        self.documents = []
        self.vectors = []
        self.embeddings_cache = {}
        self._build_vector_index()
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do índice"""
        return {
            'total_documents': len(self.documents),
            'total_vectors': len(self.vectors),
            'd_model': self.d_model,
            'cache_size': len(self.embeddings_cache)
        }
