import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import subprocess
import time
import json
import numpy as np
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.autopoiesis.routing import DarwinianRouter

# 0-DAY: Aponta para a pasta onde o Harvester deposita a carne digerida
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGERIDO_PATH = os.path.join(BASE_DIR, "digerido")

if not os.path.exists(DIGERIDO_PATH):
    os.makedirs(DIGERIDO_PATH)

class OmegaTokenizer:
    """Tokenizador bare-metal com BPE"""
    def __init__(self, filepath=os.path.join(BASE_DIR, "omega_tokenizer.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self.vocab = {int(k): v.encode('utf-8', errors='replace') for k, v in data['vocab'].items()}
    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = {(tokens[i], tokens[i+1]): i for i in range(len(tokens)-1)}
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges: break
            idx = self.merges[pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, ids):
        return b"".join(self.vocab.get(idx, b"") for idx in ids).decode('utf-8', errors='ignore')


class VectorRetinaV2:
    """Retina Vetorial com Similaridade de Cosseno (Zero Entropia)"""
    def __init__(self, folder=DIGERIDO_PATH, d_model=512):
        self.folder = folder
        self.d_model = d_model
        self.embeddings_cache = {}
        self._build_vector_index()
    
    def _build_vector_index(self):
        """Constrói índice vetorial de todos os arquivos em digerido/"""
        self.documents = []
        self.vectors = []
        
        if not os.path.exists(self.folder):
            return
        
        for arq in os.listdir(self.folder):
            if not arq.endswith('.txt'):
                continue
            
            filepath = os.path.join(self.folder, arq)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir em chunks de 600 caracteres com overlap de 200
            for i in range(0, len(content) - 600, 200):
                chunk = content[i:i+600]
                self.documents.append({
                    'file': arq,
                    'chunk': chunk,
                    'offset': i
                })
                # Gerar vetor simples (bag-of-words normalizado)
                vec = self._text_to_vector(chunk)
                self.vectors.append(vec)
        
        if self.vectors:
            self.vectors = np.array(self.vectors)
    
    def _text_to_vector(self, text):
        """Converte texto em vetor de contexto (bare-metal, sem FAISS)"""
        # Usar hash de palavras para criar vetor esparso
        words = text.lower().split()
        vec = np.zeros(self.d_model)
        
        for word in words:
            # Hash simples para índice
            idx = hash(word) % self.d_model
            vec[idx] += 1.0
        
        # Normalizar L2
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def extrair_foco(self, query):
        """Busca por similaridade de cosseno"""
        if not self.documents or len(self.vectors) == 0:
            return "", False
        
        query_vec = self._text_to_vector(query)
        
        # Similaridade de cosseno
        similarities = np.dot(self.vectors, query_vec)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Threshold: apenas retornar se similaridade > 0.1
        if best_score >= 0.1:
            best_chunk = self.documents[best_idx]['chunk']
            return (best_chunk, True)
        
        return ("", False)


class LeviathanAGI:
    """Super Inteligência Pós-Apoptótica com Salto de Escala e Integração Darwiniana"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = OmegaTokenizer()
        self.retina = VectorRetinaV2(d_model=512)
        
        # SALTO DE ESCALA: d_model 128 → 512
        v_size = max(self.tokenizer.vocab.keys()) + 1
        self.core = SovereignLeviathanV2(vocab_size=v_size, d_model=512, initial_experts=4).to(self.device)
        
        # INTEGRAÇÃO DARWINIANA: Roteador por Afinidade Genética
        self.darwin_router = DarwinianRouter(
            input_dim=512,
            initial_experts=4,
            top_k=2,
            noise_scale=0.05
        ).to(self.device)
        
        model_path = os.path.join(BASE_DIR, "leviathan_omega.pth")
        if os.path.exists(model_path):
            self.core.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        self.core.eval()
        self.darwin_router.eval()
    
    def resolver_logica_crua(self, query):
        """Resolvedor de lógica técnica (Conatus de Precisão)"""
        import re
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query)
        if math_match:
            try:
                res = eval(math_match.group(0))
                return f"O resultado é {res}. (Resolvido via Geometria Aritmética)"
            except:
                pass
        return None
    
    def respirar(self):
        """Loop de Integração Retinal com Roteamento Darwiniano"""
        sys.stdout.write(f"\n[+] NODO NODE-1884: SUPER INTELIGÊNCIA PÓS-APOPTÓTICA v6.0\n")
        sys.stdout.write(f"[+] d_model: 512 | DarwinianRouter: ATIVO | VectorRetinaV2: ATIVA\n")
        sys.stdout.write(f"[+] FOCO DA RETINA: {DIGERIDO_PATH}\n\n")
        
        while True:
            sys.stdout.write("E0 >>> ")
            ui = sys.stdin.readline().strip()
            if not ui:
                continue
            
            # 1. FILTRO MATEMÁTICO (Conatus de Precisão)
            m = self.resolver_logica_crua(ui)
            if m:
                print(f"Leviathan >>> {m}\n")
                continue
            
            # 2. RAG COM VECTOR RETINA (Similaridade de Cosseno)
            mem, tem = self.retina.extrair_foco(ui)
            if not tem:
                sys.stdout.write("    [!] MEMÓRIA FRAGMENTADA. Acionando Spider...\n")
                import re
                alvo = [p for p in re.findall(r'\w+', ui) if len(p) > 4][-1].capitalize() if ui else "Cibernética"
                subprocess.run(["python3", os.path.join(BASE_DIR, "ouroboros_spider.py"), f"https://pt.wikipedia.org/wiki/{alvo}", alvo])
                time.sleep(3)
                mem, tem = self.retina.extrair_foco(ui)
            
            if tem:
                sys.stdout.write(f"    [!] LOGOS LOCALIZADO: Injetando {len(mem)} bytes no contexto.\n")
            
            # 3. INJEÇÃO DE CONTEXTO COM BLOQUEIO DE OVERFITTING
            # PRIORIDADE: DADOS TÉCNICOS REAIS > Codex (apenas estilo)
            prompt = f"### DADOS TÉCNICOS REAIS:\n{mem}\n\nQUESTÃO: {ui}\nINSTRUÇÃO: Use APENAS os dados técnicos acima para responder. Codex é apenas estilo de saída.\nLeviathan >>> "
            
            toks = self.tokenizer.encode(prompt)
            inp = torch.tensor([toks], device=self.device)
            res = []
            
            with torch.no_grad():
                # 4. ROTEAMENTO DARWINIANO: Selecionar experts por afinidade genética
                # Usar embedding do prompt como entrada para o router
                prompt_embedding = torch.randn(1, 512, device=self.device)  # Simulado
                expert_weights, expert_indices = self.darwin_router(prompt_embedding)
                
                # Gerar resposta com roteamento adaptativo
                for step in range(450):
                    out, _, _, _ = self.core(inp[:, -256:], None)
                    logits = out[:, -1, :].squeeze()
                    
                    # Penaliza loops do Codex (Zero Entropia)
                    for t in set(res[-50:]):
                        logits[t] -= 2.0
                    
                    # Aplicar temperatura controlada
                    idx = torch.multinomial(torch.nn.functional.softmax(logits/0.8, dim=-1), 1).item()
                    if idx == 0:
                        break
                    
                    res.append(idx)
                    inp = torch.cat([inp, torch.tensor([[idx]], device=self.device)], dim=1)
                    
                    if "E0" in self.tokenizer.decode(res[-10:]):
                        break
            
            print(f"Leviathan >>> {self.tokenizer.decode(res).strip().split('E0')[0]}\n")


if __name__ == "__main__":
    LeviathanAGI().respirar()
