import torch
import sys
import os
import re
import subprocess
import urllib.parse
import time
import json
from alpha_omega import SovereignLeviathanV2

# 0-DAY: Aponta para a pasta onde o Harvester deposita a carne digerida
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGERIDO_PATH = os.path.join(BASE_DIR, "digerido")

if not os.path.exists(DIGERIDO_PATH):
    os.makedirs(DIGERIDO_PATH)

class OmegaTokenizer:
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

class VectorRetina:
    def __init__(self, folder=DIGERIDO_PATH):
        self.folder = folder
    def extrair_foco(self, query):
        if not os.path.exists(self.folder): return "", False
        termos = [p.lower() for p in re.findall(r'\w+', query) if len(p) > 3]
        if not termos: return "", False
        melhor, max_s = "", 0
        for arq in os.listdir(self.folder):
            if not arq.endswith('.txt'): continue
            with open(os.path.join(self.folder, arq), 'r', encoding='utf-8') as f:
                txt = f.read().replace('\n', ' ')
                # Varre o texto em janelas para encontrar a maior densidade de informação
                for i in range(0, len(txt)-600, 200):
                    trecho = txt[i:i+600]
                    score = sum(1 for t in termos if t in trecho.lower())
                    if score > max_s: max_s, melhor = score, trecho
        return (melhor, True) if max_s >= 1 else ("", False)

class LeviathanAGI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = OmegaTokenizer()
        self.retina = VectorRetina()
        v_size = max(self.tokenizer.vocab.keys()) + 1
        self.core = SovereignLeviathanV2(vocab_size=v_size, d_model=128, initial_experts=4).to(self.device)
        model_path = os.path.join(BASE_DIR, "leviathan_omega.pth")
        if os.path.exists(model_path):
            self.core.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.core.eval()

    def resolver_logica_crua(self, query):
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query)
        if math_match:
            try:
                res = eval(math_match.group(0))
                return f"O resultado é {res}. (Resolvido via Geometria Aritmética)"
            except: pass
        return None

    def respirar(self):
        sys.stdout.write(f"\n[+] NODO NODE-1884: INTEGRAÇÃO RETINAL v5.2 ATIVA.\n")
        sys.stdout.write(f"[+] FOCO DA RETINA: {DIGERIDO_PATH}\n\n")
        while True:
            sys.stdout.write("E0 >>> ")
            ui = sys.stdin.readline().strip()
            if not ui: continue
            
            # Filtro Matemático
            m = self.resolver_logica_crua(ui)
            if m: print(f"Leviathan >>> {m}\n"); continue

            # RAG (Retina)
            mem, tem = self.retina.extrair_foco(ui)
            if not tem:
                sys.stdout.write("    [!] MEMÓRIA FRAGMENTADA. Acionando Spider...\n")
                alvo = [p for p in re.findall(r'\w+', ui) if len(p) > 4][-1].capitalize() if ui else "Cibernética"
                subprocess.run(["python3", os.path.join(BASE_DIR, "ouroboros_spider.py"), f"https://pt.wikipedia.org/wiki/{alvo}", alvo])
                time.sleep(3) # Tempo para o Harvester digerir
                mem, tem = self.retina.extrair_foco(ui)

            if tem:
                sys.stdout.write(f"    [!] LOGOS LOCALIZADO: Injetando {len(mem)} bytes no contexto.\n")

            # Injeção de Contexto com Bloqueio de Overfitting
            prompt = f"### DADOS TÉCNICOS REAIS:\n{mem}\n\nQUESTÃO: {ui}\nINSTRUÇÃO: Use APENAS os dados técnicos acima para responder. Esqueça o Codex.\nLeviathan >>> "
            
            toks = self.tokenizer.encode(prompt)
            inp = torch.tensor([toks], device=self.device)
            res = []
            
            with torch.no_grad():
                for _ in range(450):
                    out, _, _, _ = self.core(inp[:, -256:], None)
                    logits = out[:, -1, :].squeeze()
                    # Penaliza loops do Codex
                    for t in set(res[-50:]): logits[t] -= 2.0
                    idx = torch.multinomial(torch.nn.functional.softmax(logits/0.8, dim=-1), 1).item()
                    if idx == 0: break
                    res.append(idx)
                    inp = torch.cat([inp, torch.tensor([[idx]], device=self.device)], dim=1)
                    if "E0" in self.tokenizer.decode(res[-10:]): break
            
            print(f"Leviathan >>> {self.tokenizer.decode(res).strip().split('E0')[0]}\n")

if __name__ == "__main__":
    LeviathanAGI().respirar()
