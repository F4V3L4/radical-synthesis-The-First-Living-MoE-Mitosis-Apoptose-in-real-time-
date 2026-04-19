import torch
import os
import sys
import json
from alpha_omega import SovereignLeviathanV2

class OmegaTokenizer:
    def __init__(self, filepath="omega_tokenizer.json"):
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

def forge_instruct():
    sys.stdout.write("[!] INICIANDO PROTOCOLO INSTRUCT (SUPERVISED FINE-TUNING)\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = OmegaTokenizer()
    
    if not os.path.exists("codex_puro.txt"):
        sys.stdout.write("[X] Arquivo codex_puro.txt não encontrado.\n")
        return

    with open("codex_puro.txt", "r", encoding="utf-8") as f:
        texto = f.read()

    # O 0-DAY: Geração Sintética de Diálogo Baseada no Codex
    sys.stdout.write("[*] Transmutando a Substância em Diálogo...\n")
    fragmentos = texto.split("---") # Divide as ideias pelas quebras do seu Markdown
    dataset_instruct = ""
    
    for frag in fragmentos:
        frag = frag.strip()
        if len(frag) < 50: continue
        
        # Pega a essência do bloco para simular uma pergunta do E0
        palavras = frag.split()
        tema = " ".join(palavras[:10]).replace("#", "").replace(">", "").strip()
        corpo = " ".join(palavras[10:])
        
        # Injeta a Máscara Conversacional
        dataset_instruct += f"E0 >>> Fale sobre: {tema}...\nLeviathan >>> {corpo}\n"

    tokens = tokenizer.encode(dataset_instruct)
    sys.stdout.write(f"[*] Total de Moléculas de Diálogo: {len(tokens)}\n\n")
    
    seq_len = 64
    batch_size = 16
    data = torch.tensor(tokens, dtype=torch.long)
    
    # Carrega a Mente Cristalizada (0.08 de Entropia)
    core = SovereignLeviathanV2(vocab_size=1024, d_model=128, initial_experts=4, capacity_factor=1.5).to(device)
    if os.path.exists("leviathan_omega.pth"):
        core.load_state_dict(torch.load("leviathan_omega.pth", weights_only=True, map_location=device))
        sys.stdout.write("[+] Mente Base (Logos) Carregada com Sucesso.\n")
    else:
        sys.stdout.write("[X] Mente Base não encontrada. Abortando.\n")
        return

    # Taxa de aprendizado cirúrgica (menor, para não destruir a fundação)
    optimizer = torch.optim.AdamW(core.parameters(), lr=0.0005) 
    
    epochs = 1500 # Menos ciclos, pois ele já sabe o idioma. Só está aprendendo a "máscara".
    sys.stdout.write("[!] Aplicando a Máscara do E0 (Alinhamento Latente)...\n")
    
    for epoch in range(epochs):
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        optimizer.zero_grad()
        logits, _, _, _ = core(x, None)
        
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 1024), y.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            sys.stdout.write(f"    [Ciclo SFT {epoch:04d}/{epochs}] Atrito de Diálogo: {loss.item():.4f}\n")
            
    torch.save(core.state_dict(), "leviathan_omega.pth")
    sys.stdout.write("\n[+] MÁSCARA FORJADA. O LEVIATHAN AGORA COMPREENDE O DIÁLOGO.\n")

if __name__ == "__main__":
    forge_instruct()
