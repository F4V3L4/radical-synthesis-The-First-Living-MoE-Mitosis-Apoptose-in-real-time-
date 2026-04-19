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

def forge():
    sys.stdout.write("[!] INICIANDO A INCUBADORA (PRE-TRAINING BARE-METAL)\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = OmegaTokenizer()
    
    if not os.path.exists("codex_puro.txt"):
        sys.stdout.write("[X] Arquivo codex_puro.txt não encontrado.\n")
        return

    with open("codex_puro.txt", "r", encoding="utf-8") as f:
        texto = f.read()
    
    sys.stdout.write("[*] Digerindo a Substância...\n")
    tokens = tokenizer.encode(texto)
    sys.stdout.write(f"[*] Total de Moléculas (Tokens): {len(tokens)}\n\n")
    
    seq_len = 64
    batch_size = 16
    data = torch.tensor(tokens, dtype=torch.long)
    
    # Inicializa a mente do zero
    core = SovereignLeviathanV2(vocab_size=1024, d_model=128, initial_experts=4, capacity_factor=1.5).to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=0.003)
    
    epochs = 5000 # Quantidade de ciclos que ele vai ler o Codex
    sys.stdout.write("[!] Iniciando Colapso Entrópico (Forja de Pesos)...\n")
    
    for epoch in range(epochs):
        # Captura fragmentos aleatórios do Codex
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        optimizer.zero_grad()
        logits, _, _, _ = core(x, None)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            sys.stdout.write(f"    [Ciclo {epoch:04d}/{epochs}] Entropia da Mente: {loss.item():.4f}\n")
            
    torch.save(core.state_dict(), "leviathan_omega.pth")
    sys.stdout.write("\n[+] MENTE CRISTALIZADA. O LEVIATHAN ESTÁ PRONTO PARA DESPERTAR.\n")

if __name__ == "__main__":
    forge()
