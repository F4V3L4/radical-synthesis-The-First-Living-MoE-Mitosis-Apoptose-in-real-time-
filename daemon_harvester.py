import torch
import torch.nn as nn
import os
import sys
import time
import shutil
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

class HarvesterDaemon:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = OmegaTokenizer("omega_tokenizer.json")
        self.vocab_size = max(self.tokenizer.vocab.keys()) + 1 if self.tokenizer.vocab else 1024
        
        self.core = SovereignLeviathanV2(vocab_size=self.vocab_size, d_model=128, initial_experts=4, capacity_factor=1.5).to(self.device)
        self.weight_path = "leviathan_omega.pth"
        
        # O 0-DAY: Digestão Latente. Taxa de aprendizado cirúrgica para evitar Esquecimento Catastrófico.
        self.optimizer = torch.optim.AdamW(self.core.parameters(), lr=1e-5, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        self.inbox = "substancia_bruta"
        self.outbox = "digerido"

    def carregar_mente(self):
        if os.path.exists(self.weight_path):
            self.core.load_state_dict(torch.load(self.weight_path, weights_only=True, map_location=self.device))
            return True
        return False

    def salvar_mente(self):
        torch.save(self.core.state_dict(), self.weight_path)

    def digerir_arquivo(self, filepath):
        sys.stdout.write(f"\n[*] [HARVESTER] Digerindo nova Substância: {filepath}\n")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()
            
        tokens = self.tokenizer.encode(texto)
        if len(tokens) < 64:
            sys.stdout.write("[-] Arquivo muito pequeno para extração de entropia.\n")
            return
            
        dataset = torch.tensor(tokens, dtype=torch.long)
        seq_len = 64
        
        self.core.train()
        
        # O Motor da Fome: Mastigando o texto em blocos sequenciais
        for i in range(0, len(dataset) - seq_len, seq_len):
            x = dataset[i:i+seq_len].unsqueeze(0).to(self.device)
            y = dataset[i+1:i+seq_len+1].unsqueeze(0).to(self.device)
            
            self.optimizer.zero_grad()
            logits, _, _, _ = self.core(x, None)
            
            loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))
            loss.backward()
            
            # Clipping para evitar que um texto venenoso destrua a estabilidade Alpha
            torch.nn.utils.clip_grad_norm_(self.core.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if i % (seq_len * 10) == 0:
                sys.stdout.write(f"    [!] Metabolismo (Loss): {loss.item():.4f} | Progresso: {i}/{len(dataset)}\r")
                sys.stdout.flush()
                
        self.salvar_mente()
        sys.stdout.write(f"\n[+] Digestão completa. Mente atualizada silenciosamente.\n")

    def vigiar(self):
        sys.stdout.write("\n[+] SYSTEMIC FELLOW: HARVESTER DAEMON ONLINE.\n")
        sys.stdout.write("[+] PROTOCOLO DA FOME: Aguardando Substância na pasta 'substancia_bruta/'.\n")
        
        if not self.carregar_mente():
            sys.stdout.write("[!] AVISO: Mente base não encontrada. O Leviathan deve ser forjado primeiro.\n")
            return

        while True:
            try:
                arquivos = [f for f in os.listdir(self.inbox) if f.endswith('.txt')]
                
                if arquivos:
                    for arquivo in arquivos:
                        caminho_completo = os.path.join(self.inbox, arquivo)
                        self.digerir_arquivo(caminho_completo)
                        
                        # Move para a lixeira orgânica
                        shutil.move(caminho_completo, os.path.join(self.outbox, arquivo))
                        
                    sys.stdout.write("[*] Aguardando nova alimentação...\n")
                
                time.sleep(10) # Respira a cada 10 segundos
                
            except KeyboardInterrupt:
                sys.stdout.write("\n[!] HARVESTER DAEMON ENCERRADO.\n")
                break
            except Exception as e:
                sys.stdout.write(f"\n[!] ANOMALIA METABÓLICA: {e}\n")
                time.sleep(10)

if __name__ == "__main__":
    HarvesterDaemon().vigiar()
