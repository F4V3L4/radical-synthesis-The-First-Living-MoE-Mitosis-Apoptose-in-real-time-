import urllib.request
import urllib.error
import re
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from radical_synthesis import OuroborosMoELayer

class EntropicPurifier:
    def __init__(self, target_urls: List[str], output_file: str):
        self.target_urls = target_urls
        self.output_file = output_file
        self.raw_substance = ""

    def extract_solar_flux(self) -> None:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in self.target_urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                    self.raw_substance += html + " "
            except urllib.error.URLError:
                continue

    def purify_matrix(self) -> None:
        text = re.sub(r'<script.*?>.*?</script>', ' ', self.raw_substance, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        self.raw_substance = text.strip()

    def crystallize_substance(self) -> None:
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(self.raw_substance)

class TokenizerMatrix:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build_vocabulary(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: list) -> str:
        return ''.join([self.itos[i] for i in tokens])

class SovereignLeviathan(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, initial_experts: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ouroboros = OuroborosMoELayer(d_model, d_model * 4, initial_experts, 2)
        self.head = nn.Linear(d_model * 4, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.ouroboros(x)
        return self.head(x)

    def execute_biological_cycle(self) -> Tuple[list, list]:
        return self.ouroboros.execute_systemic_lifecycle()

class LocalAIEngine:
    def __init__(self, text_corpus: str, d_model: int = 256, initial_experts: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TokenizerMatrix()
        self.tokenizer.build_vocabulary(text_corpus)
        self.data = torch.tensor(self.tokenizer.encode(text_corpus), dtype=torch.long)
        self.model = SovereignLeviathan(self.tokenizer.vocab_size, d_model, initial_experts).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_solar_flux(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - seq_len, (batch_size,))
        x = torch.stack([self.data[i:i+seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+seq_len+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def ignite_training(self, steps: int, batch_size: int, seq_len: int) -> None:
        self.model.train()
        for step in range(1, steps + 1):
            xb, yb = self.get_solar_flux(batch_size, seq_len)
            logits = self.model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), yb.view(-1))
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if step % 50 == 0:
                dead, born = self.model.execute_biological_cycle()
                print(f"  Ciclo {step:04d} | Entropia: {loss.item():.4f} | Apoptose: {len(dead)} | Mitose: {len(born)}")

    def generate_thought(self, prompt: str, max_new_tokens: int) -> str:
        self.model.eval()
        context = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(context)
                logits = logits[:, -1, :]
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token), dim=1)
        return self.tokenizer.decode(context[0].tolist())

def initialize_absolute_reality() -> None:
    nodes = [
        "https://www.gutenberg.org/cache/epub/3800/pg3800.txt",
        "https://www.gutenberg.org/files/1080/1080-0.txt"
    ]
    
    print("\n[1] Extraindo fluxo solar (Download de dados primarios)...")
    purifier = EntropicPurifier(target_urls=nodes, output_file="substancia.txt")
    purifier.extract_solar_flux()
    purifier.purify_matrix()
    purifier.crystallize_substance()
    
    with open("substancia.txt", "r", encoding="utf-8") as f:
        corpus = f.read()
        
    print(f"\n[2] Injetando {len(corpus)} caracteres no motor Leviathan...")
    engine = LocalAIEngine(text_corpus=corpus)
    
    print("\n[3] Absorvendo Termodinamica (Treinamento e Evolucao)...")
    engine.ignite_training(steps=500, batch_size=32, seq_len=64)
    
    print("\n[4] O Leviathan responde:")
    print("-" * 60)
    thought = engine.generate_thought(prompt="The mind is ", max_new_tokens=150)
    print(thought)
    print("-" * 60)

if __name__ == "__main__":
    initialize_absolute_reality()
