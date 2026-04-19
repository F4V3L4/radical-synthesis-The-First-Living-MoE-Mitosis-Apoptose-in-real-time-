# benchmark.py
# ─────────────────────────────────────────────────────────────────────────────
# Avaliação em benchmarks padronizados: WikiText-103 (perplexidade).
#
# PRÉ-REQUISITO — instale as bibliotecas:
#   pip install datasets transformers
#
# COMO RODAR:
#   python benchmark.py
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Instalação automática das dependências se não tiver
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys

def instalar_se_necessario(pacote):
    try:
        __import__(pacote.split("[")[0].replace("-", "_"))
    except ImportError:
        print(f"Instalando {pacote}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote, "-q"])

instalar_se_necessario("datasets")
instalar_se_necessario("transformers")

from datasets    import load_dataset
from transformers import AutoTokenizer

# Ajuste esse import para o caminho correto no seu projeto
from radical_synthesis import OuroborosMoELayer


# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros
# ─────────────────────────────────────────────────────────────────────────────

D_MODEL    = 512
D_FF       = 2048
N_EXPERTS  = 8     # começa com 8, igual ao baseline
TOP_K      = 2
SEQ_LEN    = 128   # tokens por amostra
N_AMOSTRAS = 500   # quantas amostras usar para o benchmark
MAX_EXPERTS = 128  # cap conservador para VRAM comparável ao MoE-64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Modelo simples para benchmark
# (Wrapper que usa o OuroborosMoELayer como FFN de um Transformer mínimo)
# ─────────────────────────────────────────────────────────────────────────────

class MiniTransformer(nn.Module):
    """
    Transformer mínimo para fins de benchmark.
    Usa o OuroborosMoELayer como camada FFN.
    Substitua por seu modelo real se tiver um.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Embedding(SEQ_LEN, d_model)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            for _ in range(n_layers)
        ])
        self.moe_layers = nn.ModuleList([
            OuroborosMoELayer(
                d_model=d_model, d_ff=D_FF, n_experts=N_EXPERTS, top_k=TOP_K,
                base_cap=MAX_EXPERTS,
            )
            for _ in range(n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head  = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embedding(tokens) + self.pos_emb(pos)

        for i, (attn, moe, n1, n2) in enumerate(
            zip(self.attn_layers, self.moe_layers, self.norm1, self.norm2)
        ):
            # Self-attention
            attn_out, _ = attn(x, x, x, need_weights=False)
            x = n1(x + attn_out)

            # MoE FFN
            moe_out = moe(x)
            x = n2(x + moe_out)

        return self.head(x)  # (B, T, vocab_size)

    def execute_lifecycle(self, loss_val: float, step: int):
        """Executa lifecycle em todas as camadas MoE."""
        for layer in self.moe_layers:
            layer.execute_systemic_lifecycle(current_loss=loss_val, step=step)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark de perplexidade no WikiText-103
# ─────────────────────────────────────────────────────────────────────────────

def calcular_perplexidade(modelo, tokenizer, textos: list[str]) -> float:
    """
    Calcula a perplexidade do modelo nos textos fornecidos.
    Perplexidade menor = modelo melhor.
    """
    modelo.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for texto in textos:
            if not texto.strip():
                continue

            # Tokeniza e trunca/pad para SEQ_LEN
            enc = tokenizer(
                texto,
                return_tensors="pt",
                max_length=SEQ_LEN + 1,
                truncation=True,
                padding="max_length",
            )
            tokens = enc["input_ids"].to(DEVICE)  # (1, SEQ_LEN+1)

            if tokens.shape[1] < 2:
                continue

            input_ids  = tokens[:, :-1]   # tokens de entrada: (1, T)
            target_ids = tokens[:, 1:]    # tokens alvo:       (1, T)

            logits = modelo(input_ids)    # (1, T, vocab_size)

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_ids.view(-1),
                ignore_index=tokenizer.pad_token_id or 0,
                reduction="sum",
            )
            n_tokens = (target_ids != (tokenizer.pad_token_id or 0)).sum().item()

            total_loss   += loss.item()
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss     = total_loss / total_tokens
    perplexidade = math.exp(avg_loss)
    return perplexidade


# ─────────────────────────────────────────────────────────────────────────────
# Execução principal
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  BENCHMARK — WikiText-103 Perplexidade")
print("=" * 60)

# ── 1. Carrega o dataset ─────────────────────────────────────────────────────
print("\n[1/4] Carregando WikiText-103...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
textos  = [t for t in dataset["text"] if len(t.strip()) > 50][:N_AMOSTRAS]
print(f"      {len(textos)} textos carregados.")

# ── 2. Tokenizer ─────────────────────────────────────────────────────────────
print("\n[2/4] Carregando tokenizer (GPT-2)...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

# ── 3. Cria e avalia o modelo Radical Synthesis ──────────────────────────────
print("\n[3/4] Criando modelo Radical Synthesis...")
modelo_rs = MiniTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=2,
).to(DEVICE)

n_params = sum(p.numel() for p in modelo_rs.parameters()) / 1e6
print(f"      Parâmetros: {n_params:.1f}M")
print("      Calculando perplexidade (pode demorar alguns minutos)...")

ppl_rs = calcular_perplexidade(modelo_rs, tokenizer, textos)

# ── 4. Baseline: MoE estático com mesmos parâmetros ─────────────────────────
print("\n[4/4] Calculando baseline (MoE estático)...")

class MoEEstatico(nn.Module):
    """MoE estático simples para comparação justa."""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(SEQ_LEN, d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, D_FF), nn.GELU(), nn.Linear(D_FF, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embedding(tokens) + self.pos_emb(pos)
        x    = self.norm(x + self.ffn(x))
        return self.head(x)

modelo_baseline = MoEEstatico(VOCAB_SIZE, D_MODEL).to(DEVICE)
ppl_baseline    = calcular_perplexidade(modelo_baseline, tokenizer, textos)

# ─────────────────────────────────────────────────────────────────────────────
# Resultados
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  RESULTADOS")
print("=" * 60)
print(f"  Radical Synthesis   : perplexidade = {ppl_rs:.2f}")
print(f"  MoE estático        : perplexidade = {ppl_baseline:.2f}")
print(f"  Diferença           : {ppl_rs - ppl_baseline:+.2f}")
print()

if ppl_rs < ppl_baseline:
    reducao = (ppl_baseline - ppl_rs) / ppl_baseline * 100
    print(f"  Radical Synthesis é {reducao:.1f}% MELHOR que o baseline.")
else:
    aumento = (ppl_rs - ppl_baseline) / ppl_baseline * 100
    print(f"  Radical Synthesis é {aumento:.1f}% pior que o baseline sem treino.")
    print("  Isso é esperado sem treino — o modelo não foi fine-tuned.")
    print("  Para resultados reais, treine o modelo e avalie novamente.")

print("=" * 60)
print()
print("NOTA: Este benchmark usa um modelo sem treino prévio.")
print("Para comparação científica, treine ambos os modelos por")
print("igual número de steps antes de avaliar a perplexidade.")
print()
