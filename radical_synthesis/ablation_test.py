# ablation_test.py
# ─────────────────────────────────────────────────────────────────────────────
# Script de ablação: roda o treino 4 vezes com diferentes combinações
# de features para medir a contribuição de cada mecanismo.
#
# COMO RODAR:
#   python ablation_test.py
#
# O script imprime no terminal os resultados de cada configuração.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

# Ajuste esse import para o caminho correto no seu projeto
from radical_synthesis import OuroborosMoELayer


# ─────────────────────────────────────────────────────────────────────────────
# Configurações a testar
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "name":       "completo",
        "mitosis":    True,
        "apoptosis":  True,
        "escape":     True,
        "descricao":  "Tudo ativo — comportamento original melhorado",
    },
    {
        "name":       "sem_escape",
        "mitosis":    True,
        "apoptosis":  True,
        "escape":     False,
        "descricao":  "Sem escape topológico — mede contribuição do escape",
    },
    {
        "name":       "so_escape",
        "mitosis":    False,
        "apoptosis":  False,
        "escape":     True,
        "descricao":  "Só escape, sem crescimento — escape sozinho ajuda?",
    },
    {
        "name":       "baseline_moe",
        "mitosis":    False,
        "apoptosis":  False,
        "escape":     False,
        "descricao":  "MoE estático puro — baseline de comparação",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros do experimento
# ─────────────────────────────────────────────────────────────────────────────

STEPS       = 20_000   # 20k steps é suficiente para comparar contribuições
SEED        = 42       # semente fixa para reprodutibilidade
D_MODEL     = 512
D_FF        = 2048
N_EXPERTS   = 8
TOP_K       = 2
BATCH_SIZE  = 32
SEQ_LEN     = 128
LIFECYCLE_EVERY = 1_000  # executa lifecycle a cada N steps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando dispositivo: {DEVICE}")
print(f"Steps por config:   {STEPS:,}")
print(f"Configurações:      {len(CONFIGS)}\n")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Função de treino para uma configuração
# ─────────────────────────────────────────────────────────────────────────────

def run_config(cfg: dict, seed: int) -> dict:
    """
    Treina uma configuração por STEPS steps e retorna as métricas.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Cria a camada com os flags desta configuração
    layer = OuroborosMoELayer(
        d_model=D_MODEL,
        d_ff=D_FF,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        base_cap=128,             # cap conservador para ablação
        enable_mitosis=cfg["mitosis"],
        enable_apoptosis=cfg["apoptosis"],
        enable_escape=cfg["escape"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-4)

    losses:          list[float] = []
    expert_counts:   list[int]   = []
    plateaux_escaped: int        = 0
    prev_stagnating: bool        = False

    for step in range(1, STEPS + 1):
        # ── Dado sintético (substitua pelo seu dado real) ─────────────────
        x      = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE)

        optimizer.zero_grad()
        out  = layer(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        expert_counts.append(layer.n_experts)

        # ── Detecta plateaux superados ────────────────────────────────────
        if step > 200:
            recent_delta = losses[-200] - losses[-1]
            stagnating   = recent_delta < 0.001
            if prev_stagnating and not stagnating:
                plateaux_escaped += 1
            prev_stagnating = stagnating

        # ── Lifecycle periódico ───────────────────────────────────────────
        if step % LIFECYCLE_EVERY == 0:
            layer.execute_systemic_lifecycle(
                current_loss=loss_val,
                step=step,
            )

        # ── Log de progresso ──────────────────────────────────────────────
        if step % 5_000 == 0:
            print(
                f"  [{cfg['name']:15s}] step {step:>6,} | "
                f"loss={loss_val:.4f} | "
                f"experts={layer.n_experts:>5,}"
            )

    return {
        "name":            cfg["name"],
        "descricao":       cfg["descricao"],
        "loss_final":      round(losses[-1], 5),
        "loss_inicial":    round(losses[0],  5),
        "reducao_loss":    round(losses[0] - losses[-1], 5),
        "experts_final":   expert_counts[-1],
        "plateaux_escaped": plateaux_escaped,
        "losses":          losses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Executa todos os configs e imprime o resultado
# ─────────────────────────────────────────────────────────────────────────────

resultados = []

for i, cfg in enumerate(CONFIGS):
    print(f"\n[{i+1}/{len(CONFIGS)}] Rodando: {cfg['name']}")
    print(f"          {cfg['descricao']}")
    resultado = run_config(cfg, seed=SEED)
    resultados.append(resultado)
    print(f"  → Loss final: {resultado['loss_final']:.5f} "
          f"| Redução: {resultado['reducao_loss']:+.5f} "
          f"| Experts: {resultado['experts_final']}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabela de resultados
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("  RESULTADO DA ABLAÇÃO")
print("=" * 65)
print(f"{'Config':<18} {'Loss final':>12} {'Redução':>10} {'Experts':>9} {'Plateaux':>10}")
print("-" * 65)

baseline = next(r for r in resultados if r["name"] == "baseline_moe")

for r in resultados:
    delta_vs_baseline = r["loss_final"] - baseline["loss_final"]
    marker = " ←" if r["name"] == "completo" else ""
    print(
        f"{r['name']:<18} "
        f"{r['loss_final']:>12.5f} "
        f"{r['reducao_loss']:>+10.5f} "
        f"{r['experts_final']:>9,} "
        f"{r['plateaux_escaped']:>10}"
        f"{marker}"
    )

print("=" * 65)
print(f"\nReferência: {baseline['name']} com loss {baseline['loss_final']:.5f}")
print("Quanto menor o loss, melhor.")
print("O 'completo' deve ter o menor loss de todos.")

# ─────────────────────────────────────────────────────────────────────────────
# Salva os resultados em JSON para análise posterior
# ─────────────────────────────────────────────────────────────────────────────

import json, os

output = {
    "parametros": {
        "steps":     STEPS,
        "seed":      SEED,
        "d_model":   D_MODEL,
        "n_experts": N_EXPERTS,
        "device":    DEVICE,
    },
    "resultados": [
        {k: v for k, v in r.items() if k != "losses"}
        for r in resultados
    ],
}

with open("ablation_results.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nResultados salvos em 'ablation_results.json'")
