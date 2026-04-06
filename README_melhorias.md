# Radical Synthesis — Instruções de integração

Este pacote contém os arquivos completos com todas as melhorias aplicadas.

## Arquivos incluídos

| Arquivo | O que faz |
|---|---|
| `adaptive_cap.py` | Controle adaptativo de VRAM — aperta/afrouxa o limite de experts |
| `phi_mmd.py` | Métrica Φ estável para pools grandes (substituição do original) |
| `lazy_router.py` | Router com cache incremental — sem rebuild completo a cada evento |
| `genealogy.py` | Rastreamento da árvore genealógica de experts |
| `layer.py` | OuroborosMoELayer com todas as melhorias integradas |
| `ablation_test.py` | Script para testar a contribuição de cada mecanismo |
| `benchmark.py` | Avaliação no WikiText-103 para comparação com literatura |

---

## Como instalar

### Passo 1 — Copiar os arquivos novos

Copie os seguintes arquivos para dentro da pasta `radical_synthesis/` do projeto:

```
adaptive_cap.py   → radical_synthesis/adaptive_cap.py
phi_mmd.py        → radical_synthesis/phi_mmd.py
lazy_router.py    → radical_synthesis/lazy_router.py
genealogy.py      → radical_synthesis/genealogy.py
```

### Passo 2 — Substituir o layer.py

**Faça backup do original primeiro:**
```bash
cp radical_synthesis/layer.py radical_synthesis/layer_ORIGINAL.py
```

Depois substitua pelo novo:
```bash
cp layer.py radical_synthesis/layer.py
```

### Passo 3 — Copiar os scripts de análise

Os scripts abaixo ficam na **raiz do projeto** (fora de `radical_synthesis/`):
```
ablation_test.py  → ablation_test.py
benchmark.py      → benchmark.py
```

### Passo 4 — Verificar a instalação

```bash
cd radical-synthesis-The-First-Living-MoE-Mitosis-Apoptose-in-real-time-
pip install -e .
python -c "from radical_synthesis import OuroborosMoELayer; print('OK')"
```

Se aparecer `OK`, tudo funcionou.

---

## Como usar no treino

```python
import torch
from radical_synthesis import OuroborosMoELayer

# Cria a camada
layer = OuroborosMoELayer(
    d_model=512,
    d_ff=2048,
    n_experts=8,
    top_k=2,
    base_cap=256,   # ajuste para sua GPU
).cuda()

# Loop de treino
for step in range(100_000):
    x    = torch.randn(32, 128, 512).cuda()
    out  = layer(x)
    loss = out.mean()
    loss.backward()
    optimizer.step()

    # A cada 1.000 steps, executa o ciclo de vida
    if step % 1_000 == 0:
        dead, born = layer.execute_systemic_lifecycle(
            current_loss=loss.item(),
            step=step,
        )
        print(f"Step {step}: {len(born)} nascimentos, {len(dead)} mortes, "
              f"{layer.n_experts} experts ativos")

# Ao final, salva a genealogia
layer.save_genealogy("genealogy_final.json")
```

---

## Como rodar a ablação

```bash
python ablation_test.py
```

Leva ~15 minutos em GPU. Ao final, imprime uma tabela comparando as 4 configurações.

---

## Como rodar o benchmark

```bash
pip install datasets transformers
python benchmark.py
```

---

## Perguntas frequentes

**Q: Apareceu `ModuleNotFoundError: No module named 'adaptive_cap'`**  
R: Confirme que o arquivo está em `radical_synthesis/adaptive_cap.py` (com a pasta, não na raiz).

**Q: Apareceu `CUDA out of memory`**  
R: Diminua o `base_cap` ao criar a camada: `OuroborosMoELayer(..., base_cap=64)`.

**Q: O layer.py novo é compatível com o código existente?**  
R: Sim. A assinatura do `__init__` e do `forward` é a mesma do original. Os novos parâmetros têm valores padrão.
