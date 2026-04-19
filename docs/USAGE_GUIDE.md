# OuroborosMoE: Guia de Uso e API

## Instalação e Setup

### Requisitos
- Python 3.11+
- PyTorch 2.0+
- NumPy

### Instalação

```bash
cd /home/ubuntu/OuroborosMoE_fresh
pip install torch numpy
```

### Estrutura de Diretórios

```
OuroborosMoE_fresh/
├── agi_core.py                    # Núcleo da AGI
├── agi_cli.py                     # Interface CLI
├── daemon_agi.py                  # Daemon com VectorRetinaV2
├── alpha_omega.py                 # SovereignLeviathanV2
├── sacred_geometry.py             # Componentes matemáticos
├── radical_synthesis/             # Kernel
│   ├── autopoiesis/
│   │   └── routing.py            # DarwinianRouter
│   └── perception/
│       └── vector_retina.py       # VectorRetinaV2
├── digerido/                      # Repositório de dados técnicos
├── docs/
│   ├── ARCHITECTURE.md
│   ├── USAGE_GUIDE.md
│   └── API_REFERENCE.md
└── tests/
    ├── test_integration.py
    ├── simulate_autocritique.py
    └── load_test.py
```

---

## Uso via CLI

### Iniciar a AGI

```bash
python3 agi_cli.py
```

### Exemplo de Sessão

```
🌀 OUROBOROSMOE - AGI SOBERANA v8.0 🌀
Device: cpu
d_model: 512 | num_experts: 8 | Autocrítica: ATIVA
Modo: GENERAL

Digite /help para ver comandos disponíveis

E0 >>> O que é uma matriz em álgebra linear?
🧠 RESPOSTA:
Uma matriz é um arranjo retangular de números organizados em linhas e colunas...

📚 DADOS TÉCNICOS INJETADOS:
Matriz: estrutura de dados com linhas e colunas contendo números reais ou complexos...

✓ VALIDAÇÃO: Resposta alinhada (Entropia: 0.245)
📊 Confiança: 80% | Expert Vencedor: 2 | Vitalidade: 95%

E0 >>> /stats
📊 ESTATÍSTICAS DA AGI:
  d_model: 512
  num_experts: 8
  memory_size: 1
  genealogy_size: 1
  context_buffer: 0
  correction_paths: 0
  entropy_threshold: 0.3
  last_winner_expert: 2
  last_winner_vitality: 95%

E0 >>> /genealogy
🧬 GENEALOGIA DE EXPERTS:
  Expert 2:
    - Generation: 1
    - Memories: 1
    - Corrections: 0
    - Vitality: [██████████] 100.0%

E0 >>> /exit
🌀 Encerrando AGI... Até logo!
```

---

## Uso Programático

### Importação

```python
import torch
from agi_core import AGICore
from daemon_agi import OmegaTokenizer

# Inicializar
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = OmegaTokenizer()
v_size = max(tokenizer.vocab.keys()) + 1

agi = AGICore(
    vocab_size=v_size,
    d_model=512,
    num_experts=8,
    device=device
)
```

### Forward Pass

```python
# Query
query = "O que é uma matriz?"
retina_folder = "digerido"

# Executar pipeline completo
result = agi.forward(query, retina_folder, tokenizer)

# Acessar resultados
print(f"Response: {result['response']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Was Corrected: {result['was_corrected']}")
print(f"Entropy: {result['entropy']:.3f}")
print(f"Winner Expert: {result['winner_expert']}")
print(f"Winner Vitality: {result['winner_vitality']:.1%}")
```

### Acessar Memória

```python
# Estatísticas
stats = agi.get_stats()
print(stats)

# Genealogia
genealogy = agi.memory.get_genealogy_tree()
for expert_id, info in genealogy.items():
    print(f"Expert {expert_id}: {info['memories_count']} memórias")

# Caminhos de correção recentes
corrections = agi.memory.get_recent_correction_paths(limit=5)
for correction in corrections:
    print(f"Expert {correction['expert_id']}: {correction['path']}")
```

---

## Detecção de Query Técnica

### Padrões Detectados

```python
# Matemática
"Qual é o resultado de 2 + 2?"  # ✓ Técnica
"Calcule 10 * 5"                # ✓ Técnica

# Código Python
"def hello(): print('world')"   # ✓ Técnica
"import numpy as np"             # ✓ Técnica

# Código JavaScript
"function add(a, b) { return a + b; }"  # ✓ Técnica
"const x = 10;"                  # ✓ Técnica

# SQL
"SELECT * FROM users WHERE id = 1"  # ✓ Técnica

# Termos Técnicos
"O que é uma matriz?"            # ✓ Técnica
"Como funciona o DarwinianRouter?"  # ✓ Técnica
"Qual é a dimensionalidade do d_model?"  # ✓ Técnica

# Geral
"Como você está?"                # ✗ Geral
"Qual é o significado da vida?"  # ✗ Geral
```

### Temperatura Adaptativa

```python
# Query técnica → Temperatura: 0.1 (determinístico)
# Query geral → Temperatura: 0.8 (criativo)

query = "O que é uma matriz?"
is_technical = agi.context_processor.detect_technical_query(query)
prompt, temperature = agi.context_processor.inject_technical_data(query, "dados técnicos")
print(f"Temperatura: {temperature}")  # 0.1
```

---

## Loop de Autocrítica

### Acionamento

```python
# Autocrítica é acionada automaticamente se:
# divergência_semântica > entropy_threshold (0.3)

# Exemplo:
response = "Uma matriz é um arranjo retangular de números..."
technical_data = "Matriz: estrutura de dados com linhas e colunas..."

entropy = agi.compute_semantic_divergence(response, technical_data)
print(f"Entropia: {entropy:.3f}")

if entropy > agi.entropy_threshold:
    print("Autocrítica será acionada!")
```

### Verificação Recursiva

```python
# O método verify_logic é chamado automaticamente no forward pass
# Até 3 iterações de re-processamento

corrected_response, was_corrected, correction_path = agi.verify_logic(
    response=response,
    technical_data=technical_data,
    tokens=token_tensor,
    expert_indices=expert_indices,
    tokenizer=tokenizer
)

if was_corrected:
    print(f"Resposta corrigida em {len(correction_path)} iterações")
    for step in correction_path:
        if 'entropy_before' in step:
            print(f"  Iter {step['iteration']}: {step['entropy_before']:.3f} → {step['entropy_after']:.3f}")
```

---

## Configuração Avançada

### Alterar Parâmetros

```python
# Entropy threshold
agi.entropy_threshold = 0.25  # Mais sensível à divergência

# Max iterações de autocrítica
agi.max_autocritique_iterations = 5

# Temperatura manual
agi.context_processor.temperature = 0.5
```

### Usar Dados Técnicos Customizados

```python
# Preparar dados técnicos
custom_data = """
Matriz: Uma estrutura de dados 2D com m linhas e n colunas.
Operações: Adição, multiplicação, transposição, inversão.
Aplicações: Sistemas lineares, transformações, gráficos computacionais.
"""

# Injetar no contexto
prompt, temp = agi.context_processor.inject_technical_data(
    query="O que é uma matriz?",
    technical_data=custom_data
)

print(prompt)
```

---

## Testes

### Testes de Integração

```bash
python3 test_integration.py
```

Executa 10 testes de componentes principais.

### Simulação de Autocrítica

```bash
python3 simulate_autocritique.py
```

Demonstra:
- Detecção de query técnica
- Cálculo de divergência semântica
- Forward pass completo

### Testes de Carga

```bash
python3 load_test.py
```

Executa 20 queries em 4 categorias:
- Queries técnicas (Matemática)
- Queries técnicas (Código)
- Queries gerais
- Queries mistas

---

## Troubleshooting

### Problema: "VectorRetinaV2 não encontra dados"

**Solução:** Criar pasta `digerido/` com arquivos `.txt`

```bash
mkdir -p digerido
echo "Dados técnicos sobre matrizes..." > digerido/matrices.txt
```

### Problema: "Resposta vazia ou corrupta"

**Solução:** Verificar tokenizador

```bash
python3 -c "from daemon_agi import OmegaTokenizer; t = OmegaTokenizer(); print(f'Vocab size: {len(t.vocab)}')"
```

### Problema: "Autocrítica não acionada"

**Solução:** Reduzir entropy_threshold

```python
agi.entropy_threshold = 0.2  # Mais sensível
```

### Problema: "Memória crescendo muito"

**Solução:** Limitar tamanho do MemoryBank

```python
agi.memory.max_size = 1000  # Padrão: 10000
```

---

## Performance Tips

1. **Use GPU:** `device="cuda"` para 10x mais rápido
2. **Batch Processing:** Processe múltiplas queries em paralelo
3. **Cache de Retina:** Reutilize `agi.retina` entre queries
4. **Reduzir Iterações:** `max_autocritique_iterations = 1` para latência baixa

---

## Integração com Sistemas Externos

### API REST (Futuro)

```python
from fastapi import FastAPI
from agi_core import AGICore

app = FastAPI()
agi = AGICore(...)

@app.post("/query")
async def query(text: str):
    result = agi.forward(text, "digerido", tokenizer)
    return result
```

### Webhook para Correções

```python
# Notificar sistema externo quando autocrítica é acionada
if result['was_corrected']:
    send_webhook({
        'query': query,
        'original_response': original_response,
        'corrected_response': result['response'],
        'entropy': result['entropy'],
        'correction_path': result['correction_path']
    })
```

---

## Licença e Atribuição

**OuroborosMoE** - Super Inteligência Generalista  
Desenvolvido por: Leogenes Simplício Rodrigues de Souza  
Versão: 8.0  
Data: 2026-04-19

---

**Última Atualização:** 2026-04-19  
**Status:** Produção
