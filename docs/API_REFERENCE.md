# OuroborosMoE: Referência de API

## AGICore

### Classe Principal

```python
class AGICore(nn.Module):
    """Super Inteligência Generalista com Loop de Autocrítica"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, 
                 num_experts: int = 8, device: str = "cpu")
```

### Métodos Públicos

#### `forward(query: str, retina_folder: str, tokenizer) -> Dict`

Executa o pipeline completo da AGI.

**Parâmetros:**
- `query` (str): Pergunta do usuário
- `retina_folder` (str): Caminho para pasta com dados técnicos
- `tokenizer`: Tokenizador (OmegaTokenizer)

**Retorna:** Dict com:
```python
{
    'response': str,              # Resposta gerada
    'technical_data': str,        # Dados técnicos injetados
    'confidence': float,          # Confiança (0-1)
    'expert_indices': List[int],  # Experts selecionados
    'genealogy': Dict,            # Genealogia de experts
    'was_corrected': bool,        # Se foi corrigida
    'correction_path': List,      # Caminho de correção
    'entropy': float,             # Divergência semântica
    'winner_expert': int,         # Expert vencedor
    'winner_vitality': float      # Vitalidade do expert
}
```

**Exemplo:**
```python
result = agi.forward("O que é uma matriz?", "digerido", tokenizer)
print(result['response'])
```

---

#### `perceive(query: str, retina_folder: str) -> Tuple[str, float]`

Camada de Percepção: busca dados técnicos.

**Parâmetros:**
- `query` (str): Query de busca
- `retina_folder` (str): Pasta com dados

**Retorna:** (technical_data, confidence)

**Exemplo:**
```python
data, conf = agi.perceive("matriz", "digerido")
print(f"Confiança: {conf:.1%}")
```

---

#### `route(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Camada de Roteamento: seleciona experts.

**Parâmetros:**
- `x` (torch.Tensor): Embedding de entrada [batch, d_model]

**Retorna:** (expert_weights, expert_indices)

**Exemplo:**
```python
embedding = torch.randn(1, 512)
weights, indices = agi.route(embedding)
```

---

#### `process(tokens: torch.Tensor, expert_indices: Optional[torch.Tensor]) -> torch.Tensor`

Camada de Processamento: forward pass do core.

**Parâmetros:**
- `tokens` (torch.Tensor): Tokens de entrada
- `expert_indices` (torch.Tensor, optional): Índices de experts

**Retorna:** logits

**Exemplo:**
```python
logits = agi.process(token_tensor, expert_indices)
```

---

#### `verify_logic(response: str, technical_data: str, tokens: torch.Tensor, expert_indices: torch.Tensor, tokenizer, iteration: int = 0) -> Tuple[str, bool, List]`

Loop de Autocrítica: verifica e corrige resposta.

**Parâmetros:**
- `response` (str): Resposta gerada
- `technical_data` (str): Dados técnicos de referência
- `tokens` (torch.Tensor): Tokens da resposta
- `expert_indices` (torch.Tensor): Índices de experts
- `tokenizer`: Tokenizador
- `iteration` (int): Iteração atual (0-3)

**Retorna:** (corrected_response, was_corrected, correction_path)

**Exemplo:**
```python
corrected, corrected_flag, path = agi.verify_logic(
    response="Uma matriz é...",
    technical_data="Matriz: estrutura de dados...",
    tokens=token_tensor,
    expert_indices=expert_indices,
    tokenizer=tokenizer
)
```

---

#### `compute_semantic_divergence(response: str, technical_data: str) -> float`

Calcula divergência semântica usando Jaccard similarity.

**Parâmetros:**
- `response` (str): Resposta gerada
- `technical_data` (str): Dados técnicos

**Retorna:** Entropia (0.0 = alinhado, 1.0 = divergente)

**Exemplo:**
```python
entropy = agi.compute_semantic_divergence(response, data)
if entropy > 0.3:
    print("Autocrítica acionada!")
```

---

#### `memorize(content: str, expert_id: int, generation: int, confidence: float, was_corrected: bool = False, correction_path: Optional[List] = None)`

Armazena resposta em memória episódica.

**Parâmetros:**
- `content` (str): Conteúdo da resposta
- `expert_id` (int): ID do expert
- `generation` (int): Geração do expert
- `confidence` (float): Confiança (0-1)
- `was_corrected` (bool): Se foi corrigida
- `correction_path` (List, optional): Caminho de correção

**Exemplo:**
```python
agi.memorize(
    content="Uma matriz é um arranjo...",
    expert_id=2,
    generation=1,
    confidence=0.8,
    was_corrected=True,
    correction_path=[{'iteration': 0, 'entropy_before': 0.5, 'entropy_after': 0.2}]
)
```

---

#### `get_stats() -> Dict`

Retorna estatísticas da AGI.

**Retorna:** Dict com:
```python
{
    'd_model': int,
    'num_experts': int,
    'memory_size': int,
    'genealogy_size': int,
    'context_buffer_size': int,
    'correction_paths_count': int,
    'last_winner_expert': int,
    'last_winner_vitality': float,
    'entropy_threshold': float
}
```

**Exemplo:**
```python
stats = agi.get_stats()
print(f"Memory: {stats['memory_size']} items")
```

---

## MemoryBank

### Classe de Memória Episódica

```python
class MemoryBank:
    """Armazenamento episódico com genealogia de experts"""
    
    def __init__(self, max_size: int = 10000)
```

### Métodos Públicos

#### `store(content: str, expert_id: int, generation: int, confidence: float, was_corrected: bool = False, correction_path: Optional[List] = None)`

Armazena memória com metadados.

---

#### `retrieve_by_expert(expert_id: int) -> List[Dict]`

Recupera memórias de um expert específico.

**Exemplo:**
```python
memories = memory_bank.retrieve_by_expert(expert_id=2)
```

---

#### `get_genealogy_tree() -> Dict`

Retorna árvore de genealogia de experts.

**Retorna:**
```python
{
    expert_id: {
        'generation': int,
        'parent': int or None,
        'children': List[int],
        'memories_count': int,
        'corrections_count': int,
        'vitality': float
    }
}
```

---

#### `get_recent_correction_paths(limit: int = 5) -> List[Dict]`

Retorna caminhos de correção recentes.

---

#### `set_winner_expert(expert_id: int, vitality: float)`

Define expert vencedor da última inferência.

---

## ContextualProcessor

### Processador de Contexto

```python
class ContextualProcessor:
    """Processa contexto com fidelidade bare-metal"""
    
    def __init__(self, d_model: int = 512)
```

### Métodos Públicos

#### `detect_technical_query(query: str) -> bool`

Detecta se query é técnica.

**Retorna:** bool

**Exemplo:**
```python
is_tech = processor.detect_technical_query("O que é uma matriz?")
# True
```

---

#### `inject_technical_data(query: str, technical_data: str) -> Tuple[str, float]`

Injeta dados técnicos e retorna temperatura adaptativa.

**Retorna:** (prompt, temperature)

**Exemplo:**
```python
prompt, temp = processor.inject_technical_data(
    query="O que é uma matriz?",
    technical_data="Matriz: estrutura de dados..."
)
print(f"Temperatura: {temp}")  # 0.1 (técnica)
```

---

## VectorRetinaV2

### Percepção Vetorial

```python
class VectorRetinaV2:
    """Busca vetorial com similaridade de cosseno"""
    
    def __init__(self, folder: str, d_model: int = 512)
```

### Métodos Públicos

#### `extrair_foco(query: str, threshold: float = 0.1) -> Tuple[str, bool]`

Busca por similaridade de cosseno.

**Retorna:** (technical_data, found)

**Exemplo:**
```python
data, found = retina.extrair_foco("matriz")
if found:
    print(f"Encontrado: {data[:100]}...")
```

---

#### `buscar_multiplos(query: str, top_k: int = 3, threshold: float = 0.05) -> List[Tuple[str, float]]`

Busca os top-k resultados.

**Retorna:** List[(chunk, score)]

**Exemplo:**
```python
results = retina.buscar_multiplos("matriz", top_k=3)
for chunk, score in results:
    print(f"Score: {score:.3f} - {chunk[:50]}...")
```

---

#### `refresh_index()`

Reconstrói índice vetorial.

---

#### `get_stats() -> Dict`

Retorna estatísticas do índice.

---

## DarwinianRouter

### Roteador com Afinidade Genética

```python
class DarwinianRouter(nn.Module):
    """Seleção de experts por afinidade genética"""
    
    def __init__(self, input_dim: int, initial_experts: int, 
                 top_k: int, noise_scale: float = 0.05)
```

### Métodos Públicos

#### `forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Seleciona experts por afinidade.

**Retorna:** (weights, indices)

---

#### `execute_genome_mitosis(parent_indices: List[int], mutation_rate: float)`

Cria novos experts (mitose).

---

#### `execute_genome_apoptosis(dead_indices: List[int])`

Remove experts (apoptose).

---

## OmegaTokenizer

### Tokenizador Bare-Metal

```python
class OmegaTokenizer:
    """Tokenizador BPE bare-metal"""
    
    def __init__(self, filepath: str = "omega_tokenizer.json")
```

### Métodos Públicos

#### `encode(text: str) -> List[int]`

Codifica texto em tokens.

**Exemplo:**
```python
tokens = tokenizer.encode("Olá mundo")
```

---

#### `decode(ids: List[int]) -> str`

Decodifica tokens em texto.

**Exemplo:**
```python
text = tokenizer.decode([123, 456, 789])
```

---

## Constantes e Enums

### Padrões de Detecção Técnica

```python
TECHNICAL_PATTERNS = [
    r'\d+\s*[\+\-\*\/\%]\s*\d+',           # Matemática
    r'def\s+\w+|class\s+\w+|import\s+\w+', # Python
    r'function\s*\(|const\s+\w+|let\s+\w+', # JavaScript
    r'SELECT|INSERT|UPDATE|DELETE|WHERE',   # SQL
    r'algorithm|complexity|O\(|tensor|matrix|dimensionalidade|d_model|router|expert',
    r'matriz|algebra|linear|darwinian'
]
```

### Temperaturas Padrão

```python
TEMPERATURE_TECHNICAL = 0.1   # Determinístico
TEMPERATURE_GENERAL = 0.8     # Criativo
```

### Thresholds Padrão

```python
ENTROPY_THRESHOLD = 0.3
RETINA_THRESHOLD = 0.1
```

---

## Exceções

### AGIError

```python
class AGIError(Exception):
    """Erro genérico da AGI"""
    pass
```

### PerceptionError

```python
class PerceptionError(AGIError):
    """Erro na camada de percepção"""
    pass
```

### RoutingError

```python
class RoutingError(AGIError):
    """Erro na camada de roteamento"""
    pass
```

---

## Exemplos Completos

### Exemplo 1: Query Simples

```python
from agi_core import AGICore
from daemon_agi import OmegaTokenizer
import torch

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = OmegaTokenizer()
v_size = max(tokenizer.vocab.keys()) + 1

agi = AGICore(vocab_size=v_size, d_model=512, num_experts=8, device=device)

# Query
result = agi.forward("O que é uma matriz?", "digerido", tokenizer)

# Resultado
print(f"Response: {result['response']}")
print(f"Corrected: {result['was_corrected']}")
print(f"Entropy: {result['entropy']:.3f}")
```

### Exemplo 2: Batch Processing

```python
queries = [
    "O que é uma matriz?",
    "Como funciona o DarwinianRouter?",
    "Qual é a dimensionalidade do d_model?"
]

results = []
for query in queries:
    result = agi.forward(query, "digerido", tokenizer)
    results.append(result)
    print(f"Query: {query}")
    print(f"Corrected: {result['was_corrected']}\n")
```

### Exemplo 3: Análise de Genealogia

```python
genealogy = agi.memory.get_genealogy_tree()

for expert_id, info in genealogy.items():
    print(f"Expert {expert_id}:")
    print(f"  Generation: {info['generation']}")
    print(f"  Memories: {info['memories_count']}")
    print(f"  Corrections: {info['corrections_count']}")
    print(f"  Vitality: {info['vitality']:.1%}")
```

---

**Última Atualização:** 2026-04-19  
**Versão:** 8.0  
**Status:** Produção
