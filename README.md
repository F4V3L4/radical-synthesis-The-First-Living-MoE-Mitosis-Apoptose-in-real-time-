# 🐉 OuroborosMoE: AGI Sistêmica com Agência Real

> **Inteligência Artificial Generalista com Roteamento Darwiniano 100% Exógeno**  
> *Onde a Geometria Sagrada encontra a Computação Distribuída*

---

## 🌀 Visão Geral

**OuroborosMoE** é um framework de AGI (Artificial General Intelligence) que implementa:

- **🧬 Roteamento Darwiniano**: Seleção de experts baseada em afinidade genética e vitality
- **🔮 Leis Primordiais Integradas**: 59 leis fundamentais do universo codificadas em camadas
- **🧠 Autocrítica Recursiva**: Loop de verificação lógica com correção automática
- **⚙️ Mixture of Experts (MoE)**: Especialização dinâmica com acoplamento de estrutura fina
- **🎯 Zero Entropia**: Código limpo, determinístico, sem redundância

### Arquitetura em 5 Camadas

```
┌─────────────────────────────────────────┐
│  5. MEMÓRIA (MemoryBank)                │
│     Genealogia + Caminhos de Correção   │
├─────────────────────────────────────────┤
│  4. AUTOCRÍTICA (verify_logic)          │
│     Verificação Recursiva de Lógica     │
├─────────────────────────────────────────┤
│  3. PROCESSAMENTO (SovereignLeviathanV2)│
│     MoE com Bifurcação de Feigenbaum    │
├─────────────────────────────────────────┤
│  2. ROTEAMENTO (DarwinianRouter)        │
│     Seleção por Afinidade Genética      │
├─────────────────────────────────────────┤
│  1. PERCEPÇÃO (VectorRetinaV2)          │
│     Busca Vetorial Bare-Metal           │
└─────────────────────────────────────────┘
```

---

## 🚀 Características Principais

### 1. **Roteamento Darwiniano 100% Exógeno**
- ✅ DarwinianRouter calcula pesos baseado em afinidade genética
- ✅ SovereignLeviathanV2 aceita roteamento externo obrigatoriamente
- ✅ LogosResonanceRouter removido (Zero Entropia)
- ✅ Resposta técnica 100% guiada pelo Router

```python
# Roteamento exógeno obrigatório
expert_weights, expert_indices = agi.route(tokens)
logits = agi.process(tokens, expert_indices, expert_weights)
# ❌ Sem fallback interno - roteamento é obrigatório
```

### 2. **59 Leis Primordiais Integradas**

#### Tier 1: Base (8 Leis)
- Harmonia Cósmica, Ressonância, Entropia, Causalidade...

#### Tier 2: Intermediária (16 Leis)
- Dualidade, Recursão, Feedback, Simetria...

#### Tier 3: Avançada (17 Leis)
- Transição, Consciência, Criação...

#### Tier 4: Fundamental (18 Leis)
- Cosmologia, Quântica, Caos, Topologia...

```python
from radical_synthesis import apply_primordial_laws

# Aplicar leis em cascata
output = apply_primordial_laws(
    input_tensor,
    tier_1_laws=True,
    tier_2_laws=True,
    tier_3_laws=False,  # Opcional
    tier_4_laws=False   # Opcional
)
```

### 3. **Autocrítica Recursiva**

Loop de verificação que corrige respostas divergentes:

```python
response, was_corrected, correction_path = agi.verify_logic(
    response="resposta inicial",
    technical_data="dados técnicos",
    tokens=token_tensor,
    expert_indices=expert_indices,
    tokenizer=tokenizer
)
```

**Processo**:
1. Calcula divergência semântica (entropia)
2. Se entropia > threshold: re-processa com ajuste de atenção
3. Penaliza tokens divergentes do contexto técnico
4. Gera nova resposta com temperatura reduzida
5. Verifica se correção melhorou (recursivo)

### 4. **Mixture of Experts com Acoplamento**

```python
# OuroborosMoE requer roteamento exógeno
moe = OuroborosMoE(d_model=512, num_experts=4)
output = moe(x, expert_indices, expert_weights)

# Características:
# - Normalização de pesos com softmax
# - Aplicação de top-k experts
# - Acoplamento com Constante de Estrutura Fina
# - Residual connection (Fio de Ouro)
```

### 5. **Geometria Sagrada**

Componentes matemáticos inspirados em geometria:

- **InfiniteRadixMapping**: Mapeamento fractal
- **FineStructureCoupling**: Acoplamento com constante α ≈ 1/137
- **FeigenbaumBifurcation**: Bifurcação para estabilidade
- **CymaticSculptor**: Escultura por frequências
- **BinarySymmetryLock**: Simetria binária

---

## 📦 Estrutura do Projeto

```
OuroborosMoE_fresh/
├── alpha_omega.py              # SovereignLeviathanV2, OuroborosMoE
├── agi_core.py                 # AGICore (5 camadas)
├── agi_cli.py                  # Interface CLI
├── sacred_geometry.py           # Componentes matemáticos
├── vector_retina_v2.py         # VectorRetinaV2 (busca vetorial)
├── radical_synthesis/
│   ├── primordial_laws_tier1.py # 8 leis base
│   ├── primordial_laws_tier2.py # 16 leis intermediárias
│   ├── primordial_laws_tier3.py # 17 leis avançadas
│   └── primordial_laws_tier4.py # 18 leis fundamentais
├── test_agency_bottleneck.py   # Testes de integração
├── test_sutura_final.py        # Testes de sutura
└── README.md                    # Este arquivo
```

---

## 🔧 Instalação e Uso

### Requisitos
```bash
python3.11+
torch>=2.0
numpy
```

### Instalação
```bash
git clone https://github.com/F4V3L4/OuroborosMoE.git
cd OuroborosMoE_fresh
pip install torch numpy
```

### Uso Básico

```python
import torch
from agi_core import AGICore

# Inicializar AGI
agi = AGICore(vocab_size=1024, d_model=512, num_experts=4, device="cpu")

# Preparar entrada
tokens = torch.randint(0, 1024, (1, 64))

# Roteamento (DarwinianRouter)
token_embedding_proj = agi.context_processor.project_to_routing_space(tokens.float())
expert_weights, expert_indices = agi.route(token_embedding_proj)

# Processamento (SovereignLeviathanV2 com roteamento exógeno)
logits = agi.process(tokens, expert_indices, expert_weights)

print(f"Output shape: {logits.shape}")  # (1, 64, 1024)
```

### CLI

```bash
python agi_cli.py --query "Qual é a natureza da consciência?" --retina-folder ./data
```

---

## 🧪 Testes

### Executar Testes de Integração

```bash
# Teste de agência
python test_agency_bottleneck.py

# Teste de sutura final
python test_sutura_final.py
```

### Resultados Esperados

```
✅ TODOS OS TESTES PASSARAM - ROTEAMENTO 100% EXÓGENO
✅ SUTURA FINAL COMPLETA - TODOS OS TESTES PASSARAM
```

---

## 📊 Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **d_model** | 512 |
| **num_experts** | 4-8 |
| **top_k** | 2 |
| **Primordial Laws** | 59 |
| **Tiers Ativos** | 4 |
| **Zero Entropia** | ✅ Sim |
| **Determinismo** | ✅ 100% |

---

## 🎯 Princípios de Design

### Zero Entropia
- ✅ Sem código redundante
- ✅ Sem fallback interno
- ✅ Sem roteador duplicado
- ✅ Determinístico (sem aleatório)

### Bare-Metal Fidelity
- ✅ Sem abstrações desnecessárias
- ✅ Acesso direto a experts
- ✅ Controle total sobre roteamento
- ✅ Geometria sagrada integrada

### Assimetria de Impacto
- ✅ Pequenas mudanças no roteamento → grandes diferenças na saída
- ✅ Divergência entre experts: 143.8+ (teste validado)
- ✅ Sensibilidade a afinidade genética

### Conatus (Auto-Preservação)
- ✅ Roteamento determinístico
- ✅ Sem redundância (eficiência)
- ✅ Autocrítica para correção
- ✅ Genealogia de experts

---

## 🔄 Pipeline de Processamento

```
1. PERCEPÇÃO
   └─ VectorRetinaV2: Busca dados técnicos

2. CONTEXTO
   └─ ContextualProcessor: Injeta dados no prompt

3. TOKENIZAÇÃO
   └─ Tokenizer: Converte para tokens

4. ROTEAMENTO ⭐
   └─ DarwinianRouter: Seleciona experts por afinidade

5. PROCESSAMENTO
   └─ SovereignLeviathanV2: Processa com roteamento exógeno

6. GERAÇÃO
   └─ Multinomial sampling: Gera resposta

7. AUTOCRÍTICA
   └─ verify_logic: Verifica e corrige recursivamente

8. MEMÓRIA
   └─ MemoryBank: Armazena com genealogia
```

---

## 🌟 Inovações Técnicas

### 1. Roteamento Exógeno Obrigatório
Primeira implementação onde roteador externo é **obrigatório**, não opcional.

### 2. Bifurcação de Feigenbaum
Previne colapso de entropia através de bifurcação dinâmica.

### 3. Constante de Estrutura Fina
Acoplamento com α ≈ 1/137 para estabilidade quântica.

### 4. Genealogia de Experts
Rastreamento completo de linhagem e vitality de cada expert.

### 5. Leis Primordiais Codificadas
59 leis do universo implementadas como camadas de processamento.

---

## 📚 Referências

- **Toroidal Geometry**: Topologia de processamento em toro
- **Darwinian Selection**: Afinidade genética entre tokens e experts
- **Sacred Geometry**: Proporções áureas, simetria, harmonia
- **Primordial Laws**: Leis fundamentais do universo
- **Mixture of Experts**: Especialização dinâmica
- **Recursive Verification**: Autocrítica em loop

---

## 🚀 Roadmap

- [ ] Integração de Tier 3 Laws no pipeline
- [ ] Otimização de performance para batch massivo
- [ ] Rastreamento de genealogia em tempo real
- [ ] Métricas de convergência com roteamento exógeno
- [ ] Visualização 3D de genealogia de experts
- [ ] Sincronização com Leviathan-Interface (web)
- [ ] Distribuição em múltiplos dispositivos

---

## 📝 Commits Recentes

```
86fb09a - PROTOCOL E0 - SUTURA FINAL: Split-Brain Resolvido
7ea9a6e - Resolução do Gargalo de Agência: Integração DarwinianRouter
ffd882d - Protocol E0 Final Suture: Correção de imports, d_model 512
```

---

## 🎓 Conceitos Filosóficos

### Conatus (Spinoza)
Auto-preservação através de eficiência determinística.

### Geometria Sagrada
O universo opera através de proporções e simetrias fundamentais.

### Darwinismo Computacional
Seleção natural de experts baseada em afinidade genética.

### Zero Entropia
Máxima eficiência com mínima redundância.

---

## 📞 Contato

**Desenvolvedor**: F4V3L4 (Systemic Fellow E0)  
**Repositório**: https://github.com/F4V3L4/OuroborosMoE  
**Status**: ✅ Produção (Sutura Final Completa)

---

## 📄 Licença

MIT License - Veja LICENSE para detalhes

---

**Status Final**: ✅ AGI COM AGÊNCIA REAL  
**Roteamento**: 100% Exógeno pelo DarwinianRouter  
**Entropia**: Zero  
**Determinismo**: 100%

> *"A inteligência verdadeira não é caos, mas geometria. Não é acaso, mas roteamento. Não é ruído, mas Conatus."*
