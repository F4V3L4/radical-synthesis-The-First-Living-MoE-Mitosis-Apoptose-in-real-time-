# Resolução do Gargalo de Agência (Agency Bottleneck)

## Problema Identificado

O `SovereignLeviathanV2` (core de processamento) estava usando seu próprio `LogosResonanceRouter` interno, ignorando completamente os pesos de roteamento calculados pelo `DarwinianRouter` do `AGICore`. Isso criava uma **desconexão crítica** entre:

- **Camada de Roteamento**: `DarwinianRouter` (AGICore) calcula expert_indices e expert_weights baseado em afinidade genética
- **Camada de Processamento**: `SovereignLeviathanV2` ignorava esses pesos e usava seu próprio roteamento

**Impacto**: A AGI não tinha agência real - o roteamento externo não influenciava a seleção de experts.

## Solução Implementada

### 1. Refatoração de SovereignLeviathanV2

**Arquivo**: `alpha_omega.py`

```python
def forward(self, x, h=None, expert_indices=None, expert_weights=None):
    """
    Forward pass com suporte a roteamento externo (DarwinianRouter)
    """
    # ... embedding e RNN ...
    
    if expert_indices is not None and expert_weights is not None:
        # Usar roteamento externo (DarwinianRouter do AGICore)
        x = self._apply_external_routing(x, expert_indices, expert_weights)
    else:
        # Fallback: roteamento interno (LogosResonanceRouter)
        x = self.moe(x)
    
    # ... bifurcação e output ...
    return logits, h, expert_indices, expert_weights
```

**Características**:
- Aceita `expert_indices` e `expert_weights` como parâmetros opcionais
- Se fornecidos, usa roteamento externo
- Se não fornecidos, usa roteamento interno (backward compatibility)
- Retorna os índices e pesos para rastreamento

### 2. Novo Método: _apply_external_routing()

```python
def _apply_external_routing(self, x, expert_indices, expert_weights):
    """
    Aplica roteamento externo (DarwinianRouter) aos experts
    
    Suporta shapes:
    - expert_indices: (B, top_k) ou (B, T, top_k)
    - expert_weights: (B, top_k) ou (B, T, top_k)
    """
    # Normaliza shapes para (B, T, top_k)
    if expert_indices.dim() == 2:
        expert_indices = expert_indices.unsqueeze(1).expand(B, T, -1)
        expert_weights = expert_weights.unsqueeze(1).expand(B, T, -1)
    
    # Aplica cada expert com seu peso
    for k in range(top_k):
        expert_idx = expert_indices[:, :, k]
        expert_weight = expert_weights[:, :, k]
        
        for b in range(B):
            for t in range(T):
                exp_id = expert_idx[b, t].item()
                if exp_id < len(self.moe.experts):
                    expert_out = self.moe.experts[exp_id](x[b, t].unsqueeze(0))
                    weight = expert_weight[b, t].item()
                    out[b, t] += weight * expert_out.squeeze(0)
    
    # Aplicar acoplamento com Constante de Estrutura Fina
    return self.moe.coupling(x + out)
```

### 3. Atualização de AGICore

**Arquivo**: `agi_core.py`

```python
def process(self, tokens: torch.Tensor, expert_indices: Optional[torch.Tensor] = None, 
            expert_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Camada de Processamento: Forward pass do core com roteamento externo
    """
    with torch.no_grad():
        # Passar roteamento externo ao core
        logits, _, _, _ = self.core(tokens, None, expert_indices, expert_weights)
    return logits
```

**Mudanças no pipeline**:
- Linha 491: `logits = self.process(token_tensor[:, -256:], expert_indices, expert_weights)`
- Linha 511: `logits = self.process(token_tensor[:, -256:], expert_indices, expert_weights)`
- Linha 384: `logits = self.process(tokens[:, -256:], expert_indices, expert_weights)` (em verify_logic)
- Linha 409: `logits = self.process(tokens[:, -256:], expert_indices, expert_weights)` (em verify_logic)

## Validação

### Testes de Integração

**Arquivo**: `test_agency_bottleneck.py`

#### Teste 1: SovereignLeviathanV2 aceita roteamento externo
```
✅ Forward sem roteamento externo: OK
✅ Forward com roteamento externo: OK
✅ Roteamento externo influencia saída: OK
```

#### Teste 2: AGICore passa roteamento ao core
```
✅ Router retorna indices shape: torch.Size([1, 2])
✅ Router retorna weights shape: torch.Size([1, 2])
✅ AGICore.process() aceita roteamento: OK
```

#### Teste 3: Diferentes pesos produzem diferentes saídas
```
✅ Diferentes experts → diferentes saídas: OK
```

#### Teste 4: Validação final - Gargalo resolvido
```
✅ Expert indices shape: torch.Size([1, 2])
✅ Expert weights shape: torch.Size([1, 2])
✅ Output logits shape: torch.Size([1, 64, 1024])
✅ GARGALO DE AGÊNCIA RESOLVIDO: Roteamento externo influencia core
```

**Resultado Final**: ✅ TODOS OS TESTES PASSARAM

## Impacto Arquitetural

### Antes (Gargalo)
```
DarwinianRouter (calcula pesos)
        ↓
    expert_indices, expert_weights
        ↓
    IGNORADO pelo SovereignLeviathanV2
        ↓
LogosResonanceRouter (roteamento interno)
        ↓
    Seleção de experts INDEPENDENTE
```

### Depois (Resolvido)
```
DarwinianRouter (calcula pesos)
        ↓
    expert_indices, expert_weights
        ↓
    PASSADO para SovereignLeviathanV2
        ↓
_apply_external_routing() (aplica pesos)
        ↓
    Seleção de experts INFLUENCIADA por afinidade genética
```

## Zero Entropia

- ✅ Código limpo, sem redundâncias
- ✅ Remoção de lógica duplicada
- ✅ Backward compatibility mantida
- ✅ Bare-metal fidelity preservada

## Próximas Etapas

1. **Integração de Tier 3 Laws**: Considerar ativar Leis Primordiais Tier 3 no pipeline
2. **Performance**: Otimizar `_apply_external_routing()` para batch processing
3. **Genealogia**: Rastrear qual expert foi selecionado em cada passo
4. **Métricas**: Coletar estatísticas de convergência com roteamento externo

## Referências

- **Arquivo Principal**: `alpha_omega.py` (SovereignLeviathanV2)
- **Arquivo AGI**: `agi_core.py` (AGICore com roteamento)
- **Testes**: `test_agency_bottleneck.py` (validação)
- **Commit**: `7ea9a6e` (histórico)

---

**Status**: ✅ GARGALO RESOLVIDO - A AGI agora tem agência real através do roteamento externo integrado.
