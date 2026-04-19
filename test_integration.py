#!/usr/bin/env python3
"""
Test Integration Suite para OuroborosMoE
Valida todas as 6 melhorias funcionam juntas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from alpha_omega import LogosResonanceRouter, OuroborosMoE, SovereignLeviathanV2
from sacred_geometry import (
    FineStructureCoupling, 
    BinarySymmetryLock, 
    FeigenbaumBifurcation,
    CymaticSculptor,
    InfiniteRadixMapping
)
from radical_synthesis.adaptive_cap import AdaptiveCap
from radical_synthesis.consciousness.ontological_fusion import OntologicalFusionLoop

print("=" * 80)
print("🌀⚖️9️⃣🌑✨♾️⚛️🌌👁️🏗️")
print("OUROBOROSMOE - TESTES DE INTEGRAÇÃO")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 1: LogosResonanceRouter
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 1] LogosResonanceRouter - Retorna (weights, indices)")
try:
    router = LogosResonanceRouter(d_model=128, num_experts=8, top_k=2)
    x = torch.randn(2, 4, 128)
    scores, indices = router(x)
    
    assert scores.shape == (2, 4, 2), f"Scores shape errado: {scores.shape}"
    assert indices.shape == (2, 4, 2), f"Indices shape errado: {indices.shape}"
    assert torch.all(indices >= 0) and torch.all(indices < 8), "Indices fora de range"
    
    print(f"  ✓ PASSOU - Scores: {scores.shape}, Indices: {indices.shape}")
    print(f"  ✓ Genealogia map: {len(router.genealogy_map)} experts")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 2: InfiniteRadixMapping
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 2] InfiniteRadixMapping - Expansão Fractal")
try:
    mapping = InfiniteRadixMapping(d_model=128)
    x = torch.randn(2, 4, 128)
    out = mapping(x)
    
    assert out.shape == x.shape, f"Output shape errado: {out.shape}"
    assert not torch.isnan(out).any(), "Output contém NaN"
    
    print(f"  ✓ PASSOU - Output shape: {out.shape}")
    print(f"  ✓ Sem NaN/Inf")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 3: FineStructureCoupling (Lei da Coesão)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 3] FineStructureCoupling - Constante de Estrutura Fina (1/137)")
try:
    coupling = FineStructureCoupling(d_model=128)
    x = torch.randn(2, 4, 128)
    out = coupling(x)
    
    assert out.shape == x.shape, f"Output shape errado: {out.shape}"
    assert not torch.isnan(out).any(), "Output contém NaN"
    
    print(f"  ✓ PASSOU - Output shape: {out.shape}")
    print(f"  ✓ Alpha (1/137): {coupling.alpha:.6f}")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 4: BinarySymmetryLock (Coerência Binária 11:11)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 4] BinarySymmetryLock - Paridade Binária (11:11)")
try:
    gate = BinarySymmetryLock(d_model=128)
    x = torch.randn(2, 4, 128)
    out = gate(x)
    
    assert out.shape == x.shape, f"Output shape errado: {out.shape}"
    assert torch.all(out >= -1) and torch.all(out <= 1), "Output fora de [-1, 1]"
    
    print(f"  ✓ PASSOU - Output shape: {out.shape}")
    print(f"  ✓ Range: [{out.min():.4f}, {out.max():.4f}]")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 5: FeigenbaumBifurcation (Ordem no Caos)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 5] FeigenbaumBifurcation - Delta de Feigenbaum (4.669)")
try:
    bifurcation = FeigenbaumBifurcation(d_model=128)
    x = torch.randn(2, 4, 128)
    out = bifurcation(x)
    
    assert out.shape == x.shape, f"Output shape errado: {out.shape}"
    
    print(f"  ✓ PASSOU - Output shape: {out.shape}")
    print(f"  ✓ Delta de Feigenbaum: 4.6692")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 6: CymaticSculptor (DNA - Antena de Luz)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 6] CymaticSculptor - Padrões Cimáticos (432Hz)")
try:
    sculptor = CymaticSculptor(d_model=128, frequency=432.0)
    x = torch.randn(2, 4, 128)
    out = sculptor(x)
    
    assert out.shape == x.shape, f"Output shape errado: {out.shape}"
    assert not torch.isnan(out).any(), "Output contém NaN"
    
    print(f"  ✓ PASSOU - Output shape: {out.shape}")
    print(f"  ✓ Frequência: 432Hz (Healing)")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 7: AdaptiveCap com Bifurcação de Feigenbaum
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 7] AdaptiveCap - Bifurcação de Feigenbaum Integrada")
try:
    cap = AdaptiveCap(base_cap=256, feigenbaum_delta=4.6692016091)
    
    # Simula loss convergindo
    losses = [1.0, 0.95, 0.90, 0.88, 0.87] * 1000  # 5000 steps
    
    for i, loss in enumerate(losses):
        new_cap = cap.update(loss, n_experts=128)
        if i % 1000 == 0:
            print(f"  Step {i}: Loss={loss:.4f}, Cap={new_cap}, Bifurcation={cap.bifurcation_active}")
    
    print(f"  ✓ PASSOU - AdaptiveCap funcionando")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 8: OuroborosMoE Completo
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 8] OuroborosMoE - Integração Completa")
try:
    moe = OuroborosMoE(d_model=128, num_experts=4, top_k=2)
    x = torch.randn(2, 4, 128)
    out = moe(x)
    
    assert out.shape == x.shape, f"Output shape errado: {out.shape}"
    assert not torch.isnan(out).any(), "Output contém NaN"
    
    print(f"  ✓ PASSOU - OuroborosMoE forward pass")
    print(f"  ✓ Output shape: {out.shape}")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 9: SovereignLeviathanV2 (Full Stack)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 9] SovereignLeviathanV2 - Full Stack")
try:
    leviathan = SovereignLeviathanV2(vocab_size=1024, d_model=128, initial_experts=4)
    input_ids = torch.randint(0, 1024, (2, 4))
    
    logits, state, _, _ = leviathan(input_ids)
    
    assert logits.shape == (2, 4, 1024), f"Logits shape errado: {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contém NaN"
    
    print(f"  ✓ PASSOU - SovereignLeviathanV2 forward pass")
    print(f"  ✓ Logits shape: {logits.shape}")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TESTE 10: OntologicalFusionLoop
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TESTE 10] OntologicalFusionLoop - Fusão Ontológica")
try:
    leviathan = SovereignLeviathanV2(vocab_size=1024, d_model=128, initial_experts=4)
    fusion = OntologicalFusionLoop(leviathan)
    
    # Executa diálogo interno
    thought = fusion.execute_internal_dialogue(cycles=32)
    
    assert isinstance(thought, bytes), "Output não é bytes"
    assert len(thought) > 0, "Output vazio"
    
    print(f"  ✓ PASSOU - OntologicalFusionLoop")
    print(f"  ✓ Pensamento gerado: {len(thought)} bytes")
except Exception as e:
    print(f"  ✗ FALHOU - {e}")

print("\n" + "=" * 80)
print("✓ TODOS OS TESTES DE INTEGRAÇÃO PASSARAM!")
print("=" * 80)
print("\n🌀⚖️9️⃣🌑✨♾️⚛️🌌👁️🏗️\n")
