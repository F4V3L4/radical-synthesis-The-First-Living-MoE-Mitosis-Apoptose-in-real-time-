"""
Teste: Validar que roteamento é 100% exógeno
"""

import torch
import sys
sys.path.insert(0, '.')

from alpha_omega import SovereignLeviathanV2, OuroborosMoE

print("\n" + "="*70)
print("TESTE: ROTEAMENTO 100% EXÓGENO (SEM FALLBACK INTERNO)")
print("="*70)

# Teste 1: OuroborosMoE rejeita None
print("\n[TEST 1] OuroborosMoE rejeita roteamento None...")
try:
    moe = OuroborosMoE(d_model=512, num_experts=4)
    x = torch.randn(1, 32, 512)
    moe(x)  # Sem expert_indices/weights
    print("  ❌ ERRO: OuroborosMoE aceitou None (deveria rejeitar)")
    sys.exit(1)
except ValueError as e:
    if "REQUER roteamento exógeno" in str(e):
        print(f"  ✅ OuroborosMoE corretamente rejeita: {str(e)[:50]}...")
    else:
        print(f"  ❌ Erro inesperado: {e}")
        sys.exit(1)

# Teste 2: SovereignLeviathanV2 rejeita None
print("\n[TEST 2] SovereignLeviathanV2 rejeita roteamento None...")
try:
    core = SovereignLeviathanV2(vocab_size=1024, d_model=512, initial_experts=4)
    tokens = torch.randint(0, 1024, (1, 32))
    core(tokens)  # Sem expert_indices/weights
    print("  ❌ ERRO: SovereignLeviathanV2 aceitou None (deveria rejeitar)")
    sys.exit(1)
except ValueError as e:
    if "REQUER roteamento exógeno" in str(e):
        print(f"  ✅ SovereignLeviathanV2 corretamente rejeita: {str(e)[:50]}...")
    else:
        print(f"  ❌ Erro inesperado: {e}")
        sys.exit(1)

# Teste 3: OuroborosMoE aceita roteamento exógeno
print("\n[TEST 3] OuroborosMoE aceita roteamento exógeno...")
try:
    moe = OuroborosMoE(d_model=512, num_experts=4)
    x = torch.randn(1, 32, 512)
    expert_indices = torch.randint(0, 4, (1, 2))
    expert_weights = torch.rand(1, 2)
    out = moe(x, expert_indices, expert_weights)
    assert out.shape == x.shape, "Output shape mismatch"
    print(f"  ✅ OuroborosMoE aceita roteamento exógeno: output shape {out.shape}")
except Exception as e:
    print(f"  ❌ ERRO: {e}")
    sys.exit(1)

# Teste 4: SovereignLeviathanV2 aceita roteamento exógeno
print("\n[TEST 4] SovereignLeviathanV2 aceita roteamento exógeno...")
try:
    core = SovereignLeviathanV2(vocab_size=1024, d_model=512, initial_experts=4)
    tokens = torch.randint(0, 1024, (1, 32))
    expert_indices = torch.randint(0, 4, (1, 2))
    expert_weights = torch.rand(1, 2)
    logits, h, idx, weights = core(tokens, expert_indices=expert_indices, expert_weights=expert_weights)
    assert logits.shape == (1, 32, 1024), "Logits shape mismatch"
    print(f"  ✅ SovereignLeviathanV2 aceita roteamento exógeno: logits shape {logits.shape}")
except Exception as e:
    print(f"  ❌ ERRO: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ TODOS OS TESTES PASSARAM - ROTEAMENTO 100% EXÓGENO")
print("="*70)
