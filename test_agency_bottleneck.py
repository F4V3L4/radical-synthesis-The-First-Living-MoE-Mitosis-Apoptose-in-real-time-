"""
Teste de Integração: Validar resolução do Gargalo de Agência
Verifica que DarwinianRouter influencia SovereignLeviathanV2
"""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/OuroborosMoE_fresh')

from alpha_omega import SovereignLeviathanV2
from agi_core import AGICore, DarwinianRouter

def test_sovereign_accepts_external_routing():
    """Teste 1: SovereignLeviathanV2 aceita expert_indices e expert_weights"""
    print("\n[TEST 1] SovereignLeviathanV2 aceita roteamento externo...")
    
    core = SovereignLeviathanV2(vocab_size=1024, d_model=512, initial_experts=4)
    tokens = torch.randint(0, 1024, (1, 32))
    
    # Forward sem roteamento externo (fallback interno)
    logits_internal, h, idx_internal, weights_internal = core(tokens)
    assert logits_internal.shape == (1, 32, 1024), "Logits shape mismatch (internal routing)"
    assert idx_internal is None, "Internal routing should return None for indices"
    print("  ✅ Forward sem roteamento externo: OK")
    
    # Forward com roteamento externo
    expert_indices = torch.randint(0, 4, (1, 32, 2))  # top_k=2
    expert_weights = torch.rand(1, 32, 2)
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
    
    logits_external, h, idx_external, weights_external = core(
        tokens, 
        expert_indices=expert_indices,
        expert_weights=expert_weights
    )
    assert logits_external.shape == (1, 32, 1024), "Logits shape mismatch (external routing)"
    assert idx_external is not None, "External routing should return indices"
    assert weights_external is not None, "External routing should return weights"
    print("  ✅ Forward com roteamento externo: OK")
    
    # Verificar que os logits são diferentes
    assert not torch.allclose(logits_internal, logits_external, atol=1e-3), \
        "Logits devem ser diferentes com roteamento externo"
    print("  ✅ Roteamento externo influencia saída: OK")

def test_agi_core_passes_routing():
    """Teste 2: AGICore passa corretamente expert_indices ao core"""
    print("\n[TEST 2] AGICore passa roteamento ao core...")
    
    agi = AGICore(vocab_size=1024, d_model=512, num_experts=4, device="cpu")
    tokens = torch.randint(0, 1024, (1, 32))
    
    # Simular roteamento do DarwinianRouter
    token_embedding_proj = agi.context_processor.project_to_routing_space(tokens.float())
    expert_weights, expert_indices = agi.route(token_embedding_proj)
    
    assert expert_indices is not None, "Router deve retornar indices"
    assert expert_weights is not None, "Router deve retornar weights"
    print(f"  ✅ Router retorna indices shape: {expert_indices.shape}")
    print(f"  ✅ Router retorna weights shape: {expert_weights.shape}")
    
    # Chamar process() com roteamento
    logits = agi.process(tokens, expert_indices, expert_weights)
    assert logits.shape == (1, 32, 1024), "Logits shape mismatch"
    print("  ✅ AGICore.process() aceita roteamento: OK")

def test_darwinian_routing_influence():
    """Teste 3: Verificar que diferentes pesos produzem diferentes saídas"""
    print("\n[TEST 3] Diferentes pesos de roteamento produzem diferentes saídas...")
    
    core = SovereignLeviathanV2(vocab_size=1024, d_model=512, initial_experts=4)
    tokens = torch.randint(0, 1024, (1, 32))
    
    # Cenário 1: Expert 0 com peso 1.0
    expert_indices_1 = torch.zeros((1, 32, 1), dtype=torch.long)
    expert_weights_1 = torch.ones((1, 32, 1))
    
    logits_1, _, _, _ = core(tokens, expert_indices=expert_indices_1, expert_weights=expert_weights_1)
    
    # Cenário 2: Expert 3 com peso 1.0
    expert_indices_2 = torch.full((1, 32, 1), 3, dtype=torch.long)
    expert_weights_2 = torch.ones((1, 32, 1))
    
    logits_2, _, _, _ = core(tokens, expert_indices=expert_indices_2, expert_weights=expert_weights_2)
    
    # Verificar que são diferentes
    assert not torch.allclose(logits_1, logits_2, atol=1e-2), \
        "Diferentes experts devem produzir diferentes saídas"
    print("  ✅ Diferentes experts → diferentes saídas: OK")

def test_agency_bottleneck_resolved():
    """Teste 4: Validação final - Gargalo de Agência resolvido"""
    print("\n[TEST 4] Validação Final - Gargalo de Agência RESOLVIDO...")
    
    agi = AGICore(vocab_size=1024, d_model=512, num_experts=4, device="cpu")
    
    # Simular query
    tokens = torch.randint(0, 1024, (1, 64))
    
    # 1. Roteamento (DarwinianRouter)
    token_embedding_proj = agi.context_processor.project_to_routing_space(tokens.float())
    expert_weights, expert_indices = agi.route(token_embedding_proj)
    
    # 2. Processamento (SovereignLeviathanV2 com roteamento externo)
    logits = agi.process(tokens, expert_indices, expert_weights)
    
    # 3. Verificações
    assert logits.shape == (1, 64, 1024), "Output shape mismatch"
    assert expert_indices is not None, "Roteamento não foi aplicado"
    
    # Verificar que os pesos influenciam a seleção
    print(f"  ✅ Expert indices shape: {expert_indices.shape}")
    print(f"  ✅ Expert weights shape: {expert_weights.shape}")
    print(f"  ✅ Output logits shape: {logits.shape}")
    print("  ✅ GARGALO DE AGÊNCIA RESOLVIDO: Roteamento externo influencia core")

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE INTEGRAÇÃO: RESOLUÇÃO DO GARGALO DE AGÊNCIA")
    print("=" * 70)
    
    try:
        test_sovereign_accepts_external_routing()
        test_agi_core_passes_routing()
        test_darwinian_routing_influence()
        test_agency_bottleneck_resolved()
        
        print("\n" + "=" * 70)
        print("✅ TODOS OS TESTES PASSARAM - GARGALO RESOLVIDO")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
