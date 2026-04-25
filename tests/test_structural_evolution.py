
import torch
import sys
import os
from pathlib import Path

# Adicionar raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alpha_omega import OuroborosMoE

def test_structural_mutation():
    print("🧪 TESTANDO EVOLUÇÃO ESTRUTURAL (PILAR 3 - MYTHOS-CAPYBARA)")
    print("-" * 60)
    
    d_model = 128
    moe = OuroborosMoE(d_model, num_experts=1)
    
    initial_expert = moe.experts[0]
    initial_dim = initial_expert.internal_dim
    print(f"  - Dimensionalidade Interna Inicial: {initial_dim}")
    
    # Forçar Mitose com alto Conatus para testar expansão
    print("\n[1] Forçando Mitose Estrutural...")
    initial_expert.conatus.fill_(5.0) # Bem acima do threshold de 3.0
    
    # Gatilhar ciclo de vida
    moe._lifecycle_management(set([0]))
    
    print(f"  - Experts após mitose: {len(moe.experts)}")
    
    # O expert original (Stable 9) permanece, e dois novos são criados
    for i, expert in enumerate(moe.experts):
        print(f"    Expert {i}: Internal Dim = {expert.internal_dim}, Conatus = {expert.conatus.item():.2f}")
        if i > 0: # Novos experts
            assert expert.internal_dim > initial_dim, f"Expert {i} deveria ter expandido a dimensionalidade!"

    # Testar persistência da estrutura
    print("\n[2] Testando Persistência da Estrutura...")
    save_path = "tests/structural_ancestry.pt"
    moe.save_ancestry(save_path)
    
    new_moe = OuroborosMoE(d_model, num_experts=1)
    new_moe.load_ancestry(save_path)
    
    print(f"  - Experts carregados: {len(new_moe.experts)}")
    for i, expert in enumerate(new_moe.experts):
        print(f"    Expert {i} Carregado: Internal Dim = {expert.internal_dim}")
        if i > 0:
            assert expert.internal_dim > initial_dim, "Falha ao persistir dimensionalidade expandida!"

    print("\n✅ TESTE DE EVOLUÇÃO ESTRUTURAL PASSOU!")
    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    test_structural_mutation()
