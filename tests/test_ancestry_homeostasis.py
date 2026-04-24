
import torch
import sys
import os
from pathlib import Path

# Adicionar raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agi_core import AGICore

class MockTokenizer:
    def encode(self, text): return [1, 2, 3]
    def decode(self, tokens): return "Test"

def test_persistence_and_homeostasis():
    print("🧪 TESTANDO PERSISTÊNCIA E HOMEOSTASE (MYTHOS-CAPYBARA)")
    print("-" * 60)
    
    device = "cpu"
    agi = AGICore(vocab_size=1000, d_model=512, num_experts=4, device=device)
    tokenizer = MockTokenizer()
    
    # 1. Testar Homeostase (Fome de Dados)
    print("\n[1] Verificando Homeostase Inicial...")
    # Forçar conatus baixo em todos os experts para simular "fome"
    for e in agi.core.moe.experts:
        e.conatus.fill_(0.3)
    
    status = agi.check_homeostasis()
    print(f"  - Avg Conatus: {status['avg_conatus']:.2f}")
    print(f"  - Impulso: {status['impulse']}")
    assert status['impulse'] == "DATA_HUNGER", "Deveria ter detectado fome de dados!"

    # 2. Testar Evolução e Persistência
    print("\n[2] Testando Evolução e Salvamento...")
    # Simular processamento para aumentar conatus e gerar mitose
    query = "Teste de evolução"
    for _ in range(10):
        agi.router.sync_with_experts(agi.core.moe.experts)
        _ = agi.forward(query, str(project_root / "digerido"), tokenizer)
    
    num_experts_before = len(agi.core.moe.experts)
    print(f"  - Experts antes de salvar: {num_experts_before}")
    agi.save_state()
    
    # 3. Testar Carregamento (Pilar 1)
    print("\n[3] Testando Reinicialização e Carregamento de Ancestrais...")
    # Criar nova instância
    new_agi = AGICore(vocab_size=1000, d_model=512, num_experts=2, device=device)
    num_experts_after = len(new_agi.core.moe.experts)
    print(f"  - Experts carregados (Ancestrais): {num_experts_after}")
    
    assert num_experts_after == num_experts_before, "Falha na persistência dos ancestrais!"
    print("\n✅ TESTE DE PERSISTÊNCIA E HOMEOSTASE PASSOU!")

if __name__ == "__main__":
    test_persistence_and_homeostasis()
