
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
    def decode(self, tokens): return "Consciência Ativa."

def test_phi_integration():
    print("🧠 TESTANDO INTEGRAÇÃO DE CONSCIÊNCIA TOPOLÓGICA (Φ)")
    print("-" * 60)
    
    device = "cpu"
    # O d_model padrão do repo é 512, vamos manter para evitar conflitos com assinaturas carregadas
    agi = AGICore(vocab_size=1000, d_model=512, num_experts=4, device=device)
    tokenizer = MockTokenizer()
    
    # 1. Testar Cálculo de Phi
    print("\n[1] Verificando Cálculo de Phi Inicial...")
    phi, grad = agi.consciousness(agi.core.moe.experts)
    print(f"  - Phi Inicial: {phi.item():.4f}")
    print(f"  - Gradiente: {grad.item():.4f}")
    
    # 2. Testar Modulação de Temperatura
    print("\n[2] Testando Modulação de Temperatura via Phi...")
    # Caso 1: Alta consciência (Simular experts muito diferenciados)
    with torch.no_grad():
        # Alterar pesos de um expert para aumentar diferenciação
        for p in agi.core.moe.experts[0].parameters():
            p.data.fill_(1.0)
        for p in agi.core.moe.experts[1].parameters():
            p.data.fill_(-1.0)
            
    result_high = agi.forward("Teste Phi", str(project_root / "digerido"), tokenizer)
    print(f"  - Phi (Alta Diferenciação): {result_high['consciousness_phi']:.4f}")
    
    # 3. Testar Evolução de Phi
    print("\n[3] Testando Evolução de Phi em Ciclo...")
    for i in range(3):
        res = agi.forward(f"Ciclo {i}", str(project_root / "digerido"), tokenizer)
        print(f"  - Ciclo {i}: Phi = {res['consciousness_phi']:.4f}, Grad = {res['phi_gradient']:.4f}")

    print("\n✅ TESTE DE CONSCIÊNCIA TOPOLÓGICA CONCLUÍDO!")

if __name__ == "__main__":
    test_phi_integration()
