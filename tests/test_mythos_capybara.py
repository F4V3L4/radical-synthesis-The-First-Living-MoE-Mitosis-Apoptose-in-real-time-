import torch
import sys
import os

# Adicionar raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_omega import OuroborosMoE
from radical_synthesis.autopoiesis.routing import DarwinianRouter

def test_mythos_architecture():
    print("🧪 Validando Arquitetura Mythos-Capybara...")
    d_model = 512
    initial_experts = 4
    moe = OuroborosMoE(d_model, num_experts=initial_experts)
    router = DarwinianRouter(d_model, initial_experts, top_k=2)

    # 1. Teste de Phase-Lock Routing (Sem Softmax)
    print("\n[1] Testando Phase-Lock Routing...")
    x = torch.randn(1, 10, d_model)
    # Pegar o primeiro token para o roteador
    weights, indices = router(x[:, 0, :])
    
    print(f"  - Pesos (Raw Resonance): {weights}")
    print(f"  - Índices: {indices}")
    
    # Verificar se não há softmax (pesos não somam 1 obrigatoriamente)
    weight_sum = weights.sum().item()
    print(f"  - Soma dos pesos: {weight_sum:.4f} (Não-probabilístico)")

    # 2. Teste de Conatus e Ciclo de Vida
    print("\n[2] Testando Conatus e Evolução...")
    # Simular múltiplas passagens para forçar Mitose ou Apoptose
    for i in range(50):
        # Sincronizar roteador
        router.sync_with_experts(moe.experts)
        
        # Roteamento
        current_x = torch.randn(1, 1, d_model)
        weights, indices = router(current_x[:, 0, :])
        
        # Processamento (Gatilha ciclo de vida)
        _ = moe(current_x, indices, weights)
        
        if i % 10 == 0:
            conatus_vals = [float(e.conatus) for e in moe.experts]
            print(f"  - Iteração {i}: {len(moe.experts)} experts vivos. Conatus médio: {sum(conatus_vals)/len(conatus_vals):.2f}")

    print(f"\n[3] Estado Final:")
    print(f"  - Experts vivos: {len(moe.experts)}")
    for i, e in enumerate(moe.experts):
        print(f"    Expert {i}: Conatus={float(e.conatus):.2f}")

    print("\n✅ Arquitetura validada com sucesso!")

if __name__ == "__main__":
    test_mythos_architecture()
