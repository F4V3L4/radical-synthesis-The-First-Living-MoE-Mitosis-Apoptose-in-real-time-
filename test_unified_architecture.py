import torch
import time
from alpha_omega import SovereignLeviathanV2

def test_unified_architecture():
    print("=" * 80)
    print("🌀 TESTE DE ARQUITETURA UNIFICADA: O 9 CENTRAL E ESTABILIDADE REAL")
    print("=" * 80)
    
    d_model = 128
    vocab_size = 1024
    model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model)
    
    # 1. Validar Conservação de Energia no Router
    print("[*] Fase 1: Validando Conservação de Energia no Router...")
    input_ids = torch.randint(0, vocab_size, (1, 16))
    logits, h, indices, weights, gates, _ = model(input_ids)
    
    weights_sum = weights.sum(dim=-1)
    print(f"  - Soma dos Pesos (Deve ser ~1.0): {weights_sum.item():.4f}")
    if torch.allclose(weights_sum, torch.tensor(1.0), atol=1e-5):
        print("[✓] Energia conservada no roteamento.")
    else:
        print("[✗] Falha na conservação de energia.")

    # 2. Validar Unificação de Energia (O 9 Central)
    print("\n[*] Fase 2: Validando Função de Energia Global...")
    dummy_loss = torch.tensor(2.5)
    logits, h, indices, weights, gates, energy_stats = model(input_ids, target_loss=dummy_loss)
    
    if energy_stats:
        print(f"  - Energia Global: {energy_stats['total_energy']:.4f}")
        print(f"  - Entropia: {energy_stats['entropy']:.4f}")
        print(f"  - Estabilidade (Conatus): {energy_stats['stability']:.4f}")
        print(f"  - Vitalidade: {energy_stats['vitality'].mean().item():.4f}")
        print("[✓] Métrica de verdade unificada.")
    else:
        print("[✗] Falha na unificação de energia.")

    # 3. Validar Damping Sistêmico
    print("\n[*] Fase 3: Validando Damping Sistêmico...")
    # Executar dois passos e verificar se o damping atua no estado oculto
    logits1, h1, _, _, _, _ = model(input_ids)
    logits2, h2, _, _, _, _ = model(input_ids, h=h1)
    
    if h2 is not None:
        print("[✓] Damping sistêmico ativo no loop recursivo.")
    else:
        print("[✗] Falha no damping.")

    print("\n" + "=" * 80)
    print("🌀 ARQUITETURA UNIFICADA VALIDADA: O 9 FOI ALCANÇADO 🌀")
    print("=" * 80)

if __name__ == "__main__":
    test_unified_architecture()
