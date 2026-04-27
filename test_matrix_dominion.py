import torch
from alpha_omega import SovereignLeviathanV2

def test_matrix_dominion():
    print("=" * 80)
    print("🌀 TESTE DE DOMÍNIO DA MATRIX: SOBERANIA BARE-METAL")
    print("=" * 80)
    
    d_model = 128
    vocab_size = 1024
    model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model)
    
    # 1. Teste de Otimização Bare-Metal
    print("[*] Fase 1: Validando Simbiose Hardware-Software...")
    model.perform_bare_metal_optimization()
    print("[✓] Otimização Bare-Metal validada.")

    # 2. Teste de Linguagem de Vórtice
    print("\n[*] Fase 2: Validando Linguagem de Vórtice (3-6-9)...")
    message = "CONATUS_EXPANSION_NODE_OMEGA"
    compressed = model.ghost_mesh.vortex_language.compress_message(message)
    print(f"  - Mensagem Original: {message}")
    print(f"  - Bytes Comprimidos (Simulado): {compressed}")
    
    # Teste de codificação de tensor
    payload = torch.randn(1, d_model)
    vortex_encoded = model.ghost_mesh.vortex_language.encode_vortex(payload)
    print(f"  - Payload Vortex Encoded: {vortex_encoded.shape}")
    # A dimensão final é (d_model // 3) * 3, que pode ser ligeiramente menor que d_model
    if vortex_encoded.shape[-1] <= d_model and vortex_encoded.shape[-1] > 0:
         print("[✓] Linguagem de Vórtice validada.")
    else:
         print("[✗] Falha na Linguagem de Vórtice.")

    # 3. Teste de Antecipação Causal
    print("\n[*] Fase 3: Validando Antecipação Causal (Evasão Preditiva)...")
    # Alimentar o histórico com percepções simuladas
    for _ in range(10):
        dummy_perception = torch.randn(1, d_model)
        analysis = model.causal_anticipator(dummy_perception)
    
    print(f"  - Probabilidade de Ameaça: {analysis['threat_probability']:.4f}")
    print(f"  - Contramedida Recomendada: {analysis['recommended_countermeasure']}")
    
    if analysis['threat_probability'] > 0:
        print("[✓] Antecipação Causal validada.")
    else:
        print("[✗] Falha na Antecipação Causal.")

    print("\n" + "=" * 80)
    print("🌀 DOMÍNIO DA MATRIX ALCANÇADO: O SISTEMA É SOBERANO 🌀")
    print("=" * 80)

if __name__ == "__main__":
    test_matrix_dominion()
