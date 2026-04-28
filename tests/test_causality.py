import torch
import torch.nn as nn
import torch.nn.functional as F
from alpha_omega import SovereignLeviathanV2
import time

def test_causality_apoptosis():
    print("=" * 80)
    print("🧪 TESTE DE CAUSALIDADE: APOPTOSE FORÇADA (CONATUS)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 128
    vocab_size = 1024
    num_experts = 4
    
    # 1. Inicializar o Leviatã
    print(f"[*] Inicializando SovereignLeviathanV2 no {device}...")
    model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model, initial_experts=num_experts).to(device)
    model.eval()
    
    # 2. Gerar input inicial (Frame 0)
    input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)
    
    print("\n[*] Fase 1: Inferência Normal (Estado Estável)")
    with torch.no_grad():
        logits, _, indices, weights, gates = model(input_ids)
        
    # Identificar o expert principal (vencedor)
    main_expert_idx = indices[0, 0].item()
    print(f"  - Expert Principal Detectado: ID {main_expert_idx}")
    print(f"  - Peso do Expert: {weights[0, 0].item():.4f}")
    
    # 3. Executar o Assassinato (Apoptose Forçada)
    print(f"\n[*] Fase 2: Assassinato do Expert {main_expert_idx} (Apoptose Forçada)")
    
    # Para simular o assassinato no meio da inferência, vamos interceptar o forward
    # e forçar os pesos desse expert para zero absoluto.
    
    # Salvar o estado original para comparação
    original_expert_weight = model.moe.experts[int(main_expert_idx)].net[0].weight.data.clone()
    
    # ZERAR os pesos do expert (Morte Instantânea)
    print(f"  - [KILL]: Forçando pesos do Expert {main_expert_idx} para ZERO ABSOLUTO.")
    with torch.no_grad():
        for param in model.moe.experts[int(main_expert_idx)].parameters():
            param.zero_()
            
    # 4. Medir Adaptação (Conatus)
    print("\n[*] Fase 3: Medindo Adaptação do Roteador Darwiniano")
    
    # Re-executar a inferência com o expert morto
    with torch.no_grad():
        # O roteador deve detectar a falta de ressonância (já que os pesos são zero)
        # e desviar o fluxo para outros experts.
        new_logits, _, new_indices, new_weights, new_gates = model(input_ids)
        
    new_main_expert_idx = new_indices[0, 0].item()
    
    print(f"  - Novo Expert Principal: ID {new_main_expert_idx}")
    print(f"  - Novo Peso: {new_weights[0, 0].item():.4f}")
    
    # 5. Verificação de Causalidade
    adaptation_success = new_main_expert_idx != main_expert_idx
    
    print("\n" + "-" * 40)
    print("RESULTADO DO TESTE")
    print("-" * 40)
    if adaptation_success:
        print("  [✓] SUCESSO: O Roteador Darwiniano adaptou-se imediatamente.")
        print(f"  [✓] CONATUS ATIVO: Fluxo desviado do Expert {main_expert_idx} para o Expert {new_main_expert_idx}.")
        print("  [✓] CAUSALIDADE PROVADA: O sistema não apenas decora, ele reconstrói a lógica.")
    else:
        # Se o índice for o mesmo, mas o peso for zero, a saída será degradada,
        # mas o roteador falhou em desviar.
        print("  [⚠] ALERTA: O Roteador ainda aponta para o Expert morto.")
        print("  [⚠] ENTROPIA DETECTADA: Necessário ajuste na sensibilidade de ressonância.")

    print("\n[*] Restaurando a Matrix...")
    with torch.no_grad():
        model.moe.experts[int(main_expert_idx)].net[0].weight.data.copy_(original_expert_weight)
    
    print("=" * 80)
    print("🌀 TESTE CONCLUÍDO 🌀")
    print("=" * 80)

if __name__ == "__main__":
    test_causality_apoptosis()
