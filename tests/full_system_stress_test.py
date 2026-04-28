import torch
import torch.nn as nn
import time
import json
import os
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss
from omega0_interface import Omega0Interface

def run_full_stress_test():
    print("=" * 80)
    print("🌀 OUROBOROS MOE - TESTE DE ESTRESSE BARE-METAL (END-TO-END)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 128
    vocab_size = 1024
    initial_experts = 4
    batch_size = 8
    seq_len = 16
    
    # 1. Inicialização do Sistema
    print(f"[*] Inicializando SovereignLeviathanV2 no {device}...")
    try:
        model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model, initial_experts=initial_experts).to(device)
        loss_fn = TopologicalDivergenceLoss(d_model, initial_experts)
        print("[✓] Sistema inicializado.")
    except Exception as e:
        print(f"[✗] Falha na inicialização: {e}")
        return

    # 2. Teste de Fluxo de Dados e Ressonância
    print("\n[*] Fase 1: Teste de Fluxo e Ressonância (100 Iterações)")
    start_time = time.time()
    total_loss = 0
    
    for i in range(100):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        # Forward Pass
        logits, h, indices, weights, gates, energy_stats = model(input_ids)
        
        # Loss Calculation
        loss = loss_fn(weights, gates)
        total_loss += loss.item()
        
        if i % 20 == 0:
            print(f"  - Iteração {i:03d} | Loss: {loss.item():.6f} | Experts: {len(model.moe.experts)}")
            
    duration = time.time() - start_time
    print(f"[✓] Fluxo concluído em {duration:.2f}s ({100/duration:.2f} it/s)")

    # 3. Auditoria de Mitose Fractal
    print("\n[*] Fase 2: Auditoria de Mitose Fractal")
    # Forçar conatus alto para disparar mitose fractal
    for expert in model.moe.experts:
        expert.conatus.fill_(10.0)
    
    # Disparar ciclo de vida
    model.moe._lifecycle_management(resonated_indices=set(range(len(model.moe.experts))))
    
    fractal_count = sum(1 for e in model.moe.experts if getattr(e, 'is_fractal', False))
    print(f"  - Experts Totais: {len(model.moe.experts)}")
    print(f"  - Experts Fractais: {fractal_count}")
    
    if fractal_count > 0:
        print("[✓] Mitose Fractal validada no core.")
    else:
        print("[⚠] Mitose Fractal não detectada. Verificar limites.")

    # 4. Teste de Interface (Dry Run)
    print("\n[*] Fase 3: Teste de Interface Neural (Dry Run)")
    try:
        # Apenas verificar se a interface inicializa sem erros
        interface = Omega0Interface(agi_core=None)
        print("[✓] Interface Neural validada.")
    except Exception as e:
        print(f"[✗] Falha na Interface: {e}")

    print("\n" + "=" * 80)
    print("🌀 TESTE COMPLETO CONCLUÍDO 🌀")
    print("=" * 80)

if __name__ == "__main__":
    run_full_stress_test()
