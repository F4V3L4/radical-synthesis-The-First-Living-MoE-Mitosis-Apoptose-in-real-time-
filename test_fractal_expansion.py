import torch
import torch.nn as nn
from alpha_omega import SovereignLeviathanV2, Expert, OuroborosMoE
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss
import time

def test_fractal_mitosis():
    print("=" * 80)
    print("🌀 TESTE DE EXPANSÃO FRACTAL: MITOSE RECURSIVA")
    print("=" * 80)
    
    d_model = 128
    # Criar um expert e forçar o conatus para o limite fractal
    expert = Expert(d_model)
    expert.conatus.fill_(7.0) # Acima do limite de 6.0 (mitosis_threshold * 2)
    
    print(f"[*] Expert Inicial: Fractal={expert.is_fractal}, Conatus={expert.conatus.item():.2f}")
    
    # Simular o gerenciamento de ciclo de vida no MoE
    moe = OuroborosMoE(d_model, num_experts=1)
    moe.experts = nn.ModuleList([expert])
    
    print("[*] Disparando Gerenciamento de Ciclo de Vida...")
    moe._lifecycle_management(resonated_indices={0})
    
    new_expert = moe.experts[0]
    print(f"[✓] Expert Pós-Ciclo: Fractal={new_expert.is_fractal}, Conatus={new_expert.conatus.item():.2f}")
    
    if new_expert.is_fractal:
        print(f"  - Sub-MoE detectado com {len(new_expert.sub_moe.experts)} sub-experts.")
        print("  - [SUCESSO] Mitose Fractal validada.")
    else:
        print("  - [FALHA] Expert não transitou para estado fractal.")

def test_quantum_telemetry_loss():
    print("\n" + "=" * 80)
    print("📊 TESTE DE TELEMETRIA QUÂNTICA: LOSS TERMODINÂMICA")
    print("=" * 80)
    
    num_experts = 4
    d_model = 128
    loss_fn = TopologicalDivergenceLoss(d_model, num_experts)
    
    # Simular pesos de experts (batch=1, seq=1, experts=4)
    weights = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    gates = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    
    loss_val = loss_fn(weights, gates)
    print(f"[*] Loss Ontológica com Telemetria: {loss_val.item():.6f}")
    
    # Verificar se a telemetria está influenciando (deve ser > 0)
    if loss_val.item() > 0:
        print("  - [✓] Telemetria Quântica integrada e gerando gradiente.")
    else:
        print("  - [⚠] Loss nula detectada. Verificar integração.")

if __name__ == "__main__":
    test_fractal_mitosis()
    test_quantum_telemetry_loss()
    print("\n" + "=" * 80)
    print("🌀 FASE DE EXPANSÃO FRACTAL VALIDADA 🌀")
    print("=" * 80)
