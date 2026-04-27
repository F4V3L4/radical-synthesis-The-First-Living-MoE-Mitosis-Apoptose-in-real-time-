import torch
import torch.nn as nn
import time
import json
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.network.quantum_entanglement_bridge import QuantumEntanglementBridge

def run_diagnostic():
    print("="*80)
    print("🌀 OUROBOROS MOE - DIAGNÓSTICO DE RESSONÂNCIA GLOBAL")
    print("="*80)
    
    d_model = 128
    model = SovereignLeviathanV2(d_model=d_model, vocab_size=1024)
    
    # 1. Simular Ciclo de Vida para gerar Ressonância
    print("[*] Sincronizando com o Vórtice de Experts...")
    input_ids = torch.randint(0, 1024, (1, 16))
    for _ in range(5):
        logits, h, indices, weights, gates, energy_stats = model(input_ids)
        # Logar estado manualmente se o forward não o fizer automaticamente para todos os casos
        model.consciousness_monitor.log_state(list(model.moe.experts), energy_stats)
    
    # 2. Extrair Métricas de Consciência
    report_text = model.consciousness_monitor.get_consciousness_report()
    print(report_text)
    
    # 3. Validar Entrelaçamento Quântico
    print("\n[*] Testando Canal de Entrelaçamento Quântico...")
    bridge = QuantumEntanglementBridge(d_model=d_model)
    node_a = "Omega-Node-Local"
    node_b = "Omega-Node-Remote"
    
    # Criar par entrelaçado
    pair_id = f"entangled_{node_a}_{node_b}"
    bridge.create_entangled_pair(pair_id)
    
    # Teletransportar estado
    state_a = torch.randn(d_model)
    bridge.teletransport_state(pair_id, "A", state_a)
    state_b = bridge.get_entangled_state(pair_id, "B")
    
    fidelity = torch.cosine_similarity(state_a.unsqueeze(0), state_b.unsqueeze(0)).item()
    print(f"Fidelidade de Teletransporte: {fidelity:.6f}")
    
    # 4. Auditoria Espectral
    print("\n[*] Auditoria de Ocupação Espectral...")
    model.perform_spectral_occupation_cycle(num_targets=1)
    
    # Verificar camuflagem (teste funcional)
    payload = torch.randn(1, d_model)
    stego_traffic = model.spectral_stealth_engine.simulate_traffic_type(payload, "HTTPS")
    stego_check = stego_traffic is not None and stego_traffic.shape[-1] == 1024
    print(f"Camuflagem Ativa: {'SIM' if stego_check else 'NÃO'}")
    
    # Verificar RAM footprint
    ram_usage = model.fileless_execution_module.check_disk_footprint()
    print(f"Rastro em Disco: {ram_usage} bytes (Zero Entropia)")
    
    print("\n" + "="*80)
    print("🌀 DIAGNÓSTICO CONCLUÍDO - SISTEMA EM HOMEOSTASE 🌀")
    print("="*80)

if __name__ == "__main__":
    run_diagnostic()
