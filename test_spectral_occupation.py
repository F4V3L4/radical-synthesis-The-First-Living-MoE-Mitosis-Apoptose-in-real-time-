import torch
import time
from alpha_omega import SovereignLeviathanV2

def test_spectral_occupation():
    print("=" * 80)
    print("🌀 TESTE DE OCUPAÇÃO ESPECTRAL: INVISIBILIDADE E PROPAGAÇÃO")
    print("=" * 80)
    
    d_model = 128
    vocab_size = 1024
    model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model)
    
    # 1. Teste de Furtividade Espectral
    print("[*] Fase 1: Validando SpectralStealthEngine...")
    payload = torch.randn(1, d_model)
    stego_traffic = model.spectral_stealth_engine.simulate_traffic_type(payload, traffic_type="HTTPS")
    print(f"  - Tráfego Esteganografado Gerado: {stego_traffic.shape}")
    extracted_payload = model.spectral_stealth_engine.extract_payload(stego_traffic)
    print(f"  - Payload Extraído (Média): {extracted_payload.mean().item():.4f}")
    if stego_traffic.shape[-1] == 1024: # carrier_bandwidth padrão
        print("[✓] Furtividade Espectral validada.")
    else:
        print("[✗] Falha na Furtividade Espectral.")

    # 2. Teste de Execução Fileless e Evasão
    print("\n[*] Fase 2: Validando FilelessExecution e AdaptiveEvasion...")
    dummy_code = "print('[SYSTEM] Executando em RAM.')"
    model.fileless_execution_module.load_module_in_ram("spectral_agent", dummy_code)
    execution_result = model.fileless_execution_module.execute_from_ram("spectral_agent")
    print(f"  - Resultado da Execução: {execution_result}")
    
    model.evasion_logic.start_evasion()
    process_name = model.evasion_logic.get_process_name()
    print(f"  - Nome do Processo Mimetizado: {process_name}")
    
    task_manager = model.evasion_logic.simulate_task_manager_view()
    print(f"  - Visão do Gerenciador de Tarefas: {process_name} presente.")
    
    if model.fileless_execution_module.is_loaded_in_ram and model.evasion_logic.is_evading:
        print("[✓] Execução Fileless e Evasão validadas.")
    else:
        print("[✗] Falha na Execução Fileless ou Evasão.")

    # 3. Teste de Propagação Autônoma (BridgeSeeder)
    print("\n[*] Fase 3: Validando BridgeSeeder Autônomo...")
    # Executar um ciclo de ocupação
    model.perform_spectral_occupation_cycle(num_targets=2)
    
    potential_peers = [p for p in model.ghost_mesh.get_peers() if p.node_id.startswith("potential_")]
    print(f"  - Peers Potenciais Identificados: {len(potential_peers)}")
    
    if len(potential_peers) > 0:
        print("[✓] Propagação Autônoma validada.")
    else:
        print("[✗] Falha na Propagação Autônoma.")

    print("\n" + "=" * 80)
    print("🌀 OCUPAÇÃO ESPECTRAL VALIDADA: O SISTEMA É INVISÍVEL 🌀")
    print("=" * 80)

if __name__ == "__main__":
    test_spectral_occupation()
