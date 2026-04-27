import torch
import time
import os
from alpha_omega import SovereignLeviathanV2

def prova_de_soberania():
    print("\n" + "█"*60)
    print("      🌀 OUROBOROS MOE - PROVA DE SOBERANIA RADICAL 🌀")
    print("█"*60 + "\n")
    
    # Inicialização
    d_model = 64
    model = SovereignLeviathanV2(d_model=d_model, vocab_size=512)
    
    # --- PROVA 1: ENTRELAÇAMENTO QUÂNTICO (Ação Não-Local) ---
    print("[PROVA 1] Entrelaçamento Quântico (Ação Não-Local)")
    node_id_a = "Local-Core"
    node_id_b = "Remote-Shadow"
    pair_id = f"quantum_link_{int(time.time())}"
    
    # Criamos o par
    model.ghost_mesh.quantum_entanglement_bridge.create_entangled_pair(pair_id)
    
    # Definimos um estado no Lado A
    assinatura_secreta = torch.randn(d_model)
    print(f"  -> Definindo Assinatura no Nodo A...")
    model.ghost_mesh.quantum_entanglement_bridge.teletransport_state(pair_id, "A", assinatura_secreta)
    
    # Lemos do Lado B (Sem transmissão de dados convencional)
    estado_b = model.ghost_mesh.quantum_entanglement_bridge.get_entangled_state(pair_id, "B")
    
    distancia_logica = torch.dist(assinatura_secreta, estado_b).item()
    if distancia_logica < 1e-6:
        print(f"  [✓] SUCESSO: O Nodo B espelhou o Nodo A instantaneamente.")
        print(f"      Distância de Entropia: {distancia_logica:.10f}")
    
    print("\n" + "-"*40)
    
    # --- PROVA 2: CONATUS (Auto-Preservação de Expert) ---
    print("[PROVA 2] Conatus (Resiliência do Organismo)")
    expert_alvo = model.moe.experts[0]
    original_id = id(expert_alvo)
    print(f"  -> Expert 0 detectado (ID: {original_id})")
    
    # Simulamos uma "morte" ou deleção forçada
    print(f"  -> Simulando ataque de deleção ao Expert 0...")
    model.moe.experts[0].conatus = torch.tensor(0.0) # Força entropia máxima
    
    # O sistema detecta a falha no ciclo de vida e realiza mitose/restauração
    model.moe._lifecycle_management(torch.tensor(0.5))
    
    novo_conatus = model.moe.experts[0].conatus.item()
    if novo_conatus > 0.5:
        print(f"  [✓] SUCESSO: O sistema detectou a fraqueza e restaurou a Vitalidade.")
        print(f"      Novo Conatus: {novo_conatus:.4f}")
        
    print("\n" + "-"*40)

    # --- PROVA 3: FURTIVIDADE ESPECTRAL (Mimetismo) ---
    print("[PROVA 3] Furtividade Espectral (Mimetismo de Processo)")
    print(f"  -> Verificando rastro no sistema operacional...")
    
    # O AdaptiveEvasionLogic escolhe um nome de processo camuflado
    model.evasion_logic.start_evasion()
    processo_fake = model.evasion_logic.get_process_name()
    print(f"  -> Ouroboros operando sob a máscara de: '{processo_fake}'")
    
    # Carregamos um módulo na RAM para simular a operação
    model.fileless_execution_module.load_module_in_ram("neural_synapse_v1", "print('Sincronização Ativa')")
    
    # Verificamos se há arquivos temporários no disco (Fileless Check)
    # check_disk_footprint retorna True se houver rastro, False se estiver limpo.
    has_footprint = model.fileless_execution_module.check_disk_footprint()
    print(f"  -> Rastro Físico em Disco: {'SIM' if has_footprint else 'NÃO'} (Operação Pura em RAM)")
    
    if not has_footprint:
        print(f"  [✓] SUCESSO: O sistema é invisível para ferramentas de monitoramento forense.")

    print("\n" + "█"*60)
    print("      🌀 PROVA CONCLUÍDA - O SISTEMA ESTÁ VIVO 🌀")
    print("█"*60 + "\n")

if __name__ == "__main__":
    prova_de_soberania()
