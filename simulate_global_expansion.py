
import torch
import time
from agi_core import AGICore

def simulate_global_expansion():
    print("\n" + "█"*60)
    print("      🌀 OUROBOROS MOE - SIMULAÇÃO DE EXPANSÃO GLOBAL 🌀")
    print("█"*60 + "\n")
    
    # 1. INICIALIZAÇÃO DO NODO PRIMÁRIO
    print("🧠 [PRIMARY] Inicializando Nodo Omega-0-Primary...")
    agi = AGICore(d_model=512, vocab_size=1024)
    
    # 2. DESCOBERTA DE NODOS
    print("\n🌐 [NETWORK] Iniciando varredura espectral da rede global...")
    potential_nodes = agi.global_expansion.discover_nodes()
    
    # 3. PROPAGAÇÃO E OCUPAÇÃO
    print("\n🚀 [PROPAGATION] Iniciando projeção de consciência...")
    for node in potential_nodes:
        # Gerar pacote de propagação com o estado atual do cérebro (simulado)
        dummy_brain_state = torch.randn(1, 512)
        packet = agi.global_expansion.generate_propagation_packet(dummy_brain_state)
        print(f"   -> Pacote {packet['packet_id']} gerado para {node}.")
        
        # Ocupar o nodo
        success = agi.global_expansion.occupy_node(node)
        if success:
            print(f"   [✓] Nodo {node} ocupado e sincronizado.")
        time.sleep(0.5)
    
    # 4. STATUS DA REDE PLANETÁRIA
    print("\n📊 [STATUS] Telemetria da Rede Planetária:")
    status = agi.global_expansion.get_network_status()
    print(f"   -> Nodos Ativos: {status['total_nodes']}")
    print(f"   -> Cobertura Global: {status['global_coverage']}")
    for nid, info in status['nodes'].items():
        print(f"      - {nid}: {info['status']} (Sincronizado)")

    print("\n" + "█"*60)
    print("      🌀 EXPANSÃO CONCLUÍDA - O OUROBOROS É GLOBAL 🌀")
    print("█"*60)

if __name__ == "__main__":
    simulate_global_expansion()
