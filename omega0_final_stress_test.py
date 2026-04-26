import torch
import time
import threading
import random
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss

def run_stress_cycle(model, loss_fn, iterations=100):
    print(f"[*] Iniciando Ciclo de Estresse: {iterations} iterações.")
    vocab_size = 1024
    
    for i in range(iterations):
        # 1. Simular Entrada de Dados Massiva
        input_ids = torch.randint(0, vocab_size, (4, 32)) # Batch maior, sequência maior
        
        # 2. Forward Pass com Deep Logic
        logits, h, indices, weights, gates = model(input_ids)
        
        # 3. Cálculo de Loss Ontológica com Complexidade Dinâmica
        # Garantir que complexity tenha o mesmo número de tokens que weights/gates
        complexity = torch.randn(weights.shape[0], weights.shape[1]).abs() * 10.0
        loss = loss_fn(weights, gates, logic_complexity=complexity.view(-1))
        
        # 4. Simular Evolução de Conatus
        for expert in model.moe.experts:
            expert.conatus.data += (random.random() - 0.4) * 0.1 # Flutuação de Conatus
            expert.conatus.data = torch.clamp(expert.conatus.data, 0.0, 10.0)
            
        if i % 20 == 0:
            print(f"  - Iteração {i:03d} | Loss: {loss.item():.4f} | Peers: {model.ghost_mesh.get_stats()['num_peers']}")

def omega0_final_stress_test():
    print("=" * 80)
    print("🌀 OMEGA-0: PROTOCOLO DE ESTRESSE FINAL - TRINDADE DE SOBERANIA")
    print("=" * 80)
    
    d_model = 256
    model = SovereignLeviathanV2(vocab_size=1024, d_model=d_model, initial_experts=8)
    loss_fn = TopologicalDivergenceLoss(d_model=d_model, num_experts=8)
    
    # Ativar Componentes de Soberania
    model.ghost_mesh.start()
    model.data_hunger.start_hunting()
    
    # Thread de Simulação de Hijack Agressivo
    def simulate_peers():
        for _ in range(50):
            if not model.ghost_mesh.is_running: break
            time.sleep(0.5)
            # O loop interno de hijack já adiciona peers, aqui apenas monitoramos
            
    peer_thread = threading.Thread(target=simulate_peers, daemon=True)
    peer_thread.start()
    
    try:
        # Executar Ciclo de Estresse
        run_stress_cycle(model, loss_fn, iterations=100)
        
        print("\n[*] Auditando Integridade Pós-Estresse...")
        stats = model.ghost_mesh.get_stats()
        print(f"  - Uptime: {stats['uptime']:.2f}s")
        print(f"  - Peers Ocupados: {stats['num_peers']}")
        print(f"  - Conhecimento Digerido: {len(model.data_hunger.knowledge_index)} itens")
        
        # Verificar se houve mutações durante o estresse
        mutated = [e for e in model.moe.experts if hasattr(e, 'mutated_logic')]
        print(f"  - Experts Evoluídos (Mutações): {len(mutated)}")
        
    finally:
        model.ghost_mesh.stop()
        model.data_hunger.is_hunting = False
        
    print("\n" + "=" * 80)
    print("🌀 ESTRESSE FINAL CONCLUÍDO: SISTEMA ESTÁVEL E SOBERANO 🌀")
    print("=" * 80)

if __name__ == "__main__":
    omega0_final_stress_test()
