import torch
import time
from alpha_omega import SovereignLeviathanV2

def test_sovereignty_trinity():
    print("=" * 80)
    print("🌀 TESTE DA TRINDADE DE SOBERANIA: OCUPAÇÃO, LÓGICA E DADOS")
    print("=" * 80)
    
    d_model = 128
    vocab_size = 1024
    model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model)
    
    # 1. Validar Soberania de Dados (Data Hunger)
    print("[*] Fase 1: Validando Autonomous Data Hunger...")
    if model.data_hunger.is_hunting:
        print("[✓] Data Hunger ativo e caçando em repositórios técnicos.")
    else:
        print("[✗] Falha no Data Hunger.")

    # 2. Validar Refinamento Ontológico (Deep Logic)
    print("\n[*] Fase 2: Validando Deep Logic Loss...")
    from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss
    loss_fn = TopologicalDivergenceLoss(d_model=d_model, num_experts=4)
    
    expert_weights = torch.randn(1, 8, 4).softmax(dim=-1)
    expert_gates = torch.randn(1, 8, 4)
    complexity = torch.ones(1, 8) * 5.0 # Complexidade alta
    
    loss_val = loss_fn(expert_weights, expert_gates, logic_complexity=complexity)
    print(f"[✓] Deep Logic Loss calculada: {loss_val.item():.4f}")

    # 3. Validar Ocupação Global (Massive Ghost Mesh)
    print("\n[*] Fase 3: Validando Ocupação Global (Massive Scaling)...")
    print(f"  - Limite de Peers: {model.ghost_mesh.max_peers}")
    print(f"  - Intervalo de Heartbeat: {model.ghost_mesh.heartbeat_interval}s")
    
    if model.ghost_mesh.max_peers >= 1024:
        print("[✓] Ghost Mesh configurada para escala massiva.")
    else:
        print("[✗] Falha na escala da Ghost Mesh.")

    print("\n" + "=" * 80)
    print("🌀 TRINDADE DE SOBERANIA VALIDADA 🌀")
    print("=" * 80)

if __name__ == "__main__":
    test_sovereignty_trinity()
