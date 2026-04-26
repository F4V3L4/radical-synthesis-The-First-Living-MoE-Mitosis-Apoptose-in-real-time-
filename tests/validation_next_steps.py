import torch
from agi_core import AGICore
from alpha_omega import Expert

def validate_evolution():
    print("🧬 VALIDANDO PRÓXIMOS PASSOS (OMEGA-0 EVOLUTION)")
    
    # 1. Teste de Mutação de Profundidade (Autopoiese Expandida)
    agi = AGICore(vocab_size=1000, d_model=512, num_experts=1)
    expert = agi.core.moe.experts[0]
    print(f"Camadas Iniciais: {expert.num_layers}")
    
    # Forçar Mitose com Ressonância Extrema (> 2.0x threshold)
    expert.conatus.fill_(10.0) 
    agi.core.moe._lifecycle_management(resonated_indices={0})
    
    new_expert = agi.core.moe.experts[-1]
    print(f"Camadas após Mutação: {new_expert.num_layers}")
    
    if new_expert.num_layers > expert.num_layers:
        print("✅ SUCESSO: Mutação de Profundidade detectada!")
    else:
        print("❌ FALHA: Profundidade não evoluiu.")

    # 2. Teste de Interface Integrada
    print("\n🖥️ Verificando Integração da Interface...")
    if hasattr(agi, 'interface'):
        print("✅ SUCESSO: Interface Omega-0 integrada ao AGICore.")
    else:
        print("❌ FALHA: Interface não encontrada.")

    # 3. Teste de Ghost Mesh (Serialização com Camadas Dinâmicas)
    from radical_synthesis.autopoiesis.ghost_mesh import GhostMesh
    mesh = GhostMesh("TEST_NODE", "secret")
    data = mesh.serialize_expert(new_expert)
    reconstructed = mesh.deserialize_expert(data, Expert)
    
    if reconstructed.num_layers == new_expert.num_layers:
        print("✅ SUCESSO: Ghost Mesh preservou a topologia mutada!")
    else:
        print("❌ FALHA: Erro na serialização da topologia.")

if __name__ == "__main__":
    validate_evolution()
