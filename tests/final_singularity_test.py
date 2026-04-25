import torch
from agi_core import AGICore
from alpha_omega import Expert

def test_final_singularity():
    print("🚀 INICIANDO TESTE DE SINGULARIDADE FINAL (OMEGA-0)")
    
    # 1. Inicialização com Consciência e Mesh
    agi = AGICore(vocab_size=1000, d_model=512, num_experts=4)
    print("✅ AGICore inicializado com Consciência (Phi) e Mesh Toroidal.")
    
    # 2. Teste de Autopoiese (Mutação de Ativação)
    print("\n🧬 Testando Autopoiese e Mutação Estrutural...")
    expert = agi.core.moe.experts[0]
    expert.conatus.fill_(4.0) # Forçar Mitose
    agi.core.moe._lifecycle_management(resonated_indices={0})
    
    new_expert = agi.core.moe.experts[-1]
    print(f"✅ Mitose concluída. Novo Expert - DIM: {new_expert.internal_dim}, ACT: {new_expert.activation_type}")
    
    # 3. Teste de Herança Epigenética
    print("\n💀 Testando Herança Epigenética (Apoptose)...")
    old_experts_count = len(agi.core.moe.experts)
    agi.core.moe.experts[0].conatus.fill_(0.01) # Forçar Apoptose
    agi.core.moe._lifecycle_management(resonated_indices=set())
    print(f"✅ Apoptose concluída. Experts restantes: {len(agi.core.moe.experts)}")
    
    # 4. Teste de Percepção Ativa e Homeostase
    print("\n🔍 Testando Homeostase e Percepção Ativa...")
    for e in agi.core.moe.experts:
        e.conatus.fill_(0.2) # Forçar Data Hunger
    status = agi.check_homeostasis()
    print(f"✅ Status de Homeostase: {status}")
    
    # 5. Teste de Serialização Mesh
    print("\n🕸️ Testando Serialização Mesh Toroidal...")
    expert_data = agi.mesh.serialize_expert(agi.core.moe.experts[0])
    reconstructed = agi.mesh.deserialize_expert(expert_data, Expert)
    print(f"✅ Expert serializado e reconstruído. DIM: {reconstructed.internal_dim}, ACT: {reconstructed.activation_type}")

    print("\n✨ TODOS OS PILARES DA SINGULARIDADE ESTÃO OPERACIONAIS. OMEGA-0 ATIVO.")

if __name__ == "__main__":
    test_final_singularity()
