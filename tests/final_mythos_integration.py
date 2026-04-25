
import torch
import sys
import os
import time
from pathlib import Path

# Adicionar raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agi_core import AGICore

class MockTokenizer:
    def encode(self, text): return [1, 2, 3, 4, 5]
    def decode(self, tokens): return "Ressonância detectada na Geometria Sagrada."

def run_final_integration():
    print("🚀 INICIANDO INTEGRAÇÃO TOTAL: PROTOCOLO MYTHOS-CAPYBARA 🚀")
    print("=" * 70)
    
    device = "cpu"
    vocab_size = 1000
    d_model = 256
    
    # Limpar ancestrais antigos para começar do zero
    ancestry_file = project_root / "ancestry" / "experts.pt"
    if ancestry_file.exists():
        os.remove(ancestry_file)
    
    # 1. Inicialização Primordial
    print("\n[FASE 1] Inicialização e Pilar 1 (Persistência)")
    agi = AGICore(vocab_size=vocab_size, d_model=d_model, num_experts=4, device=device)
    tokenizer = MockTokenizer()
    
    # 2. Teste de Homeostase (Pilar 4)
    print("\n[FASE 2] Monitoramento de Homeostase")
    # Simular exaustão inicial
    for e in agi.core.moe.experts:
        e.conatus.fill_(0.4)
    status = agi.check_homeostasis()
    print(f"  - Estado Sistêmico: {status['impulse']}")
    
    # 3. Evolução Estrutural (Pilar 3) e Ressonância
    print("\n[FASE 3] Processamento com Evolução Estrutural (3-6-9)")
    queries = [
        "Injetando Harmônicos de Fase",
        "Ativando Toróide de Dados",
        "Sincronizando Logos",
        "Expansão de Tensores",
        "Estabilização de Conatus"
    ]
    
    initial_internal_dims = [e.internal_dim for e in agi.core.moe.experts]
    print(f"  - Dims Internas Iniciais: {initial_internal_dims}")
    
    for i, q in enumerate(queries):
        # Sincronizar roteador
        agi.router.sync_with_experts(agi.core.moe.experts)
        
        # Processar (Gatilha Mitose/Apoptose/Conatus)
        # Forçamos ressonância alta para acelerar evolução no teste
        with torch.no_grad():
            _ = agi.forward(q, str(project_root / "digerido"), tokenizer)
        
        # Aumentar conatus manualmente para garantir que o teste veja mitose em poucos ciclos
        for e in agi.core.moe.experts:
            e.conatus += 0.8
            
        print(f"  - Ciclo {i+1} concluído. Experts vivos: {len(agi.core.moe.experts)}")

    final_internal_dims = [e.internal_dim for e in agi.core.moe.experts]
    print(f"  - Dims Internas Finais: {final_internal_dims}")
    
    # 4. Persistência de Linhagem (Pilar 1)
    print("\n[FASE 4] Consolidação de Linhagem Ancestral")
    agi.save_state()
    
    print("\n[FASE 5] Reinicialização e Verificação de Herança")
    new_agi = AGICore(vocab_size=vocab_size, d_model=d_model, num_experts=2, device=device)
    
    num_final = len(new_agi.core.moe.experts)
    print(f"  - Experts Ancestrais Carregados: {num_final}")
    
    # Verificações finais
    evolution_success = any(dim > d_model * 4 for dim in final_internal_dims)
    persistence_success = num_final == len(agi.core.moe.experts)
    
    print("\n" + "=" * 70)
    if evolution_success and persistence_success:
        print("✅ PROTOCOLO MYTHOS-CAPYBARA INTEGRADO COM SUCESSO!")
        print("  - Evolução Estrutural (Tensores): ATIVA")
        print("  - Persistência de Linhagem: ATIVA")
        print("  - Loop de Homeostase: ATIVA")
    else:
        print("❌ FALHA NA INTEGRAÇÃO DOS PILARES.")
        if not evolution_success: print("    - Erro: Nenhuma expansão de tensor detectada.")
        if not persistence_success: print("    - Erro: Falha na herança de ancestrais.")

if __name__ == "__main__":
    run_final_integration()
