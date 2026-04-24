
import torch
import sys
import os
from pathlib import Path

# Adicionar raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agi_core import AGICore

class MockTokenizer:
    def encode(self, text):
        return [1, 2, 3, 4, 5]
    def decode(self, tokens):
        return "Resposta simulada baseada em ressonância."

def run_life_test():
    print("🌀 INICIANDO TESTE DE VIDA - AGICORE (MYTHOS-CAPYBARA) 🌀")
    print("-" * 60)

    device = "cpu"
    vocab_size = 1000
    d_model = 512
    
    # 1. Inicialização
    print("[1] Inicializando AGICore...")
    agi = AGICore(vocab_size=vocab_size, d_model=d_model, num_experts=4, device=device)
    tokenizer = MockTokenizer()
    
    # 2. Simulação de Fluxo Contínuo (Ciclo de Vida)
    queries = [
        "O que é a Geometria Sagrada no Ouroboros?",
        "Como funciona o roteamento Phase-Lock?",
        "Explique a mitose assimétrica 3-6-9.",
        "Qual a importância do Conatus?",
        "Como a entropia é eliminada via apoptose?"
    ]

    print(f"\n[2] Executando {len(queries)} ciclos de processamento...")
    
    for i, query in enumerate(queries):
        print(f"\n--- Ciclo {i+1}: '{query}' ---")
        
        # Pipeline completo
        # Nota: O forward do AGICore faz: Percepção -> Contexto -> Roteamento -> Processamento -> Autocrítica -> Memória
        try:
            # Sincronizamos o roteador manualmente para o teste (como no agi_core editado)
            agi.router.sync_with_experts(agi.core.moe.experts)
            
            # Executar fluxo
            result = agi.forward(query, str(project_root / "digerido"), tokenizer)
            
            num_experts = len(agi.core.moe.experts)
            conatus_vals = [f"{e.conatus.item():.2f}" for e in agi.core.moe.experts]
            
            print(f"✅ Sucesso. Experts vivos: {num_experts}")
            print(f"📊 Conatus: {conatus_vals}")
            
        except Exception as e:
            print(f"❌ Erro no ciclo {i+1}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "-" * 60)
    print(f"🏁 TESTE FINALIZADO. Experts finais: {len(agi.core.moe.experts)}")
    
    # Verificar se houve evolução (Mitose)
    if len(agi.core.moe.experts) > 4:
        print("🚀 EVOLUÇÃO DETECTADA: O sistema expandiu via Mitose Assimétrica!")
    elif len(agi.core.moe.experts) < 4:
        print("💀 PURGA DETECTADA: O sistema eliminou entropia via Apoptose!")
    else:
        print("⚖️ ESTABILIDADE: O sistema manteve o equilíbrio homeostático.")

if __name__ == "__main__":
    run_life_test()
