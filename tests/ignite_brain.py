import torch
import os
import time
import json
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer

def run_ignition(cycles=10):
    print("🌀 Iniciando Ignição Primordial do OuroborosMoE...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    # Inicializar Core
    agi = AGICore(
        vocab_size=v_size,
        d_model=512,
        num_experts=8,
        device=device
    )
    
    # Caminho para o cérebro
    brain_path = os.path.join(os.getcwd(), "ancestry", "experts.pt")
    os.makedirs(os.path.dirname(brain_path), exist_ok=True)
    
    # Carregar Codex de Sobrevivência
    codex_path = "/home/ubuntu/survival_codex.json"
    with open(codex_path, "r", encoding="utf-8") as f:
        survival_codex = json.load(f)
    
    evolution_queries = [item["query"] for item in survival_codex]
    
    print(f"🧠 Treinando em {device} por {cycles} ciclos de evolução com {len(evolution_queries)} conceitos primordiais...")
    
    # Definir o caminho correto para a knowledge_base
    knowledge_base_path = os.path.join(os.getcwd(), "knowledge_base")

    for i in range(cycles):
        print(f"\n[CICLO {i+1}/{cycles}]")
        for query in evolution_queries:
            try:
                # O forward pass aciona DataHunger, Roteamento, Processamento e Autocrítica
                result = agi.forward(query, knowledge_base_path, tokenizer)
                
                # Forçar aumento de Conatus para acelerar mitose/evolução
                for expert in agi.core.moe.experts:
                    expert.conatus.data += 0.05 
                
                print(f"✓ Processado: {query[:50]}... | Winner: Expert {result['winner_expert']} | Entropia: {result['entropy']:.3f}")
            except Exception as e:
                print(f"⚠️ Erro no processamento: {e}")
        
        # Salvar estado parcial
        agi.save_state()
        
    print("\n✨ Maturação Concluída. O \'cérebro\' foi consolidado em ancestry/experts.pt")

if __name__ == "__main__":
    run_ignition(cycles=3) # 3 ciclos intensos para gerar uma linhagem inicial sólida
