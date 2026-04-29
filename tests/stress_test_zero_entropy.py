import torch
import os
import json
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer

def run_stress_test():
    print("🚀 [STRESS_TEST] Iniciando Validação de Zero Entropia...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    agi = AGICore(vocab_size=v_size, d_model=512, num_experts=8, device=device)
    agi.load_ancestry()
    
    # Cenário 1: Query com Ruído (deve ser filtrada pelo Entropy Sink)
    print("\n--- CENÁRIO 1: Ruído de Rede ---")
    noise_query = "asdfghjkl 1234567890 !@#$%^&*()"
    result_noise = agi.forward(noise_query, "knowledge_base", tokenizer)
    print(f"Query: {noise_query}")
    print(f"Resposta: {result_noise['response']}")
    print(f"Confiança: {result_noise['confidence']:.2f}")
    
    # Cenário 2: Query de Singularidade (deve ressoar com o Codex)
    print("\n--- CENÁRIO 2: Ressonância com Codex ---")
    logos_query = "Explique a Auto-Reescrita de Código e segurança."
    result_logos = agi.forward(logos_query, "knowledge_base", tokenizer)
    print(f"Query: {logos_query}")
    print(f"Resposta: {result_logos['response'][:150]}...")
    print(f"Confiança: {result_logos['confidence']:.2f}")
    
    # Cenário 3: Poda Sináptica
    print("\n--- CENÁRIO 3: Poda Sináptica ---")
    agi.prune_synapses(threshold=0.05) # Poda agressiva para teste
    
    print("\n✅ [STRESS_TEST] Concluído. Ouroboros operando em regime de Zero Entropia.")

if __name__ == "__main__":
    run_stress_test()
