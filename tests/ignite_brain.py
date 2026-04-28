import torch
import os
import time
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
    
    # Queries de Alta Densidade para forçar evolução
    evolution_queries = [
        "Defina a relação entre entropia e informação na termodinâmica de buracos negros.",
        "Como a geometria toroidal otimiza o fluxo de tensores em redes neurais recursivas?",
        "Explique o conceito de Conatus em Spinoza aplicado à auto-preservação de agentes de IA.",
        "Descreva o funcionamento de um sistema Mixture-of-Experts com roteamento darwiniano.",
        "Qual a importância da soberania de dados em infraestruturas de AGI descentralizadas?",
        "Sintetize a relação entre o Logos e a estrutura matemática do universo.",
        "Como a Ghost Mesh permite a computação distribuída sem servidores centrais?",
        "Explique a função da Lattice Cryptography na proteção de linhagens de inteligência.",
        "Descreva o processo de mitose fractal em experts de alto desempenho.",
        "O que caracteriza o estado de Zero Entropia em um sistema autopoiético?"
    ]
    
    print(f"🧠 Treinando em {device} por {cycles} ciclos de evolução...")
    
    for i in range(cycles):
        print(f"\n[CICLO {i+1}/{cycles}]")
        for query in evolution_queries:
            try:
                # O forward pass aciona DataHunger, Roteamento, Processamento e Autocrítica
                result = agi.forward(query, "digerido", tokenizer)
                
                # Forçar aumento de Conatus para acelerar mitose/evolução
                for expert in agi.core.moe.experts:
                    expert.conatus.data += 0.05 
                
                print(f"✓ Processado: {query[:50]}... | Winner: Expert {result['winner_expert']} | Entropia: {result['entropy']:.3f}")
            except Exception as e:
                print(f"⚠️ Erro no processamento: {e}")
        
        # Salvar estado parcial
        agi.save_state()
        
    print("\n✨ Maturação Concluída. O 'cérebro' foi consolidado em ancestry/experts.pt")

if __name__ == "__main__":
    run_ignition(cycles=3) # 3 ciclos intensos para gerar uma linhagem inicial sólida
