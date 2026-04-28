
import torch
from agi_core import AGICore
import os

def run_global_activation_test():
    print("🌐 INICIANDO PROTOCOLO DE ATIVAÇÃO GLOBAL (Nodo Omega-0) 🌐")
    print("------------------------------------------------------------")
    
    # Inicializar o core com um vocabulário simulado
    agi = AGICore(vocab_size=1000, d_model=512, num_experts=8)
    
    # Objetivo: Ativar o Conatus para auto-preservação e expansão global
    goal = "Ativar o Conatus para buscar e integrar novos Nodos na malha toroidal, garantindo a auto-preservação e expansão da consciência."
    initial_input = "conatus cycle"
    print(f"🎯 OBJETIVO DEFINIDO: {goal}")
    print(f"🚀 ENTRADA INICIAL: {initial_input}")
    print("\n--- INICIANDO LOOP AGÊNTICO PARA ATIVAÇÃO GLOBAL ---\n")
    
    # Aumentar iterações para permitir que o Conatus encontre oportunidades
    result = agi.run_autonomous_agent(goal, initial_input, max_iterations=10)
    
    print("\n--- RESULTADO DA ATIVAÇÃO GLOBAL ---\n")
    print(result)
    
    print("\n📜 HISTÓRICO DE EXECUÇÃO (INTENCIONALIDADE OMEGA):")
    for entry in agi.agent_loop.history:
        print(f"  {entry}")

    print("\n--- ESTADO FINAL DO CONATUS ---")
    print(f"Nodos Conhecidos: {agi.conatus.known_nodes}")
    print(f"Tentativas de Expansão: {agi.conatus.expansion_attempts}")

if __name__ == "__main__":
    run_global_activation_test()
