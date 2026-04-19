"""
AGI CLI: Interface de Linha de Comando para Super Inteligência Generalista
Modo interativo com suporte a múltiplos domínios de conhecimento
"""

import torch
import os
import sys
import json
from agi_core import AGICore
from daemon_agi import OmegaTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGERIDO_PATH = os.path.join(BASE_DIR, "digerido")


class AGICLI:
    """Interface CLI para AGI"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = OmegaTokenizer()
        
        # Inicializar AGI Core
        v_size = max(self.tokenizer.vocab.keys()) + 1
        self.agi = AGICore(
            vocab_size=v_size,
            d_model=512,
            num_experts=8,
            device=str(self.device)
        )
        
        self.session_history = []
        self.mode = "general"  # general, technical, philosophical
    
    def print_header(self):
        """Imprime cabeçalho da AGI"""
        print("\n" + "="*80)
        print("🌀 OUROBOROSMOE - AGI GENERALISTA v7.0 🌀")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"d_model: 512 | num_experts: 8 | Retina: ATIVA")
        print(f"Modo: {self.mode.upper()}")
        print("="*80 + "\n")
    
    def print_help(self):
        """Imprime ajuda"""
        help_text = """
COMANDOS:
  /help          - Mostra esta ajuda
  /mode <tipo>   - Muda modo (general, technical, philosophical)
  /stats         - Mostra estatísticas da AGI
  /genealogy     - Mostra genealogia de experts
  /memory        - Mostra memórias armazenadas
  /clear         - Limpa histórico de sessão
  /exit          - Sai da AGI
  
MODO GERAL (default):
  - Respostas balanceadas entre dados técnicos e contexto
  
MODO TÉCNICO:
  - Prioriza dados matemáticos e técnicos
  - Minimiza alucinação filosófica
  
MODO FILOSÓFICO:
  - Integra Logos e Codex
  - Usa dados técnicos como base para síntese
"""
        print(help_text)
    
    def set_mode(self, mode: str):
        """Muda modo de operação"""
        if mode in ["general", "technical", "philosophical"]:
            self.mode = mode
            print(f"✓ Modo alterado para: {mode.upper()}")
        else:
            print(f"✗ Modo inválido. Use: general, technical, philosophical")
    
    def show_stats(self):
        """Mostra estatísticas"""
        stats = self.agi.get_stats()
        print("\n📊 ESTATÍSTICAS DA AGI:")
        print(f"  d_model: {stats['d_model']}")
        print(f"  num_experts: {stats['num_experts']}")
        print(f"  memory_size: {stats['memory_size']}")
        print(f"  genealogy_size: {stats['genealogy_size']}")
        print(f"  context_buffer: {stats['context_buffer_size']}\n")
    
    def show_genealogy(self):
        """Mostra genealogia de experts"""
        genealogy = self.agi.memory.get_genealogy_tree()
        print("\n🧬 GENEALOGIA DE EXPERTS:")
        for expert_id, info in genealogy.items():
            print(f"  Expert {expert_id}:")
            print(f"    - Generation: {info['generation']}")
            print(f"    - Memories: {info['memories_count']}")
            print(f"    - Children: {len(info['children'])}")
        print()
    
    def show_memory(self):
        """Mostra memórias armazenadas"""
        memories = self.agi.memory.memories
        print(f"\n💾 MEMÓRIAS ARMAZENADAS ({len(memories)}):")
        for i, mem in enumerate(memories[-5:]):  # Últimas 5
            print(f"  [{i}] Expert {mem['expert_id']} (Gen {mem['generation']})")
            print(f"      Content: {mem['content'][:100]}...")
        print()
    
    def process_query(self, query: str) -> str:
        """Processa query através da AGI"""
        try:
            result = self.agi.forward(query, DIGERIDO_PATH, self.tokenizer)
            
            response = result['response']
            technical_data = result['technical_data']
            confidence = result['confidence']
            
            # Formatar saída
            output = f"\n🧠 RESPOSTA:\n{response}\n"
            
            if technical_data:
                output += f"\n📚 DADOS TÉCNICOS INJETADOS:\n{technical_data[:200]}...\n"
            
            output += f"\n📊 Confiança: {confidence:.2%}\n"
            
            # Armazenar no histórico
            self.session_history.append({
                'query': query,
                'response': response,
                'confidence': confidence
            })
            
            return output
        
        except Exception as e:
            return f"\n✗ ERRO: {str(e)}\n"
    
    def run(self):
        """Loop principal da CLI"""
        self.print_header()
        print("Digite /help para ver comandos disponíveis\n")
        
        while True:
            try:
                user_input = input("E0 >>> ").strip()
                
                if not user_input:
                    continue
                
                # Processar comandos
                if user_input.startswith('/'):
                    cmd = user_input.split()[0][1:]
                    args = user_input.split()[1:] if len(user_input.split()) > 1 else []
                    
                    if cmd == "help":
                        self.print_help()
                    elif cmd == "mode":
                        if args:
                            self.set_mode(args[0])
                        else:
                            print("Use: /mode <general|technical|philosophical>")
                    elif cmd == "stats":
                        self.show_stats()
                    elif cmd == "genealogy":
                        self.show_genealogy()
                    elif cmd == "memory":
                        self.show_memory()
                    elif cmd == "clear":
                        self.session_history = []
                        print("✓ Histórico limpo")
                    elif cmd == "exit":
                        print("\n🌀 Encerrando AGI... Até logo!\n")
                        break
                    else:
                        print(f"✗ Comando desconhecido: {cmd}")
                
                else:
                    # Processar query
                    output = self.process_query(user_input)
                    print(output)
            
            except KeyboardInterrupt:
                print("\n\n🌀 Encerrando AGI... Até logo!\n")
                break
            except Exception as e:
                print(f"\n✗ ERRO: {str(e)}\n")


if __name__ == "__main__":
    cli = AGICLI()
    cli.run()
