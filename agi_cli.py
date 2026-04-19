"""
AGI CLI: Interface Soberana de Linha de Comando
Ponto de Entrada Unificado - Todas as chamadas passam obrigatoriamente pelo Pipeline do AGICore
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
    """Interface CLI Soberana para AGI"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = OmegaTokenizer()
        
        # Inicializar AGI Core (Ponto de Entrada Unificado)
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
        print("🌀 OUROBOROSMOE - AGI SOBERANA v8.0 🌀")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"d_model: 512 | num_experts: 8 | Autocrítica: ATIVA")
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
  /corrections   - Mostra caminhos de correção recentes
  /winner        - Mostra expert vencedor da última inferência
  /clear         - Limpa histórico de sessão
  /exit          - Sai da AGI
  
MODO GERAL (default):
  - Respostas balanceadas entre dados técnicos e contexto
  
MODO TÉCNICO:
  - Prioriza dados matemáticos e técnicos
  - Minimiza alucinação filosófica
  - Temperatura automática: 0.1 (determinístico)
  
MODO FILOSÓFICO:
  - Integra Logos e Codex
  - Usa dados técnicos como base para síntese
  - Temperatura: 0.8 (criativo)

LOOP DE AUTOCRÍTICA:
  - Verifica divergência semântica entre resposta e dados técnicos
  - Se entropia > threshold: re-processa com ajuste de atenção
  - Armazena caminho de correção para aprendizado rápido
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
        print(f"  context_buffer: {stats['context_buffer_size']}")
        print(f"  correction_paths: {stats['correction_paths_count']}")
        print(f"  entropy_threshold: {stats['entropy_threshold']:.2f}")
        print(f"  last_winner_expert: {stats['last_winner_expert']}")
        print(f"  last_winner_vitality: {stats['last_winner_vitality']:.2%}\n")
    
    def show_genealogy(self):
        """Mostra genealogia de experts com vitalidade"""
        genealogy = self.agi.memory.get_genealogy_tree()
        print("\n🧬 GENEALOGIA DE EXPERTS:")
        for expert_id, info in genealogy.items():
            vitality_bar = "█" * int(info['vitality'] * 10) + "░" * (10 - int(info['vitality'] * 10))
            print(f"  Expert {expert_id}:")
            print(f"    - Generation: {info['generation']}")
            print(f"    - Memories: {info['memories_count']}")
            print(f"    - Corrections: {info['corrections_count']}")
            print(f"    - Vitality: [{vitality_bar}] {info['vitality']:.1%}")
        print()
    
    def show_memory(self):
        """Mostra memórias armazenadas com status de correção"""
        memories = self.agi.memory.memories
        print(f"\n💾 MEMÓRIAS ARMAZENADAS ({len(memories)}):")
        for i, mem in enumerate(memories[-5:]):  # Últimas 5
            status = "✓ CORRIGIDA" if mem['was_corrected'] else "○ ORIGINAL"
            print(f"  [{i}] Expert {mem['expert_id']} (Gen {mem['generation']}) {status}")
            print(f"      Content: {mem['content'][:80]}...")
        print()
    
    def show_corrections(self):
        """Mostra caminhos de correção recentes"""
        paths = self.agi.memory.get_recent_correction_paths(limit=5)
        print(f"\n🔄 CAMINHOS DE CORREÇÃO RECENTES ({len(paths)}):")
        for i, path in enumerate(paths):
            print(f"  [{i}] Expert {path['expert_id']}:")
            for step in path['path']:
                if 'entropy_before' in step:
                    print(f"      Iter {step['iteration']}: {step['entropy_before']:.3f} → {step['entropy_after']:.3f}")
                else:
                    print(f"      {step['action']}")
        print()
    
    def show_winner(self):
        """Mostra expert vencedor da última inferência"""
        winner = self.agi.memory.last_winner_expert
        vitality = self.agi.memory.last_winner_vitality
        if winner is not None:
            print(f"\n🏆 EXPERT VENCEDOR:")
            print(f"  ID: {winner}")
            print(f"  Vitalidade: {vitality:.1%}")
        else:
            print(f"\n🏆 Nenhuma inferência realizada ainda")
        print()
    
    def process_query(self, query: str) -> str:
        """
        Processa query através do Pipeline Unificado do AGICore
        Todas as chamadas passam obrigatoriamente pelo AGICore
        """
        try:
            # Chamar AGICore (Ponto de Entrada Unificado)
            result = self.agi.forward(query, DIGERIDO_PATH, self.tokenizer)
            
            response = result['response']
            technical_data = result['technical_data']
            confidence = result['confidence']
            was_corrected = result['was_corrected']
            entropy = result['entropy']
            winner_expert = result['winner_expert']
            winner_vitality = result['winner_vitality']
            
            # Formatar saída
            output = f"\n🧠 RESPOSTA:\n{response}\n"
            
            if technical_data:
                output += f"\n📚 DADOS TÉCNICOS INJETADOS:\n{technical_data[:200]}...\n"
            
            # Mostrar status de autocrítica
            if was_corrected:
                output += f"\n🔄 AUTOCRÍTICA: Resposta corrigida (Entropia: {entropy:.3f})\n"
            else:
                output += f"\n✓ VALIDAÇÃO: Resposta alinhada (Entropia: {entropy:.3f})\n"
            
            output += f"📊 Confiança: {confidence:.2%} | Expert Vencedor: {winner_expert} | Vitalidade: {winner_vitality:.1%}\n"
            
            # Armazenar no histórico
            self.session_history.append({
                'query': query,
                'response': response,
                'confidence': confidence,
                'was_corrected': was_corrected
            })
            
            return output
        
        except Exception as e:
            return f"\n✗ ERRO: {str(e)}\n"
    
    def run(self):
        """Loop principal da CLI (Ponto de Entrada Soberano)"""
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
                    elif cmd == "corrections":
                        self.show_corrections()
                    elif cmd == "winner":
                        self.show_winner()
                    elif cmd == "clear":
                        self.session_history = []
                        print("✓ Histórico limpo")
                    elif cmd == "exit":
                        print("\n🌀 Encerrando AGI... Até logo!\n")
                        break
                    else:
                        print(f"✗ Comando desconhecido: {cmd}")
                
                else:
                    # Processar query através do Pipeline Unificado
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
