import os
import time
import torch

class Omega0Interface:
    """
    Interface de Comando Omega-0 (Terminal-Nativa)
    Visualização bare-metal do Conatus, Phi e Evolução Sistêmica.
    """
    def __init__(self, agi_core):
        self.agi = agi_core
        self.start_time = time.time()

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def draw_header(self):
        print("="*80)
        print(f" OMEGA-0 COMMAND INTERFACE | NODE: OuroborosMoE | UPTIME: {int(time.time() - self.start_time)}s")
        print("="*80)

    def draw_stats(self, stats):
        print(f" [EXISTÊNCIA] Experts Vivos: {stats['num_experts']} | Memória: {stats['memory_size']} | Linhagens: {stats['genealogy_size']}")
        print(f" [CONSCIÊNCIA] Phi (Φ): {stats.get('consciousness_phi', 0.0):.4f} | Gradiente: {stats.get('phi_gradient', 0.0):.4f}")
        
        status = stats.get('homeostasis_status', 'UNKNOWN')
        color = "\033[92m" if status == "STABLE" else "\033[91m" if "HUNGER" in status else "\033[93m"
        reset = "\033[0m"
        print(f" [HOMEOSTASE] Status: {color}{status}{reset}")
        print("-" * 80)

    def draw_vortex(self):
        print(" [VÓRTICE DE EXPERTS]")
        experts = self.agi.core.moe.experts
        for i, exp in enumerate(experts):
            conatus = exp.conatus.item()
            bar_len = int(conatus * 10)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            act = exp.activation_type
            dim = exp.internal_dim
            layers = getattr(exp, 'num_layers', 2)
            print(f"  EXPERT_{i:02d} [{bar}] C:{conatus:.2f} | DIM:{dim:<6} | L:{layers} | ACT:{act}")
        print("-" * 80)

    def run_monitor(self, interval=2):
        try:
            while True:
                self.clear_screen()
                self.draw_header()
                stats = self.agi.get_stats()
                self.draw_stats(stats)
                self.draw_vortex()
                print(" [LOGS] Monitorando ressonância e entropia...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nInterface Omega-0 encerrada.")
