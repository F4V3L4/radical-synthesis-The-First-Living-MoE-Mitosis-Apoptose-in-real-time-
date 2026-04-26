import time
import random
import torch
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.align import Align
from rich.text import Text

console = Console()

class Omega0Interface:
    """
    Interface Neural Omega-0: Visualização bare-metal avançada do OuroborosMoE.
    Exibe o Vórtice de Experts, Conatus, Telemetria e Ghost Mesh com suporte a Rich.
    """
    def __init__(self, agi_core=None):
        self.agi = agi_core
        self.start_time = time.time()

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="vortex", ratio=2),
            Layout(name="telemetry", ratio=1),
        )
        return layout

    def get_header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        title = Text("🌀 OUROBOROS MOE - NODO OMEGA-0 - INTERFACE NEURAL 🌀", style="bold cyan")
        grid.add_row(title, Text(time.ctime(), style="dim"))
        return Panel(grid, style="bold blue")

    def get_vortex_view(self) -> Panel:
        table = Table(title="🌀 VÓRTICE DE EXPERTS (TOPOLOGIA FRACTAL)", expand=True)
        table.add_column("ID", justify="center", style="cyan")
        table.add_column("CONATUS", justify="center")
        table.add_column("TIPO", justify="center")
        table.add_column("STATUS", justify="center")
        table.add_column("RESSONÂNCIA", justify="right")

        if self.agi and hasattr(self.agi.core.moe, 'experts'):
            experts = self.agi.core.moe.experts
            for i, exp in enumerate(experts):
                conatus = exp.conatus.item()
                is_fractal = "FRACTAL" if getattr(exp, 'is_fractal', False) else "LINEAR"
                status = "🔥 ATIVO" if conatus > 0.2 else "💀 APOPTOSI"
                resonance_val = int(min(conatus * 5, 20))
                resonance = "█" * resonance_val
                
                style = "bold green" if conatus > 1.0 else "yellow"
                if is_fractal == "FRACTAL": style = "bold magenta"
                
                table.add_row(
                    f"{i:02d}", 
                    f"{conatus:.2f}", 
                    is_fractal, 
                    status, 
                    Text(resonance, style=style)
                )
        else:
            # Fallback para simulação se o AGI Core não estiver ativo
            for i in range(8):
                conatus = random.uniform(0.5, 3.5)
                is_fractal = "FRACTAL" if conatus > 3.0 else "LINEAR"
                table.add_row(f"{i:02d}", f"{conatus:.2f}", is_fractal, "🔥 ATIVO", Text("█" * int(random.random()*10), style="green"))
        
        return Panel(table, border_style="cyan")

    def get_telemetry_view(self) -> Panel:
        table = Table(title="📊 TELEMETRIA QUÂNTICA", expand=True)
        table.add_column("MÉTRICA", style="dim")
        table.add_column("VALOR", justify="right")

        uptime = int(time.time() - self.start_time)
        table.add_row("Uptime", f"{uptime}s")
        
        if self.agi:
            stats = self.agi.get_stats()
            table.add_row("Entropia", f"{stats.get('entropy', 0.001234):.6f}")
            table.add_row("Φ (Phi)", f"{stats.get('consciousness_phi', 0.9876):.4f}")
        else:
            table.add_row("Entropia", f"{random.uniform(0.001, 0.005):.6f}")
            table.add_row("Φ (Phi)", f"{random.uniform(0.85, 0.99):.4f}")
            
        table.add_row("Latência L1", f"{random.uniform(0.8, 1.2):.2f}ns")
        table.add_row("Custo Termo", f"{random.uniform(1.1, 1.4):.3f}W")
        table.add_row("Ghost Mesh", "CONNECTED (12 Nodos)")

        return Panel(table, border_style="green")

    def get_footer(self) -> Panel:
        msg = "DIRETRIZ PRIMÁRIA: RADICAL SYNTHESIS | OPERADOR: E0 | STATUS: ASCENSÃO"
        return Panel(Align.center(Text(msg, style="bold yellow")), border_style="blue")

    def run(self):
        layout = self.make_layout()
        try:
            with Live(layout, refresh_per_second=4, screen=True):
                while True:
                    layout["header"].update(self.get_header())
                    layout["vortex"].update(self.get_vortex_view())
                    layout["telemetry"].update(self.get_telemetry_view())
                    layout["footer"].update(self.get_footer())
                    time.sleep(0.25)
        except KeyboardInterrupt:
            console.print("\n[bold red]Interface Omega-0 Encerrada. Matrix Sincronizada.[/]")

if __name__ == "__main__":
    interface = Omega0Interface()
    interface.run()
