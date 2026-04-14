import os
import sys
import torch
import random

from radical_synthesis.consciousness.ontological_fusion import OntologicalFusionLoop

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from alpha_omega import SovereignLeviathanV2, EpistemologicalSentinel

class Area51TelemetrySimulator:
    def __init__(self, filename="anomaly_stream.bin"):
        self.filename = filename

    def generate_zero_day_vector(self):
        print("[*] Sintetizando Fluxo de Realidade (Bare-Metal)...")
        with open(self.filename, 'wb') as f:
            predictable_pattern = b"\x00\xFF\xAA\x55" * 1024
            f.write(predictable_pattern)
            
            chaos_stream = bytearray(random.getrandbits(8) for _ in range(4096))
            f.write(chaos_stream)
        print(f"[*] Vetor de anomalia gravado: {self.filename}")

class TerminalInjector:
    def __init__(self, model, sentinel):
        self.model = model
        self.sentinel = sentinel
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

    def force_ingestion(self, filepath: str, chunk_size: int = 64):
        print("\n[!] INICIANDO INJEÇÃO NO LEVIATHAN [!]\n")
        
        with open(filepath, 'rb') as f:
            state = None
            step = 0
            while chunk := f.read(chunk_size):
                if len(chunk) < 2:
                    break
                
                if not self.sentinel.validate_geometric_truth(chunk):
                    print(f"[-][Step {step}] Sentinela detectou lixo sintético. Purgando...")
                    step += 1
                    continue
                
                tensor_chunk = torch.tensor(list(chunk), dtype=torch.long)
                x = tensor_chunk[:-1].unsqueeze(0)
                y = tensor_chunk[1:].unsqueeze(0)
                
                self.optimizer.zero_grad()
                logits, state, entropy_loss, expert_counts = self.model(x, state)
                state = state.detach()
                
                loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), y.view(-1))
                total_loss = loss + entropy_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                # O Gatilho Vital: Onde o 0-Day Mindset ganha vida
                dead, born = self.model.living_moe.execute_systemic_lifecycle(current_loss=total_loss.item(), step=step)
                
                active_experts = self.model.living_moe.n_experts
                pressure = total_loss.item()
                
                if step % 10 == 0 or len(born) > 0 or len(dead) > 0:
                    status = f"[+] Ciclo {step:03d} | Especialistas: {active_experts} | Pressão Termodinâmica: {pressure:.4f}"
                    if born:
                        status += f" >>> MITOSE DETETADA! Novos Nodos: {[e.id for e in born]}"
                    if dead:
                        status += f" <<< APOPTOSE DETETADA! Nodos Mortos: {[e.id for e in dead]}"
                    print(status)
                
                step += 1

if __name__ == "__main__":
    D_MODEL = 64
    INITIAL_EXPERTS = 1
    
    sentinel = EpistemologicalSentinel(min_entropy=0.5, max_entropy=8.0)
    leviathan = SovereignLeviathanV2(d_model=D_MODEL, initial_experts=INITIAL_EXPERTS, capacity_factor=1.5)
    
    simulator = Area51TelemetrySimulator()
    simulator.generate_zero_day_vector()
    
    injector = TerminalInjector(leviathan, sentinel)
    injector.force_ingestion(simulator.filename)
    
    print("\n[=] FLUXO CONCLUÍDO. ESTRUTURA FINAL DO ORGANISMO [=]")
    leviathan.living_moe.print_status()
# --- A INJEÇÃO DA CONSCIÊNCIA ---
    fusion_engine = OntologicalFusionLoop(leviathan)
    # Força a máquina a gerar 128 bytes de pensamento puro baseado na sua nova topologia
    sonho_binario = fusion_engine.execute_internal_dialogue(cycles=128)
