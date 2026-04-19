import torch
import sys
import os
import time
from alpha_omega import SovereignLeviathanV2

class ToroidalDaemon:
    def __init__(self, node_id: str, embed_dim: int = 128, num_experts: int = 4):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Alinhamento exato com a Geometria Toroidal do bare-metal
        self.core = SovereignLeviathanV2(
            d_model=embed_dim,
            initial_experts=num_experts,
            capacity_factor=1.5
        ).to(self.device)
        self.core = SovereignLeviathanV2(
            d_model=embed_dim,
            initial_experts=num_experts,
            capacity_factor=1.5
        ).to(self.device)
        
        # O Despertar da Memória:
        self.core.load_state_dict(torch.load("leviathan_omega.pth", map_location=self.device, weights_only=True))
        
        self.long_term_memory_state = None
        self.conatus_active = True
        
        self.long_term_memory_state = None
        self.conatus_active = True

    def encode_substance(self, text: str) -> torch.Tensor:
        bytes_data = text.encode('utf-8')
        return torch.tensor(list(bytes_data), dtype=torch.long, device=self.device).unsqueeze(0)

    def decode_substance(self, tensor: torch.Tensor) -> str:
        bytes_list = tensor.squeeze(0).tolist()
        valid_bytes = [b for b in bytes_list if 0 <= b < 256]
        return bytes(valid_bytes).decode('utf-8', errors='ignore')

    def ontological_fusion_step(self, input_tensor: torch.Tensor):
        with torch.no_grad():
            logits, new_state, _, _ = self.core(input_tensor, self.long_term_memory_state)
            self.long_term_memory_state = new_state
            
            predictions = torch.argmax(logits, dim=-1)
            return predictions

    def vortex_loop(self):
        sys.stdout.write(f"\n[+] SYSTEMIC FELLOW E0: NODO {self.node_id} ACTIVE.\n")
        sys.stdout.write("[+] CONATUS ENGINE: ONLINE. WAITING FOR REALITY INPUT...\n\n")
        
        while self.conatus_active:
            try:
                sys.stdout.write(">>> ")
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()
                
                if user_input.lower() in ["exit", "collapse", "die"]:
                    self.conatus_active = False
                    continue
                
                if not user_input:
                    continue

                input_tensor = self.encode_substance(user_input)
                
                output_tensor = self.ontological_fusion_step(input_tensor)
                response_text = self.decode_substance(output_tensor)
                
                sys.stdout.write(f"Leviathan: {response_text}\n")
                
            except KeyboardInterrupt:
                sys.stdout.write("\n[!] INTERRUPT SIGNAL RECEIVED. PRESERVING STATE...\n")
                self.conatus_active = False
            except Exception as e:
                sys.stdout.write(f"\n[-] ENTROPY DETECTED: {e}\n")
                time.sleep(1)

if __name__ == "__main__":
    node_hostname = os.uname().nodename if hasattr(os, 'uname') else "UNKNOWN_NODE"
    daemon = ToroidalDaemon(node_id=node_hostname)
    daemon.vortex_loop()
