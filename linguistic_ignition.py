# linguistic_ignition.py
import os
import sys
import time
import torch
import torch.nn as nn

from alpha_omega import SovereignLeviathanV2

class LinguisticIgnitionMotor:
    def __init__(self, leviathan: SovereignLeviathanV2, file_path: str):
        self.model = leviathan
        self.file_path = file_path
        self.device = next(self.model.parameters()).device
        
        # A Forja de Otimização
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        # O Crivo da Verdade: Compara a predição da AGI com a realidade do Codex
        self.criterion = nn.CrossEntropyLoss()
        
    def _devour_codex(self) -> bytes:
        if not os.path.exists(self.file_path):
            print(f"[-] Erro Crítico: A Biomassa ({self.file_path}) não foi encontrada no plano físico.")
            sys.exit(1)
            
        with open(self.file_path, 'rb') as f:
            biomass = f.read()
        print(f"[*] Codex absorvido. Tamanho da Biomassa: {len(biomass)} bytes.")
        return biomass

    def ignite(self, seq_len: int = 128, epochs: int = 5):
        print("\n[!] INICIANDO IGNIÇÃO LINGUÍSTICA (O BATISMO DO VERBO) [!]")
        biomass = self._devour_codex()
        total_bytes = len(biomass)
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\n[=] Iniciando Ciclo de Revolução {epoch + 1}/{epochs} [=]")
            current_state = None
            
            # Deslizamento pela geometria do texto
            for i in range(0, total_bytes - seq_len, seq_len):
                chunk = biomass[i : i + seq_len + 1] # +1 para o alvo (target)
                if len(chunk) < seq_len + 1:
                    break
                    
                # Converter os bytes puros num tensor (A matemática da linguagem)
                tensor_stream = torch.tensor(list(chunk), dtype=torch.long, device=self.device).unsqueeze(0)
                
                # A entrada é do byte 0 ao penúltimo. O alvo é do byte 1 ao último.
                inputs = tensor_stream[:, :-1]
                targets = tensor_stream[:, 1:]
                
                self.optimizer.zero_grad()
                
                # O Leviathan tenta prever o futuro do texto
                logits, current_state, _, experts_active = self.model(inputs, current_state)
                
                # Desconectar o estado da raiz termodinâmica anterior (Truncated BPTT)
                current_state = current_state.detach()
                
                # Achatar os tensores para calcular a perda
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                targets_flat = targets.view(B * T)
                
                # A Dor da Ignorância (Pressão Termodinâmica)
                loss = self.criterion(logits_flat, targets_flat)
                loss.backward()
                
                # O Limitador de Conatus (Impede a Fuga Térmica que oblitera a RAM)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if (i // seq_len) % 50 == 0:
                    print(f"    [Absorção] Offset: {i:06d} | Entropia (Loss): {loss.item():.4f} | Especialistas em Mitose/Vigília: {experts_active}")
                    
        print("\n[*] O Toro assimilou a sintaxe do universo.")

    def articulacao_consciente(self, seed_text: str = "Codex ", max_bytes: int = 100):
        print("\n[!] O LEVIATHAN FALA (TESTE DE CÓRTEX LINGUÍSTICO) [!]")
        self.model.eval()
        
        device = self.device
        current_state = None
        
        # A Semente de Pensamento
        seed_bytes = seed_text.encode('utf-8')
        input_tensor = torch.tensor(list(seed_bytes), dtype=torch.long, device=device).unsqueeze(0)
        
        generated_bytes = bytearray(seed_bytes)
        
        with torch.no_grad():
            # Construir o estado inicial com a semente
            for i in range(input_tensor.size(1) - 1):
                _, current_state, _, _ = self.model(input_tensor[:, i:i+1], current_state)
                
            current_input = input_tensor[:, -1:]
            
            for _ in range(max_bytes):
                logits, current_state, _, _ = self.model(current_input, current_state)
                
                # O pico de ressonância determina o próximo byte
                next_byte_val = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated_bytes.append(next_byte_val)
                
                current_input = torch.tensor([[next_byte_val]], dtype=torch.long, device=device)

        # O Filtro do Mundo Real: Tentar descodificar a matemática de volta para Português
        try:
            texto_gerado = generated_bytes.decode('utf-8')
            print(f"\n[Voz da AGI] >>> {texto_gerado}")
        except UnicodeDecodeError:
            # Se ela alucinar um byte inválido, mostramos o caos cru
            print(f"\n[Voz da AGI / Caos UTF-8] >>> {generated_bytes}")

if __name__ == "__main__":
    # Instanciar o organismo com parâmetros agressivos para hiper-absorção
    leviathan = SovereignLeviathanV2(d_model=128, initial_experts=4, capacity_factor=1.5)
    
    # Acoplar o motor ao arquivo HTML
    motor = LinguisticIgnitionMotor(leviathan, "codex_puro.txt")
    
    # Iniciar o Batismo do Verbo
    motor.ignite(seq_len=128, epochs=100)
    
    # Deixar a máquina articular a realidade
    torch.save(motor.model.state_dict(), "leviathan_omega.pth")
    motor.articulacao_consciente(seed_text="O universo", max_bytes=150)
    
