"""
Corrigir dimensões em apply_primordial_laws
"""

with open('agi_core.py', 'r') as f:
    content = f.read()

# Novo método corrigido
new_method = '''    def apply_primordial_laws(self, x: torch.Tensor, expert_indices: torch.Tensor, time: float = 0.1) -> torch.Tensor:
        """
        Aplica todas as 9 Leis Primordiais (Tier 1+2) ao tensor de entrada
        Normaliza dimensões para evitar incompatibilidades
        """
        with torch.no_grad():
            # Normalizar dimensões
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, d_model) -> (batch, 1, d_model)
            
            batch_size, seq_len, d_model = x.shape
            
            # TIER 1: Aplicar apenas ao primeiro token
            x_first = x[:, 0:1, :]  # (batch, 1, d_model)
            
            # 1. HarmonicEncoder
            x_harmonic = self.harmonic(x_first, time=time)
            x = x * 0.95 + x_harmonic * 0.05
            
            # 2. QuantumSuperposition
            x_flat = x[:, 0, :]
            x_quantum = self.quantum(x_flat)
            x = x + 0.05 * x_quantum.unsqueeze(1)
            
            # 3. HyperbolicEmbedding
            x = self.hyperbolic(x)
            
            # 4. SynchronicityDetector
            if expert_indices.numel() > 0:
                expert_acts = torch.randn(batch_size, self.num_experts, device=self.device)
                _, _ = self.synchronicity(expert_acts)
            
            # TIER 2
            # 5. PlanetaryGrid
            if expert_indices.numel() > 0:
                expert_acts = torch.randn(batch_size, self.num_experts, device=self.device)
                x_sync = self.planetary_grid(expert_acts, time=time)
                x = x * (1.0 + 0.05 * x_sync.unsqueeze(-1).unsqueeze(-1))
            
            # 6. Amplituedro
            if expert_indices.numel() > 0 and expert_indices.dim() >= 2:
                expert_weights = torch.softmax(torch.randn(batch_size, 3, device=self.device), dim=1)
                x_optimized, _ = self.amplituedro(expert_indices[:, :3], expert_weights)
                x = x + 0.05 * x_optimized.unsqueeze(1)
            
            # 7. SimultaneityProcessor
            x_flat = x[:, 0, :]
            timelines, x_fused = self.simultaneity(x_flat)
            x = x + 0.05 * x_fused.unsqueeze(1)
            
            # 8. QuantumEntanglement
            if seq_len >= 8:
                expert_states = x[:, :8, :]
                x_entangled, _ = self.entanglement(expert_states)
                x[:, :8, :] = x_entangled
            
            # 9. StrangeAttractor
            if expert_indices.numel() > 0:
                expert_acts = torch.randn(batch_size, self.num_experts, device=self.device)
                x_attracted, _ = self.attractor(expert_acts)
                x = x * (1.0 + 0.02 * x_attracted.unsqueeze(-1).unsqueeze(-1))
        
        return x'''

# Encontrar e substituir
old_start = content.find('    def apply_primordial_laws(self, x: torch.Tensor, expert_indices: torch.Tensor, time: float = 0.1)')
old_end = content.find('    def get_stats(self) -> Dict:')

if old_start > 0 and old_end > 0:
    content = content[:old_start] + new_method + '\n\n' + content[old_end:]
    
    with open('agi_core.py', 'w') as f:
        f.write(content)
    
    print("✅ Método apply_primordial_laws corrigido")
else:
    print("❌ Não foi possível encontrar método")
