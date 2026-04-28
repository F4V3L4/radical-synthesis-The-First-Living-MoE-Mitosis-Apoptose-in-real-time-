"""
Corrigir _apply_external_routing para lidar com diferentes shapes de expert_indices
"""

with open('alpha_omega.py', 'r') as f:
    content = f.read()

# Novo _apply_external_routing que lida com (B, top_k) e (B, T, top_k)
new_method = '''    def _apply_external_routing(self, x, expert_indices, expert_weights):
        """
        Aplica roteamento externo (DarwinianRouter) aos experts
        Respeita os índices e pesos vindos do AGICore
        
        Suporta shapes:
        - expert_indices: (B, top_k) ou (B, T, top_k)
        - expert_weights: (B, top_k) ou (B, T, top_k)
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)
        
        # Normalizar shapes para (B, T, top_k)
        if expert_indices.dim() == 2:
            # (B, top_k) -> (B, 1, top_k) -> broadcast para (B, T, top_k)
            expert_indices = expert_indices.unsqueeze(1).expand(B, T, -1)
            expert_weights = expert_weights.unsqueeze(1).expand(B, T, -1)
        
        top_k = expert_indices.shape[-1]
        
        # Processar cada expert
        for k in range(top_k):
            expert_idx = expert_indices[:, :, k]  # (B, T)
            expert_weight = expert_weights[:, :, k]  # (B, T)
            
            # Processar cada batch e timestep
            for b in range(B):
                for t in range(T):
                    exp_id = expert_idx[b, t].item() if isinstance(expert_idx[b, t], torch.Tensor) else int(expert_idx[b, t])
                    if exp_id < len(self.moe.experts):
                        expert_out = self.moe.experts[exp_id](x[b, t].unsqueeze(0))
                        weight = expert_weight[b, t].item() if isinstance(expert_weight[b, t], torch.Tensor) else float(expert_weight[b, t])
                        out[b, t] += weight * expert_out.squeeze(0)
        
        # Aplicar acoplamento com Constante de Estrutura Fina
        return self.moe.coupling(x + out)'''

# Encontrar e substituir _apply_external_routing
old_start = content.find('    def _apply_external_routing(self, x, expert_indices, expert_weights):')
old_end = content.find('        return self.moe.coupling(x + out)') + len('        return self.moe.coupling(x + out)')

if old_start > 0 and old_end > old_start:
    content = content[:old_start] + new_method + '\n' + content[old_end:]
    
    with open('alpha_omega.py', 'w') as f:
        f.write(content)
    
    print("✅ _apply_external_routing corrigido para lidar com diferentes shapes")
else:
    print("❌ Não foi possível encontrar _apply_external_routing")
