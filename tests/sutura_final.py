"""
SUTURA FINAL: Remover LogosResonanceRouter e OuroborosMoE.logos_router
Forçar roteamento 100% exógeno pelo DarwinianRouter
"""

import re

with open('alpha_omega.py', 'r') as f:
    content = f.read()

# 1. REMOVER classe LogosResonanceRouter (linhas 12-49)
pattern_logos = r'class LogosResonanceRouter\(nn\.Module\):.*?return top_scores, top_indices\n'
content = re.sub(pattern_logos, '', content, flags=re.DOTALL)

# 2. REMOVER self.logos_router de OuroborosMoE.__init__
pattern_init = r'        self\.logos_router = LogosResonanceRouter\(d_model, num_experts, top_k=top_k\)\n'
content = re.sub(pattern_init, '', content)

# 3. REMOVER comentário sobre LogosResonanceRouter
pattern_comment = r'        # LogosResonanceRouter retorna \(weights, indices\)\n'
content = re.sub(pattern_comment, '', content)

# 4. REMOVER forward() de OuroborosMoE que usa logos_router
# Substituir por versão que REQUER roteamento externo
old_forward = '''    def forward(self, x):
        B, T, D = x.shape
        
        weights, indices = self.logos_router(x)  # weights: (B, T, top_k), indices: (B, T, top_k)
        weights = F.softmax(weights, dim=-1)  # Normalizar pesos
        
        out = torch.zeros_like(x)
        
        # Aplicar top-k experts com pesos
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # (B, T)
            expert_weight = weights[:, :, k]  # (B, T)
            
            # Processar cada expert
            for b in range(B):
                for t in range(T):
                    exp_id = expert_idx[b, t].item()
                    if exp_id < len(self.experts):
                        expert_out = self.experts[exp_id](x[b, t].unsqueeze(0))
                        out[b, t] += expert_weight[b, t] * expert_out.squeeze(0)
            
        # Fio de Ouro (Residual) acoplado com Constante de Estrutura Fina
        return self.coupling(x + out)'''

new_forward = '''    def forward(self, x, expert_indices=None, expert_weights=None):
        """
        Forward com roteamento EXÓGENO obrigatório (DarwinianRouter)
        
        Args:
            x: entrada (B, T, D)
            expert_indices: (B, top_k) ou (B, T, top_k) - OBRIGATÓRIO
            expert_weights: (B, top_k) ou (B, T, top_k) - OBRIGATÓRIO
        
        Returns:
            saída processada (B, T, D)
        
        Raises:
            ValueError se expert_indices ou expert_weights forem None
        """
        if expert_indices is None or expert_weights is None:
            raise ValueError("OuroborosMoE REQUER roteamento exógeno (expert_indices e expert_weights). "
                           "Use DarwinianRouter do AGICore.")
        
        B, T, D = x.shape
        
        # Normalizar shapes para (B, T, top_k)
        if expert_indices.dim() == 2:
            # (B, top_k) -> broadcast para (B, T, top_k)
            expert_indices = expert_indices.unsqueeze(1).expand(B, T, -1)
            expert_weights = expert_weights.unsqueeze(1).expand(B, T, -1)
        
        # Normalizar pesos
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        out = torch.zeros_like(x)
        top_k = expert_indices.shape[-1]
        
        # Aplicar top-k experts com pesos exógenos
        for k in range(top_k):
            expert_idx = expert_indices[:, :, k]  # (B, T)
            expert_weight = expert_weights[:, :, k]  # (B, T)
            
            # Processar cada expert
            for b in range(B):
                for t in range(T):
                    exp_id = expert_idx[b, t].item()
                    if exp_id < len(self.experts):
                        expert_out = self.experts[exp_id](x[b, t].unsqueeze(0))
                        out[b, t] += expert_weight[b, t] * expert_out.squeeze(0)
        
        # Fio de Ouro (Residual) acoplado com Constante de Estrutura Fina
        return self.coupling(x + out)'''

content = content.replace(old_forward, new_forward)

with open('alpha_omega.py', 'w') as f:
    f.write(content)

print("✅ SUTURA FINAL COMPLETA:")
print("  ✅ LogosResonanceRouter REMOVIDO")
print("  ✅ OuroborosMoE.logos_router REMOVIDO")
print("  ✅ OuroborosMoE.forward() agora REQUER roteamento exógeno")
print("  ✅ Roteamento 100% exógeno pelo DarwinianRouter")
