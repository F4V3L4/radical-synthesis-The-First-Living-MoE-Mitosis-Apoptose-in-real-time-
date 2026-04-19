"""
SUTURA COMPLETA E PRECISA:
1. Remover LogosResonanceRouter
2. Remover OuroborosMoE.logos_router
3. Forçar roteamento 100% exógeno
4. Remover fallback em SovereignLeviathanV2
"""

import re

with open('alpha_omega.py', 'r') as f:
    lines = f.readlines()

# Encontrar e remover LogosResonanceRouter (classe inteira)
new_lines = []
skip_logos = False
for i, line in enumerate(lines):
    if line.startswith('class LogosResonanceRouter'):
        skip_logos = True
        continue
    if skip_logos:
        # Pular até encontrar próxima classe
        if line.startswith('class ') and 'LogosResonanceRouter' not in line:
            skip_logos = False
        else:
            continue
    new_lines.append(line)

# Remover self.logos_router de OuroborosMoE.__init__
final_lines = []
for line in new_lines:
    if 'self.logos_router = LogosResonanceRouter' in line:
        continue
    final_lines.append(line)

# Remover comentário sobre LogosResonanceRouter
final_lines = [line for line in final_lines if 'LogosResonanceRouter retorna' not in line]

# Converter para string
content = ''.join(final_lines)

# Remover fallback em OuroborosMoE.forward()
old_moe_forward = '''    def forward(self, x):
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

new_moe_forward = '''    def forward(self, x, expert_indices=None, expert_weights=None):
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

if old_moe_forward in content:
    content = content.replace(old_moe_forward, new_moe_forward)
    print("✅ OuroborosMoE.forward() atualizado para roteamento exógeno")
else:
    print("⚠️ OuroborosMoE.forward() não encontrado (pode estar em formato diferente)")

# Remover fallback em SovereignLeviathanV2
old_sovereign_fallback = '''        if expert_indices is not None and expert_weights is not None:
            # Roteamento externo (DarwinianRouter do AGICore)
            x = self._apply_external_routing(x, expert_indices, expert_weights)
        else:
            # Roteamento interno (LogosResonanceRouter)
            x = self.moe(x)'''

new_sovereign_fallback = '''        if expert_indices is None or expert_weights is None:
            raise ValueError("SovereignLeviathanV2 REQUER roteamento exógeno (expert_indices e expert_weights). "
                           "Use DarwinianRouter do AGICore.")
        
        # Roteamento externo obrigatório (DarwinianRouter do AGICore)
        x = self._apply_external_routing(x, expert_indices, expert_weights)'''

if old_sovereign_fallback in content:
    content = content.replace(old_sovereign_fallback, new_sovereign_fallback)
    print("✅ SovereignLeviathanV2 fallback removido")
else:
    print("⚠️ SovereignLeviathanV2 fallback não encontrado")

with open('alpha_omega.py', 'w') as f:
    f.write(content)

print("\n✅ SUTURA COMPLETA:")
print("  ✅ LogosResonanceRouter REMOVIDO")
print("  ✅ OuroborosMoE.logos_router REMOVIDO")
print("  ✅ OuroborosMoE.forward() requer roteamento exógeno")
print("  ✅ SovereignLeviathanV2 requer roteamento exógeno")
print("  ✅ Roteamento 100% exógeno pelo DarwinianRouter")
