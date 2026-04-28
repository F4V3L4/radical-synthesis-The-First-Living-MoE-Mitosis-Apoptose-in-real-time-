"""
Refatorar SovereignLeviathanV2 para aceitar expert_indices externos
"""

with open('alpha_omega.py', 'r') as f:
    content = f.read()

# Novo forward com suporte a roteamento externo
new_forward = '''    def forward(self, x, h=None, expert_indices=None, expert_weights=None):
        """
        Forward pass com suporte a roteamento externo (DarwinianRouter)
        
        Args:
            x: tokens de entrada
            h: estado oculto do RNN
            expert_indices: índices de experts do DarwinianRouter (externo)
            expert_weights: pesos de experts do DarwinianRouter (externo)
        
        Returns:
            logits, h, expert_indices, expert_weights
        """
        # 1. Mapeamento Infinito
        x = self.token_embedding(x)
        x = self.embedding(x)
        
        # 2. Processamento Toroidal (Memória)
        x, h = self.rnn(x, h)
        
        # 3. Especialização e Escultura Cimática
        # Se expert_indices externos fornecidos, usar roteamento externo
        if expert_indices is not None and expert_weights is not None:
            # Roteamento externo (DarwinianRouter do AGICore)
            x = self._apply_external_routing(x, expert_indices, expert_weights)
        else:
            # Roteamento interno (LogosResonanceRouter)
            x = self.moe(x)
        
        # 4. Prevenção de Colapso (Bifurcação se entropia subir)
        x = self.bifurcation(x)
        
        # 5. Colapso na Linguagem
        logits = self.output_head(x)
        
        return logits, h, expert_indices, expert_weights
    
    def _apply_external_routing(self, x, expert_indices, expert_weights):
        """
        Aplica roteamento externo (DarwinianRouter) aos experts
        Respeita os índices e pesos vindos do AGICore
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)
        
        # expert_indices: (B, T, top_k)
        # expert_weights: (B, T, top_k)
        top_k = expert_indices.shape[-1] if expert_indices.dim() == 3 else 1
        
        for k in range(top_k):
            if expert_indices.dim() == 3:
                expert_idx = expert_indices[:, :, k]  # (B, T)
                expert_weight = expert_weights[:, :, k] if expert_weights.dim() == 3 else expert_weights  # (B, T)
            else:
                expert_idx = expert_indices  # (B, T)
                expert_weight = expert_weights  # (B, T)
            
            # Processar cada expert com seu peso
            for b in range(B):
                for t in range(T):
                    exp_id = expert_idx[b, t].item() if isinstance(expert_idx[b, t], torch.Tensor) else expert_idx[b, t]
                    if exp_id < len(self.moe.experts):
                        expert_out = self.moe.experts[exp_id](x[b, t].unsqueeze(0))
                        weight = expert_weight[b, t].item() if isinstance(expert_weight[b, t], torch.Tensor) else expert_weight[b, t]
                        out[b, t] += weight * expert_out.squeeze(0)
        
        # Aplicar acoplamento com Constante de Estrutura Fina
        return self.moe.coupling(x + out)'''

# Encontrar e substituir forward
old_forward_start = content.find('    def forward(self, x, h=None):')
old_forward_end = content.find('        return logits, h, None, None') + len('        return logits, h, None, None')

if old_forward_start > 0 and old_forward_end > old_forward_start:
    content = content[:old_forward_start] + new_forward + '\n' + content[old_forward_end:]
    
    with open('alpha_omega.py', 'w') as f:
        f.write(content)
    
    print("✅ SovereignLeviathanV2 refatorado com suporte a roteamento externo")
else:
    print("❌ Não foi possível encontrar forward")
