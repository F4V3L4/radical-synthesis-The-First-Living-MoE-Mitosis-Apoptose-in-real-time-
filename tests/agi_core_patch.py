# Adicionar método project_to_routing_space ao ContextualProcessor

patch_code = '''
    def project_to_routing_space(self, token_tensor: torch.Tensor) -> torch.Tensor:
        """
        Projeta tokens de entrada para espaço de roteamento
        Converte (batch, seq_len) em (batch, d_model) para roteamento
        """
        # Se tokens são índices (2D), converter para embedding médio
        if token_tensor.dim() == 2:
            # Média dos índices como proxy para embedding
            token_mean = token_tensor.float().mean(dim=1, keepdim=True)  # (batch, 1)
            # Expandir para d_model dimensões
            embedding = token_mean.expand(-1, self.d_model)  # (batch, d_model)
            return embedding
        elif token_tensor.dim() == 3:
            # Se já é (batch, seq_len, d_model), retornar média ao longo de seq_len
            return token_tensor.mean(dim=1)  # (batch, d_model)
        else:
            # Fallback: retornar tensor aleatório normalizado
            return torch.randn(token_tensor.shape[0], self.d_model, device=token_tensor.device)
'''

with open('agi_core.py', 'r') as f:
    content = f.read()

# Encontrar fim de ContextualProcessor
insert_pos = content.find('class AGICore(nn.Module):')

if insert_pos > 0:
    content = content[:insert_pos] + patch_code + '\n\n' + content[insert_pos:]
    
    with open('agi_core.py', 'w') as f:
        f.write(content)
    
    print("✅ Método project_to_routing_space adicionado")
else:
    print("❌ Não foi possível encontrar AGICore")
