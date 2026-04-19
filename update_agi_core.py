"""
Atualizar AGICore para passar expert_indices e expert_weights ao core
"""

with open('agi_core.py', 'r') as f:
    content = f.read()

# Atualizar método process()
old_process = '''    def process(self, tokens: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Camada de Processamento: Forward pass do core
        
        Returns:
            logits
        """
        with torch.no_grad():
            # O core já tem roteamento interno, não precisa passar expert_indices
            logits, _, _, _ = self.core(tokens, None)
        return logits'''

new_process = '''    def process(self, tokens: torch.Tensor, expert_indices: Optional[torch.Tensor] = None, 
                expert_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Camada de Processamento: Forward pass do core com roteamento externo
        
        Args:
            tokens: tensor de tokens
            expert_indices: índices de experts do DarwinianRouter
            expert_weights: pesos de experts do DarwinianRouter
        
        Returns:
            logits
        """
        with torch.no_grad():
            # Passar roteamento externo ao core
            logits, _, _, _ = self.core(tokens, None, expert_indices, expert_weights)
        return logits'''

content = content.replace(old_process, new_process)

# Atualizar chamadas em verify_logic (linha 378)
old_call_1 = '''            logits = self.process(tokens[:, -256:], expert_indices)'''
new_call_1 = '''            logits = self.process(tokens[:, -256:], expert_indices, expert_weights)'''
content = content.replace(old_call_1, new_call_1)

# Atualizar chamada em verify_logic (linha 403)
old_call_2 = '''                logits = self.process(tokens[:, -256:], expert_indices)'''
new_call_2 = '''                logits = self.process(tokens[:, -256:], expert_indices, expert_weights)'''
# Precisa fazer com cuidado para não substituir a anterior
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'logits = self.process(tokens[:, -256:], expert_indices)' in line and i > 400:
        lines[i] = line.replace('logits = self.process(tokens[:, -256:], expert_indices)', 
                               'logits = self.process(tokens[:, -256:], expert_indices, expert_weights)')
content = '\n'.join(lines)

# Atualizar chamadas em forward() (linhas 485 e 505)
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'logits = self.process(token_tensor[:, -256:], expert_indices)' in line:
        lines[i] = line.replace('logits = self.process(token_tensor[:, -256:], expert_indices)',
                               'logits = self.process(token_tensor[:, -256:], expert_indices, expert_weights)')
content = '\n'.join(lines)

with open('agi_core.py', 'w') as f:
    f.write(content)

print("✅ AGICore atualizado para passar roteamento ao core")
