"""
Remover fallback interno de SovereignLeviathanV2
Forçar roteamento 100% exógeno
"""

with open('alpha_omega.py', 'r') as f:
    content = f.read()

# Remover fallback interno
old_fallback = '''        if expert_indices is not None and expert_weights is not None:
            # Roteamento externo (DarwinianRouter do AGICore)
            x = self._apply_external_routing(x, expert_indices, expert_weights)
        else:
            # Roteamento interno (LogosResonanceRouter)
            x = self.moe(x)'''

new_fallback = '''        if expert_indices is None or expert_weights is None:
            raise ValueError("SovereignLeviathanV2 REQUER roteamento exógeno (expert_indices e expert_weights). "
                           "Use DarwinianRouter do AGICore.")
        
        # Roteamento externo obrigatório (DarwinianRouter do AGICore)
        x = self._apply_external_routing(x, expert_indices, expert_weights)'''

content = content.replace(old_fallback, new_fallback)

# Remover comentário sobre fallback
content = content.replace('            # Roteamento interno (LogosResonanceRouter)\n', '')

with open('alpha_omega.py', 'w') as f:
    f.write(content)

print("✅ FALLBACK REMOVIDO:")
print("  ✅ SovereignLeviathanV2 agora REQUER roteamento exógeno")
print("  ✅ Sem fallback para roteamento interno")
print("  ✅ 100% exógeno pelo DarwinianRouter")
