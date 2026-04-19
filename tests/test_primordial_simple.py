"""
Teste simples de apply_primordial_laws com dimensões corretas
"""
import torch
from agi_core import AGICore

device = "cpu"
agi = AGICore(vocab_size=1024, d_model=512, num_experts=8, device=device)

print("Testando apply_primordial_laws com diferentes dimensões:\n")

# Teste 1: (batch, d_model)
print("1️⃣  Input: (2, 512)")
x1 = torch.randn(2, 512)
try:
    out1 = agi.apply_primordial_laws(x1, torch.randint(0, 8, (2, 2)), time=0.1)
    print(f"   ✅ Output shape: {out1.shape}\n")
except Exception as e:
    print(f"   ❌ Erro: {e}\n")

# Teste 2: (batch, seq_len, d_model)
print("2️⃣  Input: (2, 10, 512)")
x2 = torch.randn(2, 10, 512)
try:
    out2 = agi.apply_primordial_laws(x2, torch.randint(0, 8, (2, 2)), time=0.1)
    print(f"   ✅ Output shape: {out2.shape}\n")
except Exception as e:
    print(f"   ❌ Erro: {e}\n")

# Teste 3: (batch, 1, d_model)
print("3️⃣  Input: (2, 1, 512)")
x3 = torch.randn(2, 1, 512)
try:
    out3 = agi.apply_primordial_laws(x3, torch.randint(0, 8, (2, 2)), time=0.1)
    print(f"   ✅ Output shape: {out3.shape}\n")
except Exception as e:
    print(f"   ❌ Erro: {e}\n")

print("✅ Testes de dimensões completados!")
