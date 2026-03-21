from radical_synthesis import OuroborosMoELayer
import torch

print("🚀 Iniciando Leviathan Demo...")

model = OuroborosMoELayer(256, 1024, initial_experts=4, top_k=2).cuda()
x = torch.randn(16, 64, 256).cuda()

for epoch in range(5):
    out = model(x)
    dead, born = model.execute_systemic_lifecycle()
    print(f"Epoch {epoch} | Apoptose: {len(dead)} | Mitose: {len(born)}")
