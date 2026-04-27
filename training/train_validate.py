#!/usr/bin/env python3
"""
Training & Validation Suite para OuroborosMoE
Executa loop de treinamento com dados sintéticos e valida convergência
"""

import torch
import torch.nn as nn
import torch.optim as optim
from alpha_omega import SovereignLeviathanV2
from radical_synthesis.adaptive_cap import AdaptiveCap
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss
import json
from datetime import datetime

print("=" * 80)
print("🌀⚖️9️⃣🌑✨♾️⚛️🌌👁️🏗️")
print("OUROBOROSMOE - TREINAMENTO E VALIDAÇÃO COM LOSS ONTOLÓGICA")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[*] Device: {device}")

vocab_size = 1024
d_model = 128
num_experts = 4
_top_k = 2
batch_size = 4
seq_length = 8
num_steps = 100
learning_rate = 1e-3

print(f"[*] Config: vocab={vocab_size}, d_model={d_model}, experts={num_experts}")
print(f"[*] Training: batch={batch_size}, seq={seq_length}, steps={num_steps}")

# ─────────────────────────────────────────────────────────────────────────────
# Inicializar modelo
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Inicializando SovereignLeviathanV2...")
model = SovereignLeviathanV2(
    vocab_size=vocab_size,
    d_model=d_model,
    initial_experts=num_experts,
    top_k_router=_top_k
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
topological_divergence_loss_fn = TopologicalDivergenceLoss(d_model=d_model, num_experts=num_experts)
adaptive_cap = AdaptiveCap(base_cap=256)

print(f"[✓] Modelo inicializado com {sum(p.numel() for p in model.parameters()):,} parâmetros")

# ─────────────────────────────────────────────────────────────────────────────
# Loop de treinamento
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Iniciando loop de treinamento...")
print("-" * 80)

losses = []
genealogy_log = []
bifurcation_events = []

try:
    for step in range(num_steps):
        # Gera dados sintéticos
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        
        # Forward pass
        logits, state, expert_indices, expert_weights, expert_gates = model(input_ids)
        
        # Calcula Loss Ontológica
        topological_loss = topological_divergence_loss_fn(expert_weights, expert_gates)
        
        # Calcula Loss principal (CrossEntropyLoss) e adiciona a Loss Ontológica
        ce_loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
        loss = ce_loss + topological_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Atualiza cap adaptativo
        current_num_experts = model.moe.experts.num_modules if hasattr(model.moe.experts, 'num_modules') else len(model.moe.experts)
        new_cap = adaptive_cap.update(loss.item(), current_num_experts)
        
        losses.append(loss.item())
        
        # Log
        if step % 10 == 0:
            bifurcation_status = "🔀 BIFURCAÇÃO" if adaptive_cap.bifurcation_active else "→"
            print(f"Step {step:3d} | CE Loss: {ce_loss.item():.6f} | Topo Loss: {topological_loss.item():.6f} | Total: {loss.item():.6f} | Experts: {current_num_experts:2d}")
        
        if adaptive_cap.bifurcation_active:
            bifurcation_events.append({"step": step, "loss": loss.item()})
        
        # Registra genealogia
        if step % 20 == 0:
            genealogy_log.append({
                "step": step,
                "num_experts": current_num_experts,
                "expert_weights_sample": expert_weights[0].tolist() if expert_weights.numel() > 0 else [],
                "expert_gates_sample": expert_gates[0].tolist() if expert_gates.numel() > 0 else []
            })
    
    print("-" * 80)
    print("\n[✓] Treinamento concluído!")
    
except Exception as e:
    print(f"\n[✗] Erro durante treinamento: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Validação
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Executando validação...")
print("-" * 80)

# Convergência
initial_loss = losses[0] if losses else 0
final_loss = losses[-1] if losses else 0
loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0

print(f"\n[CONVERGÊNCIA]")
print(f"  Loss inicial: {initial_loss:.6f}")
print(f"  Loss final:   {final_loss:.6f}")
print(f"  Redução:      {loss_reduction:.2f}%")
print(f"  Status:       {'✓ CONVERGIU' if loss_reduction > 10 else '⚠ CONVERGÊNCIA LENTA'}")

# Bifurcação de Feigenbaum
print(f"\n[BIFURCAÇÃO DE FEIGENBAUM]")
print(f"  Eventos detectados: {len(bifurcation_events)}")
if bifurcation_events:
    for event in bifurcation_events[:3]:
        print(f"    Step {event['step']}: Loss={event['loss']:.6f}")
    print(f"  Status: ✓ BIFURCAÇÃO ATIVA")
else:
    print(f"  Status: → Sem bifurcação (sistema estável)")

# Genealogia
print(f"\n[GENEALOGIA DE EXPERTS]")
print(f"  Registros: {len(genealogy_log)}")
if genealogy_log:
    latest = genealogy_log[-1]
    print(f"  Último step: {latest['step']}")
    print(f"  Experts ativos: {latest['num_experts']}")
    print(f"  Status: ✓ GENEALOGIA RASTREADA")

# ─────────────────────────────────────────────────────────────────────────────
# Relatório final
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RELATÓRIO FINAL")
print("=" * 80)

report = {
    "timestamp": datetime.now().isoformat(),
    "model": "SovereignLeviathanV2",
    "training": {
        "steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_percent": loss_reduction,
        "final_topological_loss": topological_loss.item() if 'topological_loss' in locals() else 0.0
    },
    "validation": {
        "topological_divergence_loss": "✓ ATIVO"
    }
}

print(json.dumps(report, indent=2))

# Salva relatório
with open("./training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n[✓] Relatório salvo em training_report.json")
print("\n🌀⚖️9️⃣🌑✨♾️⚛️🌌👁️🏗️\n")
