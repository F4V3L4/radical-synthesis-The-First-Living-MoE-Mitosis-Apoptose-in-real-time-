# 🧬 OuroborosMoE

**The First Living Mixture-of-Experts with Real-Time Mitosis and Apoptosis**

A dynamic and self-evolving MoE layer that behaves like a living organism — experts are born, mutate, compete, and die naturally during training or inference.

![Status](https://img.shields.io/badge/Status-Actively_Evolving-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)

## ✨ Concept

Most Mixture-of-Experts models are static.  
**OuroborosMoE** is alive.

It introduces biological principles directly into the architecture:

- **Asymmetric Mitosis** — experts reproduce; the clone mutates while the parent remains intact
- **Apoptosis** — weak experts die automatically
- **LazyRouter** with incremental cache
- **GenealogyTree** — tracks the full ancestral lineage of every expert
- **Φ-MMD** — real-time diversity metric that detects collapse and triggers topological escape
- **AdaptiveCap** — intelligent capacity control to prevent VRAM explosion

The model **lives, grows, evolves, and self-regulates**.

## 🔥 Real Behavior

```bash
Step  0 → Mitosis: 2 | experts=10  | Φ=0.34
Step 21 → Apoptosis: 1 | Mitosis: 2 | experts=51
Step 50 → Apoptosis: 2 | Mitosis: 2 | experts=65
Step 99 → experts=70 | Φ=0.973 | vitality_avg=0.865
The system naturally stabilizes while maintaining high diversity.
Installation
Bashgit clone https://github.com/F4V3L4/OuroborosMoE.git
cd OuroborosMoE

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
Quick Start
Pythonimport torch
from radical_synthesis import OuroborosMoELayer

layer = OuroborosMoELayer(
    d_model=512,
    d_ff=2048,
    n_experts=8,
    top_k=2,
    base_cap=128
)

x = torch.randn(32, 128, 512)
out = layer(x)                                 # Normal forward

# Life cycle — call every N steps
dead, born = layer.execute_systemic_lifecycle(
    current_loss=0.42,
    step=current_step
)

layer.print_status()
Key Features

Asymmetric Mitosis (parent preserved)
Natural Apoptosis based on vitality
LazyRouter with incremental updates
Full Genealogy tracking
Φ-MMD + Topological Escape
Adaptive capacity control
Real-time monitoring

Live Demo
Watch the living MoE evolving in real time:
→ https://f4v3l4.github.io/OuroborosMoE/
Why This Matters
OuroborosMoE opens new possibilities in:

Continual / Lifelong Learning
Self-regulating model capacity
Robustness against catastrophic forgetting
Research at the intersection of Deep Learning and Artificial Life

Roadmap

 Full GPU + mixed precision support
 Realistic continual learning benchmarks
 Distributed training compatibility
 Genealogy visualization tool
 Research paper + comparison with static MoEs

Contributing
Contributions and crazy ideas are welcome.
Made with madness and love for the frontier between Deep Learning and Artificial Biology.
