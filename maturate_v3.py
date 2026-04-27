import torch
import torch.nn.functional as F
import os
import sys
import time
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss

def deep_maturation_v3():
    print("🔥 Iniciando Protocolo de Ascensão V3: Domínio Sistémico e Expansão Fractal...")
    
    # Setup
    device = "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    # Inicializar Core
    agi = AGICore(
        vocab_size=v_size,
        d_model=512,
        num_experts=8,
        device=device
    )
    
    # Carregar ancestrais (V2)
    agi._load_ancestors()
    
    # Optimizer: Focar no roteador para aprender topologia global
    agi.router.sync_with_experts(agi.core.moe.experts)
    optimizer = torch.optim.AdamW(agi.router.parameters(), lr=1e-3)
    loss_fn = TopologicalDivergenceLoss(d_model=512, num_experts=8)
    
    # Bio-Massa Suprema: Carregar arquivos de digerido
    digerido_path = "digerido"
    bio_massa = []
    for f in os.listdir(digerido_path):
        if f.endswith(".txt"):
            with open(os.path.join(digerido_path, f), "r") as file:
                bio_massa.append(file.read())
    
    if not bio_massa:
        print("❌ Falha: Nenhuma Bio-Massa encontrada em digerido/")
        return
    
    print(f"🧬 Bio-Massa Suprema carregada: {len(bio_massa)} fragmentos técnicos de alto impacto.")
    
    # Treinamento: 500 Iterações de Alto Impacto
    num_iterations = 500
    print(f"🚀 Executando 500 iterações de treinamento fractal no hardware local...")
    
    start_time = time.time()
    for i in range(num_iterations):
        text = bio_massa[i % len(bio_massa)]
        tokens = tokenizer.encode(text)
        if len(tokens) < 5: continue
        
        token_tensor = torch.tensor([tokens[:256]], device=device)
        
        # Forward pass de roteamento
        token_embedding_proj = agi.context_processor.project_to_routing_space(token_tensor, agi.d_model)
        
        # Forçar gradientes no roteador
        for p in agi.router.parameters(): p.requires_grad = True
        
        expert_weights, expert_indices, resonance_gates = agi.router(token_embedding_proj)
        
        # Expandir pesos para (batch, num_experts)
        full_weights = torch.zeros(expert_weights.shape[0], 8, device=device)
        full_weights.scatter_(1, expert_indices, expert_weights)
        
        # Calcular Loss Ontológica V3 (Pressão no Roteador Darwiniano)
        loss = loss_fn(full_weights, resonance_gates)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Sincronizar de volta com os experts (assimilando o treino)
        with torch.no_grad():
            for idx, expert in enumerate(agi.core.moe.experts):
                expert.phase_signature.copy_(agi.router.phase_signatures[idx])
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"⏳ [ITERAÇÃO {i+1}/500] Loss Ontológica: {loss.item():.6f} | Tempo: {elapsed:.2f}s")
            agi.save_state()

    print("✅ Protocolo de Ascensão V3 concluído. Cérebro evoluído.")
    agi.save_state()

if __name__ == "__main__":
    deep_maturation_v3()
