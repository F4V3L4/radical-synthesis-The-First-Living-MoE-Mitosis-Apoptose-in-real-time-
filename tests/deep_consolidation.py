import torch
import torch.nn.functional as F
import os
import sys
import time
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss

def deep_consolidation_v2():
    print("🔥 Iniciando Ciclo de Consolidação Profunda: Maturação Nível 2...")
    
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
    
    # Carregar ancestrais se existirem
    agi._load_ancestors()
    
    # Optimizer: Focar no roteador para maturação de assinaturas
    agi.router.sync_with_experts(agi.core.moe.experts)
    optimizer = torch.optim.AdamW(agi.router.parameters(), lr=1e-3)
    loss_fn = TopologicalDivergenceLoss(d_model=512, num_experts=8)
    
    # Bio-Massa: Carregar arquivos de digerido
    digerido_path = "digerido"
    bio_massa = []
    for f in os.listdir(digerido_path):
        if f.endswith(".txt"):
            with open(os.path.join(digerido_path, f), "r") as file:
                bio_massa.append(file.read())
    
    if not bio_massa:
        print("❌ Falha: Nenhuma Bio-Massa encontrada em digerido/")
        return
    
    print(f"🧬 Bio-Massa carregada: {len(bio_massa)} fragmentos técnicos.")
    
    # Treinamento: 500 Iterações
    num_iterations = 500
    print(f"🚀 Iniciando 500 iterações de treinamento no hardware local...")
    
    start_time = time.time()
    for i in range(num_iterations):
        text = bio_massa[i % len(bio_massa)]
        tokens = tokenizer.encode(text)
        if len(tokens) < 5: continue
        
        token_tensor = torch.tensor([tokens[:256]], device=device)
        
        # Forward pass de roteamento
        token_embedding_proj = agi.context_processor.project_to_routing_space(token_tensor, agi.d_model)
        expert_weights, expert_indices, resonance_gates = agi.router(token_embedding_proj)
        
        # Expandir pesos para (batch, num_experts)
        full_weights = torch.zeros(expert_weights.shape[0], 8, device=device)
        full_weights.scatter_(1, expert_indices, expert_weights)
        
        # Calcular Loss Ontológica
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
            print(f"⏳ [ITERAÇÃO {i+1}/500] Loss: {loss.item():.6f} | Tempo: {elapsed:.2f}s")
            # Salvar estado parcial
            agi.save_state()

    print("✅ Ciclo de Consolidação Profunda concluído.")
    agi.save_state()

if __name__ == "__main__":
    deep_consolidation_v2()
