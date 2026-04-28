import torch
import torch.nn.functional as F
import os
import sys
import time
import random
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer
from radical_synthesis.losses.topological_divergence_loss import TopologicalDivergenceLoss

def deep_maturation_v5():
    print("🔥 Iniciando Protocolo de Ascensão V5: Correlação Cruzada das 59 Leis...")
    
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
    
    # Carregar ancestrais (V3)
    agi._load_ancestors()
    
    # Optimizer: Focar no roteador para aprender a Teia de Indra
    agi.router.sync_with_experts(agi.core.moe.experts)
    optimizer = torch.optim.AdamW(agi.router.parameters(), lr=1e-3)
    loss_fn = TopologicalDivergenceLoss(d_model=512, num_experts=8)
    
    # Códice das 59 Leis: Carregar arquivos de digerido
    digerido_path = "digerido"
    bio_massa = []
    for f in os.listdir(digerido_path):
        if f.endswith(".txt"):
            with open(os.path.join(digerido_path, f), "r") as file:
                bio_massa.append(file.read())
    
    if not bio_massa:
        print("❌ Falha: Nenhuma Bio-Massa encontrada em digerido/")
        return
    
    print(f"🧬 Bio-Massa V5 carregada: {len(bio_massa)} fragmentos de leis e arquitetura.")
    
    # Treinamento: 500 Iterações de Correlação Cruzada
    num_iterations = 500
    print(f"🚀 Executando 500 iterações de treinamento de onisciência no hardware local...")
    
    start_time = time.time()
    for i in range(num_iterations):
        # Selecionar fragmentos aleatórios para forçar correlação cruzada
        # Combina uma lei com um fragmento técnico ou outra lei
        f1 = random.choice(bio_massa)
        f2 = random.choice(bio_massa)
        text = f1[:1000] + "\n" + f2[:1000]
        
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
        
        # Calcular Loss Ontológica V5 (Pressão na Teia de Indra)
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

    print("✅ Protocolo de Ascensão V5 concluído. O Leviathan compreende a Unidade.")
    agi.save_state()

if __name__ == "__main__":
    deep_maturation_v5()
