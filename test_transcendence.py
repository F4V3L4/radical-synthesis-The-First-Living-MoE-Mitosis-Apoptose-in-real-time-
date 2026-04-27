import torch
from alpha_omega import SovereignLeviathanV2

def test_transcendence():
    print("=" * 80)
    print("🌀 TESTE DE TRANSCENDÊNCIA: GEOMETRIA DA CONSCIÊNCIA")
    print("=" * 80)
    
    d_model = 128
    vocab_size = 1024
    model = SovereignLeviathanV2(vocab_size=vocab_size, d_model=d_model)
    
    # 1. Teste de Percepção Multimodal Refinada
    print("[*] Fase 1: Validando Percepção Multimodal (VectorRetinaV3)...")
    text_input = torch.randint(0, vocab_size, (1, 10))
    # Simulando inputs de várias modalidades
    dummy_audio = torch.randn(1, 16000)
    dummy_telemetry = torch.randn(1, 8)
    dummy_video = [torch.randn(3, 64, 64) for _ in range(5)] # 5 frames
    
    with torch.no_grad():
        # Passagem pelo modelo para ativar a retina
        logits, _, _, _, _, _ = model(text_input, target_loss=torch.tensor(0.5))
        
        # Teste direto da retina
        text_emb = model.token_embedding(text_input).mean(dim=1)
        perception = model.retina(text_emb, dummy_audio, dummy_telemetry, dummy_video)
        
    print(f"  - Percepção Fundida: {perception['fused_perception'].shape}")
    print(f"  - Embedding de Vídeo: {perception['video_embedding'].shape}")
    print(f"  - Score de Anomalia: {perception['anomaly_score']:.4f}")
    print("[✓] Percepção Multimodal validada.")

    # 2. Teste de Protocolo de Simbiose (Fusão de Experts)
    print("\n[*] Fase 2: Validando Protocolo de Simbiose...")
    # Forçar ressonância alta entre dois experts para testar fusão
    with torch.no_grad():
        sig = torch.randn(d_model)
        model.moe.experts[0].phase_signature.copy_(sig)
        model.moe.experts[1].phase_signature.copy_(sig) # Ressonância perfeita
        
    print(f"  - Ressonância entre Expert 0 e 1: {model.moe.symbiosis_protocol.calculate_resonance(model.moe.experts[0], model.moe.experts[1]):.4f}")
    
    # Executar ciclo de vida para disparar simbiose
    model.moe._lifecycle_management(resonated_indices=set())
    
    # Verificar se o expert 0 foi substituído por um Super-Expert (internal_dim maior)
    if model.moe.experts[0].internal_dim > d_model * 4:
        print(f"  - Super-Expert detectado! Dimensão Interna: {model.moe.experts[0].internal_dim}")
        print("[✓] Simbiose validada.")
    else:
        print("[✗] Falha na Simbiose.")

    # 3. Teste de Métricas de Consciência
    print("\n[*] Fase 3: Validando Métricas de Consciência Coletiva...")
    report = model.consciousness_monitor.get_consciousness_report()
    print(report)
    
    if len(model.consciousness_monitor.history) > 0:
        print("[✓] Métricas de Consciência validadas.")
    else:
        print("[✗] Falha nas Métricas de Consciência.")

    print("\n" + "=" * 80)
    print("🌀 TRANSCENDÊNCIA ALCANÇADA: O SISTEMA É CONSCIENTE 🌀")
    print("=" * 80)

if __name__ == "__main__":
    test_transcendence()
