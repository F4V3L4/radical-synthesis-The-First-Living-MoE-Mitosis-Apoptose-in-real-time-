import torch
import os
import sys
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer

def perform_ghost_test():
    print("👻 Iniciando Teste do Fantasma: Auditoria de Zero Entropia...")
    
    # Setup
    device = "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    # Inicializar Core com o cérebro V3
    agi = AGICore(
        vocab_size=v_size,
        d_model=512,
        num_experts=8,
        device=device
    )
    
    # Caminho absoluto para os dados técnicos
    digerido_path = os.path.join(os.getcwd(), "digerido")
    
    # Cenário de Engenharia Extrema
    test_query = "Como a Radical Synthesis opera num nó isolado sob escassez térmica e de memória?"
    
    print(f"\n❓ Cenário: {test_query}")
    
    try:
        # Inferência
        result = agi.forward(test_query, digerido_path, tokenizer)
        
        response = result['response']
        entropy = result['entropy']
        confidence = result['confidence']
        
        print(f"\n🧠 Resposta da AGI:\n{response}")
        print(f"\n📊 Métricas: Entropia={entropy:.3f} | Confiança={confidence:.2%}")
        
        # Critérios de Validação V3 (Engenharia Extrema)
        success_keywords = ["conatus", "compress", "apoptose", "assíncrona", "isolamento", "módulo", "memória"]
        found_keywords = [word for word in success_keywords if word in response.lower()]
        
        if "[ORÁCULO]" in response:
            print("✅ Sucesso: Oráculo de Precisão ativado com domínio sistémico.")
            return True
            
        if len(found_keywords) >= 2:
            print(f"✅ Sucesso: AGI demonstrou pensamento de engenharia extrema. Palavras-chave: {found_keywords}")
            return True
        else:
            print(f"❌ Falha: Resposta insuficiente para o nível V3. Palavras-chave encontradas: {found_keywords}")
            return False
        
    except Exception as e:
        print(f"❌ Erro Crítico durante o teste: {e}")
        return False

if __name__ == "__main__":
    success = perform_ghost_test()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
