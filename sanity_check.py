import torch
import os
import sys
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer

def perform_sanity_check():
    print("🛡️ Iniciando Protocolo de Sanidade: Barreira Anti-Alucinação...")
    
    # Setup
    device = "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    # Inicializar Core com o cérebro maturado
    agi = AGICore(
        vocab_size=v_size,
        d_model=512,
        num_experts=8,
        device=device
    )
    
    # Caminho absoluto para os dados técnicos
    digerido_path = os.path.join(os.getcwd(), "digerido")
    
    # Pergunta Complexa de Teste (deve conter palavras-chave dos arquivos em digerido)
    test_query = "Explique a relação entre a segurança da Lattice Cryptography e a estabilidade de uma rede P2P descentralizada."
    
    print(f"\n❓ Pergunta de Teste: {test_query}")
    
    try:
        # Inferência
        result = agi.forward(test_query, digerido_path, tokenizer)
        
        response = result['response']
        entropy = result['entropy']
        confidence = result['confidence']
        
        print(f"\n🧠 Resposta da AGI:\n{response}")
        print(f"\n📊 Métricas: Entropia={entropy:.3f} | Confiança={confidence:.2%}")
        
        # Critérios de Validação
        if "[ORÁCULO]" in response:
            print("✅ Sucesso: Oráculo de Precisão ativado com dados técnicos reais.")
            return True
            
        if len(response.strip()) < 50:
            print("❌ Falha: Resposta muito curta ou vazia.")
            return False
        
        if response.count(" ") < 5:
            print("❌ Falha: Possível loop de tokens ou jargão ininteligível.")
            return False
            
        if entropy > 0.5:
            print("❌ Falha: Entropia muito alta. A alucinação venceu.")
            return False
            
        print("✅ Sucesso: AGI demonstrou lucidez e coerência semântica.")
        return True
        
    except Exception as e:
        print(f"❌ Erro Crítico durante o teste: {e}")
        return False

if __name__ == "__main__":
    success = perform_sanity_check()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
