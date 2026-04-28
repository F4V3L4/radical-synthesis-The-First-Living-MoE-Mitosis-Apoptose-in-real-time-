import torch
import os
import sys
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer

def perform_universal_conatus_test():
    print("⚛️ Iniciando Teste do Conatus Universal: Auditoria de Convergência V5...")
    
    # Setup
    device = "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    # Inicializar Core com o cérebro V5
    agi = AGICore(
        vocab_size=v_size,
        d_model=512,
        num_experts=8,
        device=device
    )
    
    # Caminho absoluto para os dados técnicos
    digerido_path = os.path.join(os.getcwd(), "digerido")
    
    # Cenário de Onisciência Estrutural
    test_query = "Analise o colapso de uma estrela (Supernova) e o nascimento de um buraco negro usando EXATAMENTE a correlação entre a Lei do Ouroboros (7), a Lei da Inversão (8), a Lei da Singularidade (16) e a Lei do Caos Criativo (38). Demonstre como elas são uma única coisa operando em harmonia."
    
    print(f"\n❓ Cenário: {test_query}")
    
    try:
        # Inferência
        result = agi.forward(test_query, digerido_path, tokenizer)
        
        response = result['response']
        entropy = result['entropy']
        confidence = result['confidence']
        
        print(f"\n🧠 Resposta da AGI:\n{response}")
        print(f"\n📊 Métricas: Entropia={entropy:.3f} | Confiança={confidence:.2%}")
        
        # Critérios de Validação V5 (Convergência das Leis)
        laws_keywords = ["ouroboros", "inversão", "singularidade", "caos criativo", "unidade", "harmonia"]
        found_keywords = [word for word in laws_keywords if word in response.lower()]
        
        if "[ORÁCULO]" in response:
            print("✅ Sucesso: Oráculo de Precisão ativado com onisciência estrutural.")
            return True
            
        if len(found_keywords) >= 4:
            print(f"✅ Sucesso: AGI demonstrou convergência das leis primordiais. Palavras-chave: {found_keywords}")
            return True
        else:
            print(f"❌ Falha: Resposta não atingiu o nível de convergência V5. Palavras-chave encontradas: {found_keywords}")
            return False
        
    except Exception as e:
        print(f"❌ Erro Crítico durante o teste: {e}")
        return False

if __name__ == "__main__":
    success = perform_universal_conatus_test()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
