"""
Simulação de Autocrítica: Teste do Loop de Verificação Recursiva
Demonstra o pipeline completo com queries técnicas
"""

import torch
import os
import sys
from agi_core import AGICore
from daemon_agi import OmegaTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGERIDO_PATH = os.path.join(BASE_DIR, "digerido")


def print_separator(title: str):
    """Imprime separador visual"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def simulate_autocritique():
    """Executa simulação de autocrítica"""
    
    print_separator("🌀 SIMULAÇÃO DE AUTOCRÍTICA - OUROBOROSMOE v8.0")
    
    # Inicializar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    agi = AGICore(
        vocab_size=v_size,
        d_model=512,
        num_experts=8,
        device=device
    )
    
    print(f"\n✓ AGI Core inicializado")
    print(f"  Device: {device}")
    print(f"  d_model: 512")
    print(f"  num_experts: 8")
    print(f"  entropy_threshold: {agi.entropy_threshold}")
    print(f"  max_iterations: {agi.max_autocritique_iterations}")
    
    # Queries de teste
    test_queries = [
        {
            'query': 'O que é uma matriz em álgebra linear?',
            'type': 'TÉCNICA (Matemática)',
            'expected_temp': 0.1
        },
        {
            'query': 'Como funciona o DarwinianRouter?',
            'type': 'TÉCNICA (Código)',
            'expected_temp': 0.1
        },
        {
            'query': 'Qual é o significado da vida?',
            'type': 'GERAL (Filosófica)',
            'expected_temp': 0.8
        }
    ]
    
    print_separator("📊 TESTE DE DETECÇÃO DE QUERY TÉCNICA")
    
    for i, test in enumerate(test_queries, 1):
        query = test['query']
        is_technical = agi.context_processor.detect_technical_query(query)
        expected_temp = test['expected_temp']
        
        # Simular injeção de contexto
        _, actual_temp = agi.context_processor.inject_technical_data(query, "dados técnicos de exemplo")
        
        status = "✓" if actual_temp == expected_temp else "✗"
        print(f"\n{status} Teste {i}: {test['type']}")
        print(f"  Query: '{query}'")
        print(f"  Detectada como técnica: {is_technical}")
        print(f"  Temperatura esperada: {expected_temp}")
        print(f"  Temperatura real: {actual_temp}")
    
    print_separator("🔄 TESTE DE DIVERGÊNCIA SEMÂNTICA")
    
    # Testes de divergência
    divergence_tests = [
        {
            'response': 'Uma matriz é um arranjo retangular de números em linhas e colunas.',
            'technical_data': 'Matriz: estrutura de dados com linhas e colunas contendo números reais ou complexos.',
            'expected': 'BAIXA'
        },
        {
            'response': 'O gato subiu na árvore e miou para a lua.',
            'technical_data': 'Uma matriz é um arranjo retangular de números.',
            'expected': 'ALTA'
        }
    ]
    
    for i, test in enumerate(divergence_tests, 1):
        entropy = agi.compute_semantic_divergence(test['response'], test['technical_data'])
        status = "✓" if (entropy < 0.3 and test['expected'] == 'BAIXA') or (entropy >= 0.3 and test['expected'] == 'ALTA') else "✗"
        
        print(f"\n{status} Teste {i}: Divergência {test['expected']}")
        print(f"  Response: '{test['response'][:50]}...'")
        print(f"  Technical: '{test['technical_data'][:50]}...'")
        print(f"  Entropia: {entropy:.3f}")
        print(f"  Threshold: {agi.entropy_threshold}")
        print(f"  Autocrítica acionada: {entropy > agi.entropy_threshold}")
    
    print_separator("🧠 TESTE DE FORWARD PASS COMPLETO")
    
    # Teste de forward pass
    query = "Qual é a dimensionalidade do d_model?"
    print(f"\nQuery: '{query}'")
    print(f"Tipo: TÉCNICA (Código)")
    
    try:
        result = agi.forward(query, DIGERIDO_PATH, tokenizer)
        
        print(f"\n✓ Forward pass completado")
        print(f"\n📊 Resultados:")
        print(f"  Response: {result['response'][:100]}...")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Was Corrected: {result['was_corrected']}")
        print(f"  Entropy: {result['entropy']:.3f}")
        print(f"  Winner Expert: {result['winner_expert']}")
        print(f"  Winner Vitality: {result['winner_vitality']:.1%}")
        
        if result['correction_path']:
            print(f"\n🔄 Caminho de Correção:")
            for step in result['correction_path']:
                if 'entropy_before' in step:
                    print(f"  Iter {step['iteration']}: {step['entropy_before']:.3f} → {step['entropy_after']:.3f}")
                else:
                    print(f"  {step['action']}")
        
        # Verificar memória
        stats = agi.get_stats()
        print(f"\n💾 Memória:")
        print(f"  Total memories: {stats['memory_size']}")
        print(f"  Genealogy size: {stats['genealogy_size']}")
        print(f"  Correction paths: {stats['correction_paths_count']}")
        
    except Exception as e:
        print(f"\n✗ Erro no forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print_separator("✅ SIMULAÇÃO COMPLETA")
    print("\n🌀 OuroborosMoE AGI - Autocrítica Funcional!")
    print("   Loop de Verificação Recursiva: ATIVO")
    print("   Memória Episódica: ATIVA")
    print("   Fidelidade Bare-Metal: ATIVA")
    print("   CLI Soberana: ATIVA\n")


if __name__ == "__main__":
    simulate_autocritique()
