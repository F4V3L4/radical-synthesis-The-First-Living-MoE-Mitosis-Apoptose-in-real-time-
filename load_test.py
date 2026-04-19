"""
Testes de Carga: Múltiplas Queries com Medição de Performance
Valida estabilidade, latência e uso de memória
"""

import torch
import os
import sys
import time
import json
from datetime import datetime
from agi_core import AGICore
from daemon_agi import OmegaTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGERIDO_PATH = os.path.join(BASE_DIR, "digerido")


class LoadTester:
    """Executor de testes de carga"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = OmegaTokenizer()
        v_size = max(self.tokenizer.vocab.keys()) + 1
        
        self.agi = AGICore(
            vocab_size=v_size,
            d_model=512,
            num_experts=8,
            device=str(self.device)
        )
        
        self.results = []
    
    def print_header(self, title: str):
        """Imprime cabeçalho"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
    
    def run_load_test(self, queries: list, test_name: str):
        """Executa teste de carga com múltiplas queries"""
        self.print_header(f"🔥 TESTE DE CARGA: {test_name}")
        
        print(f"\nTotal de queries: {len(queries)}")
        print(f"Device: {self.device}\n")
        
        start_time = time.time()
        successful = 0
        failed = 0
        total_latency = 0
        corrections_count = 0
        
        for i, query in enumerate(queries, 1):
            query_start = time.time()
            
            try:
                result = self.agi.forward(query, DIGERIDO_PATH, self.tokenizer)
                query_latency = time.time() - query_start
                
                successful += 1
                total_latency += query_latency
                
                if result['was_corrected']:
                    corrections_count += 1
                
                # Mostrar progresso a cada 5 queries
                if i % 5 == 0 or i == len(queries):
                    avg_latency = total_latency / successful
                    print(f"  [{i:3d}/{len(queries)}] ✓ Latência: {query_latency:.3f}s | Média: {avg_latency:.3f}s | Correções: {corrections_count}")
                
                self.results.append({
                    'query': query,
                    'latency': query_latency,
                    'was_corrected': result['was_corrected'],
                    'entropy': result['entropy'],
                    'confidence': result['confidence'],
                    'winner_expert': result['winner_expert']
                })
            
            except Exception as e:
                failed += 1
                print(f"  [{i:3d}/{len(queries)}] ✗ ERRO: {str(e)[:50]}")
        
        total_time = time.time() - start_time
        avg_latency = total_latency / max(successful, 1)
        
        print(f"\n📊 RESULTADOS DO TESTE:")
        print(f"  Total de queries: {len(queries)}")
        print(f"  Sucesso: {successful} ({100*successful/len(queries):.1f}%)")
        print(f"  Falhas: {failed}")
        print(f"  Tempo total: {total_time:.2f}s")
        print(f"  Latência média: {avg_latency:.3f}s")
        print(f"  Throughput: {successful/total_time:.1f} queries/s")
        print(f"  Correções acionadas: {corrections_count}")
        
        # Estatísticas de memória
        stats = self.agi.get_stats()
        print(f"\n💾 MEMÓRIA:")
        print(f"  Total memories: {stats['memory_size']}")
        print(f"  Genealogy size: {stats['genealogy_size']}")
        print(f"  Correction paths: {stats['correction_paths_count']}")
        
        # Estatísticas de experts
        genealogy = self.agi.memory.get_genealogy_tree()
        print(f"\n🧬 EXPERTS:")
        for expert_id, info in genealogy.items():
            print(f"  Expert {expert_id}: {info['memories_count']} memórias, vitalidade {info['vitality']:.1%}")
        
        return {
            'test_name': test_name,
            'total_queries': len(queries),
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'avg_latency': avg_latency,
            'throughput': successful/total_time,
            'corrections': corrections_count
        }
    
    def run_all_tests(self):
        """Executa todos os testes de carga"""
        
        # Teste 1: Queries Técnicas (Matemática)
        technical_math_queries = [
            "Qual é o resultado de 2 + 2?",
            "Como calcular a derivada de x²?",
            "O que é uma matriz identidade?",
            "Explique a transformada de Fourier",
            "Qual é a complexidade do algoritmo quicksort?",
        ]
        
        # Teste 2: Queries Técnicas (Código)
        technical_code_queries = [
            "Como funciona o DarwinianRouter?",
            "Explique o decorator @property em Python",
            "O que é uma closure em JavaScript?",
            "Como implementar um heap em C++?",
            "Qual é a diferença entre var, let e const?",
        ]
        
        # Teste 3: Queries Gerais
        general_queries = [
            "Como você está?",
            "Qual é o significado da vida?",
            "O que você pensa sobre inteligência artificial?",
            "Descreva um dia perfeito",
            "O que é felicidade?",
        ]
        
        # Teste 4: Queries Mistas
        mixed_queries = [
            "Qual é a dimensionalidade do d_model?",
            "Como a entropia afeta a autocrítica?",
            "Explique o conceito de vitalidade em experts",
            "O que é um loop de verificação recursiva?",
            "Como detectar queries técnicas?",
        ]
        
        results = []
        
        # Executar testes
        results.append(self.run_load_test(technical_math_queries, "Queries Técnicas (Matemática)"))
        results.append(self.run_load_test(technical_code_queries, "Queries Técnicas (Código)"))
        results.append(self.run_load_test(general_queries, "Queries Gerais"))
        results.append(self.run_load_test(mixed_queries, "Queries Mistas"))
        
        # Resumo final
        self.print_header("📈 RESUMO FINAL DE TESTES")
        
        print("\n")
        print(f"{'Teste':<35} {'Queries':<10} {'Sucesso':<10} {'Latência':<12} {'Throughput':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['test_name']:<35} {result['total_queries']:<10} {result['successful']:<10} "
                  f"{result['avg_latency']:.3f}s{'':<7} {result['throughput']:.1f}/s")
        
        # Estatísticas globais
        total_queries = sum(r['total_queries'] for r in results)
        total_successful = sum(r['successful'] for r in results)
        total_time = sum(r['total_time'] for r in results)
        total_corrections = sum(r['corrections'] for r in results)
        
        print("-" * 80)
        print(f"{'TOTAL':<35} {total_queries:<10} {total_successful:<10} "
              f"{(total_time/total_successful):.3f}s{'':<7} {total_successful/total_time:.1f}/s")
        
        print(f"\n🎯 ESTATÍSTICAS GLOBAIS:")
        print(f"  Total de queries: {total_queries}")
        print(f"  Taxa de sucesso: {100*total_successful/total_queries:.1f}%")
        print(f"  Tempo total: {total_time:.2f}s")
        print(f"  Latência média: {(total_time/total_successful):.3f}s")
        print(f"  Throughput: {total_successful/total_time:.1f} queries/s")
        print(f"  Correções acionadas: {total_corrections}")
        
        # Salvar resultados em JSON
        output_file = os.path.join(BASE_DIR, "load_test_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'tests': results,
                'global_stats': {
                    'total_queries': total_queries,
                    'total_successful': total_successful,
                    'success_rate': 100*total_successful/total_queries,
                    'total_time': total_time,
                    'avg_latency': total_time/total_successful,
                    'throughput': total_successful/total_time,
                    'total_corrections': total_corrections
                }
            }, f, indent=2)
        
        print(f"\n✓ Resultados salvos em: {output_file}")


if __name__ == "__main__":
    tester = LoadTester()
    tester.run_all_tests()
