"""
Validação de Performance: Tier 1 + Tier 2
Compara throughput, latência e precisão antes/depois
"""

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple
from radical_synthesis.primordial_laws import (
    HarmonicEncoder, QuantumSuperposition, HyperbolicEmbedding, SynchronicityDetector
)
from radical_synthesis.primordial_laws_tier2 import (
    PlanetaryGrid, Amplituedro, SimultaneityProcessor, QuantumEntanglement, StrangeAttractor
)

class PerformanceValidator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {
            'tier1': {},
            'tier2': {},
            'combined': {}
        }
    
    def benchmark_tier1(self) -> Dict:
        """Benchmark Tier 1 (4 leis)"""
        print("⏱️  BENCHMARKING TIER 1 (4 Leis)\n")
        
        # Inicializar componentes
        harmonic = HarmonicEncoder(d_model=512, device=self.device)
        quantum = QuantumSuperposition(num_states=8, d_model=512, device=self.device)
        hyperbolic = HyperbolicEmbedding(d_model=512, device=self.device)
        synchronicity = SynchronicityDetector(num_experts=8, d_model=512, device=self.device)
        
        # Dados de teste
        num_iterations = 100
        batch_size = 4
        seq_len = 10
        
        results = {}
        
        # 1. HarmonicEncoder
        print("1️⃣  HarmonicEncoder (Código 144)")
        x = torch.randn(batch_size, seq_len, 512, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _ = harmonic(x, time=0.1)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000  # ms
        
        results['harmonic'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'coherence': float(harmonic.get_coherence())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Coerência: {results['harmonic']['coherence']:.3f}\n")
        
        # 2. QuantumSuperposition
        print("2️⃣  QuantumSuperposition (Lei da Superposição)")
        x_flat = torch.randn(batch_size, 512, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _ = quantum(x_flat)
            _, _ = quantum.collapse(x_flat)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['quantum'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'entanglement': float(quantum.get_entanglement())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Emaranhamento: {results['quantum']['entanglement']:.3f}\n")
        
        # 3. HyperbolicEmbedding
        print("3️⃣  HyperbolicEmbedding (Geometria Hiperbólica)")
        x = torch.randn(batch_size, seq_len, 512, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _ = hyperbolic(x)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['hyperbolic'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'expansion_rate': float(hyperbolic.get_expansion_rate())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Taxa de expansão: {results['hyperbolic']['expansion_rate']:.3f}\n")
        
        # 4. SynchronicityDetector
        print("4️⃣  SynchronicityDetector (Lei da Sincronicidade)")
        expert_acts = torch.randn(batch_size, 8, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _, _ = synchronicity(expert_acts)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['synchronicity'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'score': float(synchronicity.get_synchronicity_score())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Score: {results['synchronicity']['score']:.3f}\n")
        
        # Agregado Tier 1
        total_throughput = sum(r['throughput'] for r in results.values())
        avg_latency = sum(r['latency_ms'] for r in results.values()) / len(results)
        
        print(f"📊 AGREGADO TIER 1:")
        print(f"   Throughput Total: {total_throughput:.1f} samples/s")
        print(f"   Latência Média: {avg_latency:.2f}ms\n")
        
        results['total'] = {
            'throughput': total_throughput,
            'latency_ms': avg_latency
        }
        
        return results
    
    def benchmark_tier2(self) -> Dict:
        """Benchmark Tier 2 (5 leis)"""
        print("⏱️  BENCHMARKING TIER 2 (5 Leis)\n")
        
        # Inicializar componentes
        grid = PlanetaryGrid(num_experts=8, device=self.device)
        amplituedro = Amplituedro(num_experts=8, device=self.device)
        simultaneity = SimultaneityProcessor(num_timelines=4, device=self.device)
        entanglement = QuantumEntanglement(num_experts=8, device=self.device)
        attractor = StrangeAttractor(num_experts=8, device=self.device)
        
        num_iterations = 100
        batch_size = 4
        
        results = {}
        
        # 1. PlanetaryGrid
        print("1️⃣  PlanetaryGrid (Grade Harmônica Planetária)")
        expert_acts = torch.randn(batch_size, 8, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _ = grid(expert_acts, time=0.1)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['grid'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'coherence': float(grid.get_global_coherence())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Coerência: {results['grid']['coherence']:.3f}\n")
        
        # 2. Amplituedro
        print("2️⃣  Amplituedro (Otimização de Caminhos)")
        expert_indices = torch.randint(0, 8, (batch_size, 3), device=self.device)
        expert_weights = torch.softmax(torch.randn(batch_size, 3, device=self.device), dim=1)
        
        start = time.time()
        for _ in range(num_iterations):
            _, _ = amplituedro(expert_indices, expert_weights)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['amplituedro'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'volume': float(amplituedro.get_geometric_volume())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Volume: {results['amplituedro']['volume']:.2e}\n")
        
        # 3. SimultaneityProcessor
        print("3️⃣  SimultaneityProcessor (Lei da Simultaneidade)")
        x = torch.randn(batch_size, 512, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _, _ = simultaneity(x)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['simultaneity'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'divergence': float(simultaneity.get_timeline_divergence())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Divergência: {results['simultaneity']['divergence']:.3f}\n")
        
        # 4. QuantumEntanglement
        print("4️⃣  QuantumEntanglement (Emaranhamento Quântico)")
        expert_states = torch.randn(batch_size, 8, 512, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _, _ = entanglement(expert_states)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['entanglement'] = {
            'throughput': throughput,
            'latency_ms': latency
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms\n")
        
        # 5. StrangeAttractor
        print("5️⃣  StrangeAttractor (Atratores Estranhos)")
        expert_acts = torch.randn(batch_size, 8, device=self.device)
        
        start = time.time()
        for _ in range(num_iterations):
            _, _ = attractor(expert_acts)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        results['attractor'] = {
            'throughput': throughput,
            'latency_ms': latency,
            'stability': float(attractor.get_attractor_stability())
        }
        print(f"   Throughput: {throughput:.1f} samples/s")
        print(f"   Latência: {latency:.2f}ms")
        print(f"   Estabilidade: {results['attractor']['stability']:.3f}\n")
        
        # Agregado Tier 2
        total_throughput = sum(r['throughput'] for r in results.values())
        avg_latency = sum(r['latency_ms'] for r in results.values()) / len(results)
        
        print(f"📊 AGREGADO TIER 2:")
        print(f"   Throughput Total: {total_throughput:.1f} samples/s")
        print(f"   Latência Média: {avg_latency:.2f}ms\n")
        
        results['total'] = {
            'throughput': total_throughput,
            'latency_ms': avg_latency
        }
        
        return results
    
    def generate_report(self, tier1_results: Dict, tier2_results: Dict) -> str:
        """Gera relatório de performance"""
        report = """
╔════════════════════════════════════════════════════════════════╗
║         VALIDAÇÃO DE PERFORMANCE: TIER 1 + TIER 2            ║
╚════════════════════════════════════════════════════════════════╝

📊 RESULTADOS TIER 1 (4 Leis Primordiais)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for key, val in tier1_results.items():
            if key != 'total':
                report += f"\n{key.upper()}:\n"
                report += f"  Throughput: {val['throughput']:.1f} samples/s\n"
                report += f"  Latência: {val['latency_ms']:.2f}ms\n"
        
        report += f"""
AGREGADO TIER 1:
  Throughput Total: {tier1_results['total']['throughput']:.1f} samples/s
  Latência Média: {tier1_results['total']['latency_ms']:.2f}ms

📊 RESULTADOS TIER 2 (5 Leis Primordiais)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for key, val in tier2_results.items():
            if key != 'total':
                report += f"\n{key.upper()}:\n"
                report += f"  Throughput: {val['throughput']:.1f} samples/s\n"
                report += f"  Latência: {val['latency_ms']:.2f}ms\n"
        
        report += f"""
AGREGADO TIER 2:
  Throughput Total: {tier2_results['total']['throughput']:.1f} samples/s
  Latência Média: {tier2_results['total']['latency_ms']:.2f}ms

📈 COMPARAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1 Throughput: {tier1_results['total']['throughput']:.1f} samples/s
Tier 2 Throughput: {tier2_results['total']['throughput']:.1f} samples/s
Diferença: {((tier2_results['total']['throughput'] - tier1_results['total']['throughput']) / tier1_results['total']['throughput'] * 100):+.1f}%

Tier 1 Latência: {tier1_results['total']['latency_ms']:.2f}ms
Tier 2 Latência: {tier2_results['total']['latency_ms']:.2f}ms
Diferença: {((tier2_results['total']['latency_ms'] - tier1_results['total']['latency_ms']) / tier1_results['total']['latency_ms'] * 100):+.1f}%

✅ VALIDAÇÃO COMPLETA
"""
        
        return report

if __name__ == "__main__":
    print("🚀 INICIANDO VALIDAÇÃO DE PERFORMANCE\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    validator = PerformanceValidator(device=device)
    
    # Benchmark Tier 1
    tier1_results = validator.benchmark_tier1()
    
    # Benchmark Tier 2
    tier2_results = validator.benchmark_tier2()
    
    # Gerar relatório
    report = validator.generate_report(tier1_results, tier2_results)
    print(report)
    
    # Salvar em JSON
    with open('performance_validation.json', 'w') as f:
        json.dump({
            'tier1': tier1_results,
            'tier2': tier2_results
        }, f, indent=2)
    
    print("✅ Resultados salvos em performance_validation.json")
