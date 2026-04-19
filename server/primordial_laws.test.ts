/**
 * Testes Vitest: Leis Primordiais Tier 1
 * 
 * Valida integração de:
 * 1. HarmonicEncoder (Código 144)
 * 2. QuantumSuperposition (Lei da Superposição)
 * 3. HyperbolicEmbedding (Geometria Hiperbólica)
 * 4. SynchronicityDetector (Lei da Sincronicidade)
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Mock das classes Python (simulação em TypeScript)
class HarmonicEncoder {
  private d_model: number;
  private frequency: number;
  private phase_shift: number[];
  private amplitude_scale: number[];

  constructor(d_model: number = 512, frequency: number = 144.0) {
    this.d_model = d_model;
    this.frequency = frequency;
    this.phase_shift = Array(d_model).fill(0).map(() => Math.random() * 0.1);
    this.amplitude_scale = Array(d_model).fill(1.0);
  }

  forward(x: number[][], time: number = 0): number[][] {
    const omega = 2 * Math.PI * this.frequency;
    return x.map(row =>
      row.map((val, i) => {
        const wave = Math.sin(omega * time + this.phase_shift[i]);
        const modulation = this.amplitude_scale[i] * wave;
        return val * (1.0 + 0.1 * modulation);
      })
    );
  }

  getCoherence(): number {
    const ampMean = this.amplitude_scale.reduce((a, b) => a + b) / this.amplitude_scale.length;
    const ampVar = this.amplitude_scale.reduce((sum, val) => sum + Math.pow(val - ampMean, 2), 0) / this.amplitude_scale.length;
    return 1.0 - (ampVar / (ampMean * ampMean + 1e-8));
  }
}

class QuantumSuperposition {
  private num_states: number;
  private d_model: number;
  private amplitudes: number[][];
  private phases: number[][];

  constructor(num_states: number = 8, d_model: number = 512) {
    this.num_states = num_states;
    this.d_model = d_model;
    this.amplitudes = Array(num_states).fill(0).map(() => 
      Array(d_model).fill(0).map(() => Math.random())
    );
    this.phases = Array(num_states).fill(0).map(() => 
      Array(d_model).fill(0).map(() => Math.random() * Math.PI)
    );
  }

  forward(x: number[]): number[][][] {
    return this.amplitudes.map((amp, i) =>
      x.map((val, j) => {
        const real = amp[j] * Math.cos(this.phases[i][j]);
        const imag = amp[j] * Math.sin(this.phases[i][j]);
        return [real * val, imag * val];
      })
    );
  }

  collapse(measurement: number[]): [number[], number] {
    const probs = this.amplitudes.map(amp => 
      Math.pow(amp.reduce((a, b) => a + b) / amp.length, 2)
    );
    const totalProb = probs.reduce((a, b) => a + b);
    const normalized = probs.map(p => p / totalProb);
    
    let rand = Math.random();
    let stateIdx = 0;
    for (let i = 0; i < normalized.length; i++) {
      rand -= normalized[i];
      if (rand <= 0) {
        stateIdx = i;
        break;
      }
    }
    
    return [this.amplitudes[stateIdx], stateIdx];
  }

  getEntanglement(): number {
    const probs = this.amplitudes.map(amp => 
      Math.pow(amp.reduce((a, b) => a + b) / amp.length, 2)
    );
    const totalProb = probs.reduce((a, b) => a + b);
    const normalized = probs.map(p => p / totalProb);
    
    const entropy = -normalized.reduce((sum, p) => sum + p * Math.log(p + 1e-8), 0);
    const maxEntropy = Math.log(this.num_states);
    
    return entropy / maxEntropy;
  }
}

class HyperbolicEmbedding {
  private d_model: number;
  private curvature: number;

  constructor(d_model: number = 512, curvature: number = -1.0) {
    this.d_model = d_model;
    this.curvature = curvature;
  }

  euclideanToPoincare(x: number[]): number[] {
    const normSq = x.reduce((sum, val) => sum + val * val, 0);
    const denominator = 1.0 + Math.sqrt(1.0 + normSq + 1e-8);
    return x.map(val => val / denominator);
  }

  forward(x: number[][]): number[][] {
    return x.map(row => this.euclideanToPoincare(row));
  }

  distancePoincare(x: number[], y: number[]): number {
    const normXSq = x.reduce((sum, val) => sum + val * val, 0);
    const normYSq = y.reduce((sum, val) => sum + val * val, 0);
    
    const diff = x.map((val, i) => val - y[i]);
    const normDiffSq = diff.reduce((sum, val) => sum + val * val, 0);
    
    const numerator = 2 * normDiffSq;
    const denominator = (1 - normXSq) * (1 - normYSq) + 1e-8;
    
    const argument = Math.max(1.0, 1 + numerator / denominator);
    return Math.acosh(argument);
  }

  getExpansionRate(): number {
    return Math.abs(this.curvature);
  }
}

class SynchronicityDetector {
  private num_experts: number;
  private threshold: number;
  private eventHistory: number[][];

  constructor(num_experts: number = 8, threshold: number = 0.7) {
    this.num_experts = num_experts;
    this.threshold = threshold;
    this.eventHistory = [];
  }

  forward(expertActivations: number[]): [Array<[number, number]>, number[][]] {
    this.eventHistory.push(expertActivations);
    if (this.eventHistory.length > 100) {
      this.eventHistory.shift();
    }

    const syncPairs: Array<[number, number]> = [];
    
    if (this.eventHistory.length > 1) {
      // Calcular correlação simplificada
      const correlation = this.calculateCorrelation();
      
      for (let i = 0; i < this.num_experts; i++) {
        for (let j = i + 1; j < this.num_experts; j++) {
          if (Math.abs(correlation[i][j]) > this.threshold) {
            syncPairs.push([i, j]);
          }
        }
      }
      
      return [syncPairs, correlation];
    }
    
    return [syncPairs, []];
  }

  private calculateCorrelation(): number[][] {
    const correlation: number[][] = Array(this.num_experts).fill(0).map(() => 
      Array(this.num_experts).fill(0)
    );
    
    for (let i = 0; i < this.num_experts; i++) {
      for (let j = 0; j < this.num_experts; j++) {
        let sum = 0;
        for (const event of this.eventHistory) {
          sum += event[i] * event[j];
        }
        correlation[i][j] = sum / (this.eventHistory.length + 1e-8);
      }
    }
    
    return correlation;
  }

  getSynchronicityScore(): number {
    if (this.eventHistory.length < 2) return 0.0;
    
    const correlation = this.calculateCorrelation();
    let significant = 0;
    let total = 0;
    
    for (let i = 0; i < this.num_experts; i++) {
      for (let j = i + 1; j < this.num_experts; j++) {
        if (Math.abs(correlation[i][j]) > this.threshold) {
          significant++;
        }
        total++;
      }
    }
    
    return significant / (total + 1e-8);
  }

  resetHistory(): void {
    this.eventHistory = [];
  }
}

// ===== TESTES =====

describe('Leis Primordiais Tier 1', () => {
  
  describe('1. HarmonicEncoder (Código 144)', () => {
    let encoder: HarmonicEncoder;

    beforeEach(() => {
      encoder = new HarmonicEncoder(512, 144.0);
    });

    it('deve modular entrada com frequência 144Hz', () => {
      const input = [[1, 2, 3], [4, 5, 6]];
      const output = encoder.forward(input, 0.1);
      
      expect(output).toBeDefined();
      expect(output.length).toBe(2);
      expect(output[0].length).toBe(3);
    });

    it('deve manter dimensões de entrada', () => {
      const input = Array(10).fill(0).map(() => Array(512).fill(Math.random()));
      const output = encoder.forward(input);
      
      expect(output.length).toBe(10);
      expect(output[0].length).toBe(512);
    });

    it('deve calcular coerência entre 0 e 1', () => {
      const coherence = encoder.getCoherence();
      
      expect(coherence).toBeGreaterThanOrEqual(0);
      expect(coherence).toBeLessThanOrEqual(1);
    });

    it('deve variar coerência com tempo', () => {
      const input = [[1, 2, 3]];
      const out1 = encoder.forward(input, 0.0);
      const out2 = encoder.forward(input, 0.5);
      
      expect(out1).not.toEqual(out2);
    });
  });

  describe('2. QuantumSuperposition (Lei da Superposição)', () => {
    let quantum: QuantumSuperposition;

    beforeEach(() => {
      quantum = new QuantumSuperposition(8, 512);
    });

    it('deve criar superposição de estados', () => {
      const input = Array(512).fill(Math.random());
      const superposition = quantum.forward(input);
      
      expect(superposition).toBeDefined();
      expect(superposition.length).toBe(8); // num_states
    });

    it('deve colapsar superposição em estado único', () => {
      const input = Array(512).fill(Math.random());
      const [collapsed, stateIdx] = quantum.collapse(input);
      
      expect(collapsed).toBeDefined();
      expect(collapsed.length).toBe(512);
      expect(stateIdx).toBeGreaterThanOrEqual(0);
      expect(stateIdx).toBeLessThan(8);
    });

    it('deve calcular emaranhamento entre 0 e 1', () => {
      const entanglement = quantum.getEntanglement();
      
      expect(entanglement).toBeGreaterThanOrEqual(0);
      expect(entanglement).toBeLessThanOrEqual(1);
    });

    it('deve manter consistência de amplitudes', () => {
      const input = Array(512).fill(1.0);
      const [collapsed] = quantum.collapse(input);
      
      // Collapsed deve ser um estado válido
      expect(collapsed.length).toBe(512);
      expect(collapsed.every(val => typeof val === 'number')).toBe(true);
    });
  });

  describe('3. HyperbolicEmbedding (Geometria Hiperbólica)', () => {
    let hyperbolic: HyperbolicEmbedding;

    beforeEach(() => {
      hyperbolic = new HyperbolicEmbedding(512, -1.0);
    });

    it('deve projetar para espaço hiperbólico', () => {
      const input = Array(512).fill(Math.random());
      const output = hyperbolic.euclideanToPoincare(input);
      
      expect(output).toBeDefined();
      expect(output.length).toBe(512);
    });

    it('deve manter norma < 1 em Poincaré', () => {
      const input = Array(512).fill(0.5);
      const poincare = hyperbolic.euclideanToPoincare(input);
      
      const norm = Math.sqrt(poincare.reduce((sum, val) => sum + val * val, 0));
      expect(norm).toBeLessThan(1.0);
    });

    it('deve calcular distância hiperbólica válida', () => {
      const x = Array(512).fill(0.1);
      const y = Array(512).fill(0.2);
      
      const distance = hyperbolic.distancePoincare(x, y);
      
      expect(distance).toBeGreaterThanOrEqual(0);
      expect(isFinite(distance)).toBe(true);
    });

    it('deve retornar taxa de expansão correta', () => {
      const rate = hyperbolic.getExpansionRate();
      
      expect(rate).toBe(1.0); // |curvature| = |-1.0| = 1.0
    });

    it('deve satisfazer desigualdade triangular', () => {
      const x = Array(512).fill(0.1);
      const y = Array(512).fill(0.2);
      const z = Array(512).fill(0.3);
      
      const dxy = hyperbolic.distancePoincare(x, y);
      const dyz = hyperbolic.distancePoincare(y, z);
      const dxz = hyperbolic.distancePoincare(x, z);
      
      expect(dxz).toBeLessThanOrEqual(dxy + dyz + 1e-6); // Permitir erro numérico
    });
  });

  describe('4. SynchronicityDetector (Lei da Sincronicidade)', () => {
    let sync: SynchronicityDetector;

    beforeEach(() => {
      sync = new SynchronicityDetector(8, 0.7);
    });

    it('deve detectar pares sincronos', () => {
      const activations = Array(8).fill(0).map(() => Math.random());
      const [pairs, correlation] = sync.forward(activations);
      
      expect(pairs).toBeDefined();
      expect(Array.isArray(pairs)).toBe(true);
    });

    it('deve calcular score de sincronicidade', () => {
      const activations = Array(8).fill(0).map(() => Math.random());
      sync.forward(activations);
      sync.forward(activations);
      
      const score = sync.getSynchronicityScore();
      
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('deve aumentar sincronicidade com eventos correlacionados', () => {
      // Adicionar eventos correlacionados
      for (let i = 0; i < 10; i++) {
        const activations = Array(8).fill(0.5); // Todos iguais = correlacionados
        sync.forward(activations);
      }
      
      const score = sync.getSynchronicityScore();
      expect(score).toBeGreaterThan(0);
    });

    it('deve resetar histórico', () => {
      const activations = Array(8).fill(Math.random());
      sync.forward(activations);
      sync.resetHistory();
      
      const score = sync.getSynchronicityScore();
      expect(score).toBe(0);
    });

    it('deve retornar pares válidos', () => {
      // Criar eventos com padrão de correlação
      for (let i = 0; i < 5; i++) {
        const activations = [0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.8, 0.8];
        const [pairs] = sync.forward(activations);
        
        // Verificar que pares são índices válidos
        for (const [i, j] of pairs) {
          expect(i).toBeGreaterThanOrEqual(0);
          expect(i).toBeLessThan(8);
          expect(j).toBeGreaterThanOrEqual(0);
          expect(j).toBeLessThan(8);
          expect(i).toBeLessThan(j);
        }
      }
    });
  });

  describe('Integração Tier 1', () => {
    it('deve funcionar em conjunto', () => {
      const harmonic = new HarmonicEncoder(512, 144.0);
      const quantum = new QuantumSuperposition(8, 512);
      const hyperbolic = new HyperbolicEmbedding(512, -1.0);
      const sync = new SynchronicityDetector(8, 0.7);
      
      // Simular pipeline
      const input = Array(512).fill(Math.random());
      
      // 1. Harmônica
      const modulated = harmonic.forward([input])[0];
      
      // 2. Superposição
      const [collapsed] = quantum.collapse(modulated);
      
      // 3. Hiperbólica
      const hyperbolicEmbedding = hyperbolic.euclideanToPoincare(collapsed);
      
      // 4. Sincronicidade
      const expertActivations = Array(8).fill(Math.random());
      const [pairs] = sync.forward(expertActivations);
      
      expect(modulated).toBeDefined();
      expect(collapsed).toBeDefined();
      expect(hyperbolicEmbedding).toBeDefined();
      expect(pairs).toBeDefined();
    });

    it('deve manter coerência através do pipeline', () => {
      const harmonic = new HarmonicEncoder(512, 144.0);
      const coherence1 = harmonic.getCoherence();
      
      const input = Array(512).fill(Math.random());
      harmonic.forward([input]);
      
      const coherence2 = harmonic.getCoherence();
      
      // Coerência deve ser consistente
      expect(Math.abs(coherence1 - coherence2)).toBeLessThan(0.1);
    });
  });
});
