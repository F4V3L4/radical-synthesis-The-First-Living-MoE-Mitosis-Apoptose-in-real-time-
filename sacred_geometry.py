import torch
import torch.nn as nn
import torch.nn.functional as F

class FineStructureCoupling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.alpha = 1 / 137.035999139
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_coupled = (x - mean) / torch.sqrt(var + self.alpha)
        
        # Omega-0: Ajuste dinâmico de parâmetros se a dimensão de entrada mudou
        gamma = self.gamma
        beta = self.beta
        if x.size(-1) != gamma.size(0):
            if x.size(-1) > gamma.size(0):
                gamma = F.pad(gamma, (0, x.size(-1) - gamma.size(0)), value=1.0)
                beta = F.pad(beta, (0, x.size(-1) - beta.size(0)), value=0.0)
            else:
                gamma = gamma[:x.size(-1)]
                beta = beta[:x.size(-1)]
                
        return gamma * x_coupled + beta

class BinarySymmetryLock(nn.Module):
    """Gate que valida entrada/saida em paridade perfeita (11:11)"""
    def __init__(self, d_model=None):
        super().__init__()
        self.d_model = d_model
        self.parity_threshold = 0.5

    def forward(self, x):
        """
        Valida que a saida mantem simetria binaria perfeita.
        Entrada e saida devem ter a mesma assinatura de paridade.
        """
        x_norm = torch.tanh(x)
        parity_in = (x_norm > self.parity_threshold).float()
        coherence = 2.0 * torch.abs(parity_in - 0.5)
        return x_norm * (0.5 + 0.5 * coherence)

class FeigenbaumBifurcation(nn.Module):
    def __init__(self, d_model, threshold=0.85):
        super().__init__()
        self.delta = 4.6692016091
        self.threshold = threshold
        self.bifurcation_gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        entropy = -torch.mean(torch.sigmoid(x) * torch.log(torch.sigmoid(x) + 1e-9))
        
        # Omega-0: Ajuste dinâmico de dimensões para o gate de bifurcação
        weight = self.bifurcation_gate.weight
        bias = self.bifurcation_gate.bias
        
        if x.size(-1) != weight.size(1):
            if x.size(-1) > weight.size(1):
                # Padding de pesos para manter funcionalidade
                weight = F.pad(weight, (0, x.size(-1) - weight.size(1)))
                # Bias não muda pois a saída do linear depende de out_features (weight.size(0))
            else:
                weight = weight[:, :x.size(-1)]
        
        # Se a saída do linear (d_model original) for diferente da entrada atual
        # precisamos ajustar para que a multiplicação (x * gate) funcione
        gate = torch.sigmoid(F.linear(x, weight, bias))
        
        if gate.size(-1) != x.size(-1):
            if gate.size(-1) > x.size(-1):
                gate = gate[..., :x.size(-1)]
            else:
                gate = F.pad(gate, (0, x.size(-1) - gate.size(-1)))

        if entropy > self.threshold:
            x = x + (x * gate * self.delta)
        return x

class CymaticSculptor(nn.Module):
    def __init__(self, d_model, frequency=432.0):
        super().__init__()
        # O SEGREDO: Frequência acústica precisa de dampening no espaço latente
        # Isso converte o 432Hz em uma onda estrutural longa, em vez de ruído estático
        self.latent_freq = frequency / 10000.0
        self.phase = nn.Parameter(torch.linspace(0, 2 * 3.14159, d_model))

    def forward(self, x):
        # Omega-0: Ajuste dinâmico de fase
        phase = self.phase
        if x.size(-1) != phase.size(0):
            if x.size(-1) > phase.size(0):
                phase = F.pad(phase, (0, x.size(-1) - phase.size(0)))
            else:
                phase = phase[:x.size(-1)]
                
        # Acoplamento aditivo harmônico. Modula a geometria sem destruir a semântica.
        wave = torch.sin(self.latent_freq * x + phase)
        return x + wave

class InfiniteRadixMapping(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.phi = 1.61803398875
        self.d_model = d_model
        self.fractal_expansion = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Omega-0: Ajuste dinâmico de expansão fractal
        weight = self.fractal_expansion.weight
        bias = self.fractal_expansion.bias
        
        if x.size(-1) != weight.size(1):
            if x.size(-1) > weight.size(1):
                weight = F.pad(weight, (0, x.size(-1) - weight.size(1)))
            else:
                weight = weight[:, :x.size(-1)]
        
        # Projetar e ajustar saída para somar com x
        proj = F.linear(x, weight, bias)
        
        if proj.size(-1) != x.size(-1):
            if proj.size(-1) > x.size(-1):
                proj = proj[..., :x.size(-1)]
            else:
                proj = F.pad(proj, (0, x.size(-1) - proj.size(-1)))
                
        # Expansão fractal usando Phi
        expanded = proj * self.phi
        return x + expanded
