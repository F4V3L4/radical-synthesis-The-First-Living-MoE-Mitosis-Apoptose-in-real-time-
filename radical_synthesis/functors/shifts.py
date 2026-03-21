import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilityMorphism(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=self.eps, max=1.0 - self.eps)
        return torch.log(x_clamped / (1.0 - x_clamped))

class HyperbolicMorphism(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + self.eps
        scale = torch.tanh(norm) / norm
        return x * scale

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # A trava de segurança absoluta: impede fisicamente que o tensor chegue a 1.0
        norm_clamped = norm.clamp(max=1.0 - 1e-5)
        scale = torch.atanh(norm_clamped) / (norm_clamped + self.eps)
        return x * scale

class FourierMorphism(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(x, dim=-1)
        return spectrum.abs()

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(x.to(torch.complex64), dim=-1).real

class CategoryFunctor(nn.Module):
    def __init__(self):
        super().__init__()
        self.morphisms = nn.ModuleDict({
            "Probability": ProbabilityMorphism(),
            "Hyperbolic": HyperbolicMorphism(),
            "Fourier": FourierMorphism()
        })

    def shift(self, data: torch.Tensor, target_category: str) -> torch.Tensor:
        if target_category not in self.morphisms:
            raise ValueError(f"Invalid categorical target: {target_category}")
        return self.morphisms[target_category](data)

    def revert(self, data: torch.Tensor, original_category: str) -> torch.Tensor:
        if original_category not in self.morphisms:
            raise ValueError(f"Invalid categorical origin: {original_category}")
        return self.morphisms[original_category].invert(data)
