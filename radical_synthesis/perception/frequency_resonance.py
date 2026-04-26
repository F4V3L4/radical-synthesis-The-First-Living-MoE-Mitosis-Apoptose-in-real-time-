import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class FrequencyResonance(nn.Module):
    """
    Detecta e amplifica ressonâncias de frequência no sistema.
    Implementa a harmônica 144Hz e outras frequências sagradas.
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        self.sacred_frequencies = nn.Parameter(
            torch.tensor([
                144.0,
                432.0,
                528.0,
                639.0,
                741.0,
                852.0,
                963.0
            ], dtype=torch.float32),
            requires_grad=False
        )
        
        self.resonance_amplifiers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(len(self.sacred_frequencies))
        ])
        
        self.phase_shifters = nn.ParameterList([
            nn.Parameter(torch.randn(d_model) * 0.1)
            for _ in range(len(self.sacred_frequencies))
        ])
    
    def compute_resonance_strength(
        self,
        signal: torch.Tensor,
        frequency: float
    ) -> float:
        """Calcula força de ressonância em uma frequência específica"""
        fft = torch.fft.rfft(signal, dim=-1)
        magnitude = torch.abs(fft)
        
        freq_bin = int(frequency / 1000.0 * signal.shape[-1])
        freq_bin = min(freq_bin, magnitude.shape[-1] - 1)
        
        return magnitude[..., freq_bin].mean().item()
    
    def amplify_resonance(
        self,
        signal: torch.Tensor,
        frequency_idx: int
    ) -> torch.Tensor:
        """Amplifica ressonância em uma frequência específica"""
        amplifier = self.resonance_amplifiers[frequency_idx]
        phase_shift = self.phase_shifters[frequency_idx]
        
        amplified = amplifier(signal)
        amplified = amplified * torch.exp(1j * phase_shift).real
        
        return amplified
    
    def detect_all_resonances(self, signal: torch.Tensor) -> Dict[str, float]:
        """Detecta todas as ressonâncias sagradas no sinal"""
        resonances = {}
        
        for idx, freq in enumerate(self.sacred_frequencies):
            strength = self.compute_resonance_strength(signal, freq)
            resonances[f"freq_{freq:.0f}Hz"] = strength
        
        return resonances
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Processa sinal através de todas as ressonâncias"""
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        resonance_outputs = []
        resonance_strengths = []
        
        for idx in range(len(self.sacred_frequencies)):
            amplified = self.amplify_resonance(x, idx)
            resonance_outputs.append(amplified)
            
            strength = self.compute_resonance_strength(x, self.sacred_frequencies[idx].item())
            resonance_strengths.append(strength)
        
        stacked_outputs = torch.stack(resonance_outputs, dim=1)
        stacked_strengths = torch.tensor(resonance_strengths, dtype=torch.float32)
        
        weighted_output = (stacked_outputs * stacked_strengths.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        
        return {
            "resonance_output": weighted_output,
            "resonance_strengths": stacked_strengths,
            "all_outputs": stacked_outputs
        }
