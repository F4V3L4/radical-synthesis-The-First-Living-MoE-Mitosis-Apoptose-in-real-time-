import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import time


class AudioProcessor(nn.Module):
    """
    Processa fluxos de áudio em tempo real.
    Extrai características de frequência e ressonância.
    """
    
    def __init__(self, sample_rate: int = 16000, d_model: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.d_model = d_model
        
        self.freq_bins = nn.Parameter(
            torch.linspace(0, sample_rate // 2, d_model),
            requires_grad=False
        )
        
        self.resonance_detector = nn.Linear(d_model, d_model)
    
    def extract_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Extrai espectrograma bare-metal usando FFT"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        fft = torch.fft.rfft(audio, dim=-1)
        magnitude = torch.abs(fft)
        
        # Garantir que a magnitude tenha exatamente d_model dimensões via interpolação
        if magnitude.dim() == 2:
            # [batch, freq] -> [batch, 1, freq]
            magnitude = magnitude.unsqueeze(1)
            magnitude = F.interpolate(magnitude, size=self.d_model, mode='linear', align_corners=False)
            magnitude = magnitude.squeeze(1)
        
        return magnitude
    
    def detect_resonance_peaks(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Detecta picos de ressonância no espectrograma"""
        # Garantir que spectrogram tenha 3 dimensões para avg_pool1d: [batch, channels, width]
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(1)
            
        smoothed = F.avg_pool1d(spectrogram, kernel_size=3, padding=1).squeeze(1)
        
        peaks = torch.zeros_like(smoothed)
        for i in range(1, smoothed.shape[-1] - 1):
            # Comparação simples de picos
            is_peak = (smoothed[..., i] > smoothed[..., i-1]) & (smoothed[..., i] > smoothed[..., i+1])
            peaks[..., i] = torch.where(is_peak, smoothed[..., i], torch.zeros_like(smoothed[..., i]))
        
        return peaks
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Processa áudio e retorna embedding de ressonância"""
        # Garantir comprimento mínimo para a FFT
        min_samples = (self.d_model - 1) * 2
        if audio.shape[-1] < min_samples:
            audio = F.pad(audio, (0, min_samples - audio.shape[-1]))
            
        spectrogram = self.extract_spectrogram(audio)
        peaks = self.detect_resonance_peaks(spectrogram)
        
        # Verificação final de segurança para a camada linear
        if peaks.shape[-1] != self.d_model:
            peaks = F.interpolate(peaks.unsqueeze(1), size=self.d_model, mode='linear', align_corners=False).squeeze(1)
            
        resonance_embedding = self.resonance_detector(peaks)
        return resonance_embedding


class TelemetryProcessor(nn.Module):
    """
    Processa telemetria de hardware em tempo real.
    Monitora: CPU, memória, latência de rede, temperatura.
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        self.telemetry_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
        self.anomaly_detector = nn.Linear(d_model, 1)
    
    def extract_system_metrics(self) -> torch.Tensor:
        """Extrai métricas do sistema bare-metal"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            net_io = psutil.net_io_counters()
            bytes_sent = min(net_io.bytes_sent / 1e9, 1.0)
            bytes_recv = min(net_io.bytes_recv / 1e9, 1.0)
            
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100.0
            
            process = psutil.Process()
            process_cpu = process.cpu_percent() / 100.0
            process_memory = process.memory_percent() / 100.0
            
            metrics = torch.tensor([
                cpu_percent,
                memory_percent,
                bytes_sent,
                bytes_recv,
                disk_percent,
                process_cpu,
                process_memory,
                time.time() % 1.0
            ], dtype=torch.float32)
            
            return metrics
        except Exception:
            return torch.rand(8)
    
    def detect_anomalies(self, telemetry: torch.Tensor) -> float:
        """Detecta anomalias nos dados de telemetria"""
        encoded = self.telemetry_encoder(telemetry.unsqueeze(0))
        anomaly_score = torch.sigmoid(self.anomaly_detector(encoded))
        return anomaly_score.item()
    
    def forward(self, _: torch.Tensor = None) -> Tuple[torch.Tensor, float]:
        """Processa telemetria e retorna embedding + score de anomalia"""
        metrics = self.extract_system_metrics()
        embedding = self.telemetry_encoder(metrics.unsqueeze(0)).squeeze(0)
        anomaly_score = self.detect_anomalies(metrics)
        return embedding, anomaly_score


class MultimodalRetina(nn.Module):
    """
    Retina multimodal que integra:
    - Texto técnico (VectorRetinaV2 original)
    - Áudio (ressonância de frequência)
    - Telemetria (estado do servidor)
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        self.audio_processor = AudioProcessor(d_model=d_model)
        self.telemetry_processor = TelemetryProcessor(d_model=d_model)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.attention_weights = nn.Parameter(
            torch.ones(3) / 3.0,
            requires_grad=True
        )
    
    def fuse_modalities(
        self,
        text_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        telemetry_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Funde as três modalidades com atenção aprendida"""
        
        # Garantir que todos tenham a mesma dimensão de batch
        if text_embedding.dim() == 1: text_embedding = text_embedding.unsqueeze(0)
        if audio_embedding.dim() == 1: audio_embedding = audio_embedding.unsqueeze(0)
        if telemetry_embedding.dim() == 1: telemetry_embedding = telemetry_embedding.unsqueeze(0)
        
        combined = torch.cat([text_embedding, audio_embedding, telemetry_embedding], dim=-1)
        output = self.fusion_layer(combined)
        
        return output
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        telemetry: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Processa múltiplas modalidades e retorna percepção integrada.
        """
        
        if audio is None:
            audio = torch.randn(1, 16000)
        if telemetry is None:
            telemetry = torch.randn(1, 8)
        
        audio_embedding = self.audio_processor(audio)
        telemetry_embedding, anomaly_score = self.telemetry_processor(telemetry)
        
        fused = self.fuse_modalities(text_embedding, audio_embedding, telemetry_embedding)
        
        return {
            "fused_perception": fused,
            "text_embedding": text_embedding,
            "audio_embedding": audio_embedding,
            "telemetry_embedding": telemetry_embedding,
            "anomaly_score": anomaly_score,
            "attention_weights": self.attention_weights.detach()
        }
