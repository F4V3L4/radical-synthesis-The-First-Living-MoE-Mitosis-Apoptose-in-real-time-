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


class VideoProcessor(nn.Module):
    """
    Processa fluxos de vídeo em tempo real.
    Extrai características visuais e de movimento.
    """

    def __init__(self, d_model: int = 512, frame_size: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.d_model = d_model
        self.frame_size = frame_size

        # CNN para extração de características de frames
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Calcular o tamanho da saída da CNN
        # Assumindo entrada de 3 canais (RGB), frame_size x frame_size
        dummy_input = torch.randn(1, 3, frame_size[0], frame_size[1])
        with torch.no_grad():
            cnn_output_size = self.cnn_encoder(dummy_input).shape[1]

        self.linear_projection = nn.Linear(cnn_output_size, d_model)

    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Processa um único frame de vídeo.
        frame: Tensor de imagem [C, H, W] ou [B, C, H, W]
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0) # Adiciona dimensão de batch

        # Redimensionar frame para o tamanho esperado pela CNN
        frame = F.interpolate(frame, size=self.frame_size, mode='bilinear', align_corners=False)

        features = self.cnn_encoder(frame)
        embedding = self.linear_projection(features)
        return embedding.squeeze(0) # Remove dimensão de batch se foi adicionada

    def forward(self, video_frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Processa uma sequência de frames de vídeo e retorna um embedding agregado.
        video_frames: Lista de tensores de frames [C, H, W].
        """
        if not video_frames:
            return torch.zeros(1, self.d_model) # Retorna embedding zero se não houver frames

        # Empilhar todos os frames em um único tensor de batch para processamento eficiente
        frames_tensor = torch.stack(video_frames) # [Batch, C, H, W]
        
        # Redimensionar frames para o tamanho esperado pela CNN
        frames_tensor = F.interpolate(frames_tensor, size=self.frame_size, mode='bilinear', align_corners=False)

        features = self.cnn_encoder(frames_tensor)
        embeddings = self.linear_projection(features) # [Batch, d_model]
        
        return embeddings


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
    
    def detect_anomalies(self, telemetry: torch.Tensor) -> torch.Tensor:
        """Detecta anomalias nos dados de telemetria"""
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)
        encoded = self.telemetry_encoder(telemetry)
        anomaly_score = torch.sigmoid(self.anomaly_detector(encoded))
        return anomaly_score
    
    def forward(self, telemetry: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processa telemetria e retorna embedding + score de anomalia"""
        if telemetry is None:
            telemetry = self.extract_system_metrics().unsqueeze(0)
        
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)
            
        embedding = self.telemetry_encoder(telemetry)
        anomaly_score = self.detect_anomalies(telemetry)
        return embedding, anomaly_score


class MultimodalRetina(nn.Module):
    """
    Retina multimodal que integra:
    - Texto técnico (VectorRetinaV2 original)
    - Áudio (ressonância de frequência)
    - Telemetria (estado do servidor)
    """
    
    def __init__(self, d_model: int = 512, vocab_size: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.text_encoder = nn.Embedding(vocab_size, d_model) # Adicionado para processar tokens
        self.audio_processor = AudioProcessor(d_model=d_model)
        self.telemetry_processor = TelemetryProcessor(d_model=d_model)
        self.video_processor = VideoProcessor(d_model=d_model)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.attention_weights = nn.Parameter(
            torch.ones(4) / 4.0, # Atualizado para 4 modalidades (texto, áudio, telemetria, vídeo)
            requires_grad=True
        )
    
    def fuse_modalities(
        self,
        text_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        telemetry_embedding: torch.Tensor,
        video_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Funde as quatro modalidades com atenção aprendida"""
        
        # Garantir que todos tenham a mesma dimensão de batch (dim 0)
        # E que todos sejam tensores 2D [batch, d_model]
        def ensure_2d(t):
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        text_embedding = ensure_2d(text_embedding)
        audio_embedding = ensure_2d(audio_embedding)
        telemetry_embedding = ensure_2d(telemetry_embedding)
        video_embedding = ensure_2d(video_embedding)
        
        # Sincronizar dimensão de batch se necessário (usando o batch do texto como referência)
        batch_size = text_embedding.shape[0]
        if audio_embedding.shape[0] != batch_size:
            audio_embedding = audio_embedding.expand(batch_size, -1).contiguous()
        if telemetry_embedding.shape[0] != batch_size:
            telemetry_embedding = telemetry_embedding.expand(batch_size, -1).contiguous()
        if video_embedding.shape[0] != batch_size:
            video_embedding = video_embedding.expand(batch_size, -1).contiguous()
        
        combined = torch.cat([text_embedding, audio_embedding, telemetry_embedding, video_embedding], dim=-1).contiguous()
        output = self.fusion_layer(combined)
        
        return output
    
    def forward(
        self,
        text_tokens: torch.Tensor, # Agora recebe tokens diretamente
        audio: Optional[torch.Tensor] = None,
        telemetry: Optional[torch.Tensor] = None,
        video_frames: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Processa múltiplas modalidades e retorna percepção integrada.
        """
        # Bare-metal Fix: Garantir que text_tokens seja Long para o embedding
        if text_tokens.dtype != torch.long:
            text_tokens = text_tokens.long()
            
        # Codificar tokens de texto em embedding
        text_embedding = self.text_encoder(text_tokens).mean(dim=1) # Média dos embeddings dos tokens

        # Garantir que text_embedding seja [Batch, d_model]
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        batch_size = text_embedding.shape[0]

        if audio is None:
            audio = torch.randn(batch_size, 16000)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        if telemetry is None:
            telemetry = torch.randn(batch_size, 8)
        elif telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)
            
        if video_frames is None:
            video_frames = [torch.randn(3, 64, 64) for _ in range(batch_size)]
        
        audio_embedding = self.audio_processor(audio)
        telemetry_embedding, anomaly_score = self.telemetry_processor(telemetry)
        video_embedding = self.video_processor(video_frames)
        
        # Sincronização de batch para video_embedding
        # VideoProcessor retorna [len(video_frames), d_model]
        if video_embedding.shape[0] != batch_size:
            if video_embedding.shape[0] > batch_size:
                # Se temos mais frames que o batch, truncamos (simplificação)
                video_embedding = video_embedding[:batch_size]
            else:
                # Se temos menos, expandimos
                video_embedding = video_embedding.expand(batch_size, -1).contiguous()
        
        fused = self.fuse_modalities(text_embedding, audio_embedding, telemetry_embedding, video_embedding)
        
        return {
            "fused_perception": fused,
            "text_embedding": text_embedding,
            "audio_embedding": audio_embedding,
            "telemetry_embedding": telemetry_embedding,
            "video_embedding": video_embedding,
            "anomaly_score": anomaly_score,
            "attention_weights": self.attention_weights.detach()
        }
