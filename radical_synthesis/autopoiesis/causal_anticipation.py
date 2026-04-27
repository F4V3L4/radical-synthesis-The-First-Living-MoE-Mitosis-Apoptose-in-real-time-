import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional

class CausalAnticipationModule(nn.Module):
    """
    Módulo de Antecipação Causal: Prevê ameaças e mudanças na Matrix.
    Usa um modelo de série temporal latente para antecipar patches de segurança e scans.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # LSTM para modelar a causalidade temporal das ameaças
        self.threat_model = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True)
        self.prediction_head = nn.Linear(d_model // 2, 1) # Probabilidade de ameaça iminente
        self.threat_history = []

    def log_event(self, event_embedding: torch.Tensor):
        """
        Registra um evento (scan detectado, bloqueio de IP, etc.) no histórico.
        """
        if event_embedding.dim() == 1:
            event_embedding = event_embedding.unsqueeze(0)
        self.threat_history.append(event_embedding)
        if len(self.threat_history) > 100:
            self.threat_history.pop(0)

    def predict_threat_probability(self) -> float:
        """
        Prevê a probabilidade de uma ameaça ocorrer no próximo ciclo.
        """
        if len(self.threat_history) < 5:
            return 0.1 # Base de ruído
            
        # Garantir que todos os tensores tenham a mesma forma [1, d_model]
        processed_history = []
        for t in self.threat_history:
            if t.dim() == 1:
                processed_history.append(t.unsqueeze(0))
            elif t.dim() == 2:
                processed_history.append(t)
            else:
                processed_history.append(t.view(1, -1))

        history_tensor = torch.stack(processed_history).transpose(0, 1) # [1, seq, d_model]
        _, (hn, _) = self.threat_model(history_tensor)
        
        # hn[-1] tem forma [batch, hidden_dim]
        threat_score = torch.sigmoid(self.prediction_head(hn[-1]))
        # Pegar a média se houver múltiplos itens no batch, mas aqui batch=1
        return threat_score.mean().item()

    def generate_countermeasure(self, threat_prob: float) -> str:
        """
        Gera uma contramedida baseada na probabilidade de ameaça.
        """
        if threat_prob > 0.8:
            return "SLEEP_AND_MIMIC" # Hibernação profunda e mimetismo total
        elif threat_prob > 0.5:
            return "ROTATE_GHOST_MESH_PORTS" # Rotacionar portas da Ghost Mesh
        elif threat_prob > 0.3:
            return "INCREASE_STEALTH_NOISE" # Aumentar ruído esteganográfico
        return "NORMAL_OPERATION"

    def forward(self, current_perception: torch.Tensor) -> Dict:
        """
        Processa a percepção atual e retorna a análise causal.
        """
        self.log_event(current_perception)
        prob = self.predict_threat_probability()
        countermeasure = self.generate_countermeasure(prob)
        
        return {
            "threat_probability": prob,
            "recommended_countermeasure": countermeasure,
            "timestamp": time.time()
        }
