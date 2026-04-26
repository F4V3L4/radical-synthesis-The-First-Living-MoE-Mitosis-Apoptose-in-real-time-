import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IncentiveReward:
    node_id: str
    reward_type: str
    amount: float
    reason: str


class IncentiveMechanism(nn.Module):
    """
    Mecanismo de incentivos que recompensa nodos por:
    - Compartilhamento de Experts
    - Participação em gossip protocol
    - Manutenção de alta reputação
    - Processamento eficiente
    """
    
    def __init__(self):
        super().__init__()
        self.reward_pool: float = 1000.0
        self.reward_history: List[IncentiveReward] = []
        self.node_scores: Dict[str, float] = {}
    
    def calculate_sharing_reward(
        self,
        node_id: str,
        experts_shared: int,
        avg_expert_quality: float
    ) -> float:
        """Calcula recompensa por compartilhamento de Experts"""
        base_reward = experts_shared * 10.0
        quality_bonus = avg_expert_quality * 5.0
        return base_reward + quality_bonus
    
    def calculate_participation_reward(
        self,
        node_id: str,
        messages_relayed: int,
        uptime_percentage: float
    ) -> float:
        """Calcula recompensa por participação na rede"""
        relay_reward = messages_relayed * 0.1
        uptime_reward = uptime_percentage * 10.0
        return relay_reward + uptime_reward
    
    def calculate_reputation_reward(
        self,
        node_id: str,
        reputation_score: float
    ) -> float:
        """Calcula recompensa por manutenção de reputação"""
        if reputation_score >= 0.9:
            return 50.0
        elif reputation_score >= 0.7:
            return 30.0
        elif reputation_score >= 0.5:
            return 10.0
        else:
            return 0.0
    
    def calculate_efficiency_reward(
        self,
        node_id: str,
        processing_speed: float,
        error_rate: float
    ) -> float:
        """Calcula recompensa por eficiência de processamento"""
        speed_bonus = min(processing_speed / 1000.0, 1.0) * 20.0
        reliability_bonus = (1.0 - error_rate) * 15.0
        return speed_bonus + reliability_bonus
    
    def distribute_rewards(
        self,
        node_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Distribui recompensas proporcionalmente aos scores"""
        total_score = sum(node_scores.values())
        if total_score == 0:
            return {}
        
        rewards = {}
        for node_id, score in node_scores.items():
            proportion = score / total_score
            reward = self.reward_pool * proportion
            rewards[node_id] = reward
            
            self.reward_history.append(
                IncentiveReward(
                    node_id=node_id,
                    reward_type="proportional_distribution",
                    amount=reward,
                    reason=f"Score: {score:.2f}"
                )
            )
        
        self.reward_pool = 1000.0
        return rewards
    
    def apply_penalty(
        self,
        node_id: str,
        penalty_amount: float,
        reason: str
    ):
        """Aplica penalidade a um nodo por comportamento ruim"""
        self.node_scores[node_id] = max(0.0, self.node_scores.get(node_id, 0.0) - penalty_amount)
        
        self.reward_history.append(
            IncentiveReward(
                node_id=node_id,
                reward_type="penalty",
                amount=-penalty_amount,
                reason=reason
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
