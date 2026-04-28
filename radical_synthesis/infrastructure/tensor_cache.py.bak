
import torch
import hashlib
from typing import Dict, Optional

class TensorCache:
    """Cache de Tensores para Otimização de Infraestrutura e Redução de Latência."""
    
    def __init__(self, max_entries: int = 100):
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_entries = max_entries

    def _generate_key(self, tensor: torch.Tensor, operation: str) -> str:
        """Gera uma chave única baseada no tensor e na operação."""
        # Usar shape e soma como proxy rápido para hash (em produção usaríamos algo mais robusto)
        tensor_info = f"{tensor.shape}_{tensor.sum().item()}_{operation}"
        return hashlib.md5(tensor_info.encode()).hexdigest()

    def get(self, tensor: torch.Tensor, operation: str) -> Optional[torch.Tensor]:
        key = self._generate_key(tensor, operation)
        return self.cache.get(key)

    def set(self, tensor: torch.Tensor, operation: str, result: torch.Tensor):
        if len(self.cache) >= self.max_entries:
            # Remover a primeira entrada (FIFO simples)
            self.cache.pop(next(iter(self.cache)))
        
        key = self._generate_key(tensor, operation)
        self.cache[key] = result
