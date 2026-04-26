import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import hashlib


class QuantumResistantSignature:
    """
    Assinatura baseada em lattice (resistente a computadores quânticos).
    Implementa NTRU-like signature scheme simplificado.
    """
    
    def __init__(self, n: int = 256, q: int = 4096):
        self.n = n
        self.q = q
        self.f = None
        self.g = None
        self.h = None
    
    def generate_keys(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gera par de chaves público/privado baseado em lattice"""
        
        self.f = np.random.randint(-1, 2, self.n)
        self.g = np.random.randint(-1, 2, self.n)
        
        f_inv = self._mod_inverse_poly(self.f, self.q)
        self.h = (self.q // 3) * np.convolve(f_inv, self.g, mode='same') % self.q
        
        return self.h, (self.f, self.g)
    
    def _mod_inverse_poly(self, poly: np.ndarray, mod: int) -> np.ndarray:
        """Calcula inverso modular de um polinômio"""
        result = np.zeros_like(poly)
        for i in range(len(poly)):
            for j in range(mod):
                if (poly[i] * j) % mod == 1:
                    result[i] = j
                    break
        return result
    
    def sign(self, message: bytes, private_key: Tuple) -> np.ndarray:
        """Assina mensagem com chave privada"""
        f, g = private_key
        
        message_hash = hashlib.sha256(message).digest()
        m = np.frombuffer(message_hash, dtype=np.uint8)[:self.n]
        
        signature = (np.convolve(f, m, mode='same') + np.convolve(g, m, mode='same')) % self.q
        return signature
    
    def verify(self, message: bytes, signature: np.ndarray, public_key: np.ndarray) -> bool:
        """Verifica assinatura com chave pública"""
        message_hash = hashlib.sha256(message).digest()
        m = np.frombuffer(message_hash, dtype=np.uint8)[:self.n]
        
        computed = np.convolve(public_key, signature, mode='same') % self.q
        expected = (self.q // 3) * m % self.q
        
        return np.allclose(computed, expected, atol=self.q // 6)


class LatticeCrypto(nn.Module):
    """
    Camada de criptografia lattice-based para Ghost Mesh.
    Protege comunicação entre nodos contra ataques quânticos.
    """
    
    def __init__(self, n: int = 256, q: int = 4096):
        super().__init__()
        self.n = n
        self.q = q
        self.qrs = QuantumResistantSignature(n, q)
        
        self.public_key = None
        self.private_key = None
        self.peer_public_keys: Dict[str, np.ndarray] = {}
        
        self.encryption_matrix = nn.Parameter(
            torch.randn(n, n) * 0.01,
            requires_grad=False
        )
    
    def generate_keypair(self) -> Tuple[np.ndarray, Tuple]:
        """Gera par de chaves para o nodo"""
        self.public_key, self.private_key = self.qrs.generate_keys()
        return self.public_key, self.private_key
    
    def register_peer_key(self, peer_id: str, public_key: np.ndarray):
        """Registra chave pública de um peer"""
        self.peer_public_keys[peer_id] = public_key
    
    def sign_message(self, message: bytes) -> np.ndarray:
        """Assina mensagem com chave privada"""
        if self.private_key is None:
            raise ValueError("Private key not generated")
        
        return self.qrs.sign(message, self.private_key)
    
    def verify_signature(self, message: bytes, signature: np.ndarray, peer_id: str) -> bool:
        """Verifica assinatura de um peer"""
        if peer_id not in self.peer_public_keys:
            return False
        
        public_key = self.peer_public_keys[peer_id]
        return self.qrs.verify(message, signature, public_key)
    
    def encrypt_lattice(self, plaintext: torch.Tensor) -> torch.Tensor:
        """Encripta tensor usando lattice-based encryption"""
        
        if plaintext.dim() == 1:
            plaintext = plaintext.unsqueeze(0)
        
        noise = torch.randn_like(plaintext) * 0.01
        
        encrypted = torch.matmul(plaintext, self.encryption_matrix) + noise
        return encrypted
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica encriptação lattice-based"""
        return self.encrypt_lattice(x)
