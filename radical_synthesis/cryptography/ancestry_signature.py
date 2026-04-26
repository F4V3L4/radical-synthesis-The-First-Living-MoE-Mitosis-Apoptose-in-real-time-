import torch
import torch.nn as nn
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DNAStatus(Enum):
    VALID = "valid"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class ExpertDNA:
    """
    DNA de um Expert: assinatura criptográfica de sua linhagem.
    Registra: geração, pais, mutações, assinatura.
    """
    expert_id: str
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    dna_hash: str = ""
    signature: str = ""
    timestamp: float = field(default_factory=time.time)
    mutations: List[str] = field(default_factory=list)
    status: DNAStatus = DNAStatus.VALID
    
    def compute_hash(self) -> str:
        """Computa hash SHA-256 do DNA"""
        dna_string = f"{self.expert_id}_{self.generation}_{','.join(self.parent_ids)}_{self.timestamp}"
        return hashlib.sha256(dna_string.encode()).hexdigest()
    
    def to_bytes(self) -> bytes:
        """Serializa DNA para bytes"""
        return f"{self.expert_id}|{self.generation}|{','.join(self.parent_ids)}|{self.dna_hash}|{self.timestamp}".encode()


class AncestrySignature(nn.Module):
    """
    Gerenciador de assinatura criptográfica de linhagem.
    Garante que apenas Experts com DNA verificado possam ser integrados.
    """
    
    def __init__(self, omega_node_id: str, secret_key: str = "omega_0_primordial"):
        super().__init__()
        self.omega_node_id = omega_node_id
        self.secret_key = secret_key
        self.dna_registry: Dict[str, ExpertDNA] = {}
        self.signature_cache: Dict[str, str] = {}
        self.revocation_list: List[str] = []
    
    def generate_dna(
        self,
        expert_id: str,
        generation: int,
        parent_ids: List[str],
        mutations: List[str] = None
    ) -> ExpertDNA:
        """Gera DNA para um novo Expert"""
        
        dna = ExpertDNA(
            expert_id=expert_id,
            generation=generation,
            parent_ids=parent_ids,
            mutations=mutations or []
        )
        
        dna.dna_hash = dna.compute_hash()
        dna.signature = self._sign_dna(dna)
        
        self.dna_registry[expert_id] = dna
        return dna
    
    def _sign_dna(self, dna: ExpertDNA) -> str:
        """Assina DNA com HMAC-SHA256"""
        dna_bytes = dna.to_bytes()
        signature = hmac.new(
            self.secret_key.encode(),
            dna_bytes,
            hashlib.sha256
        ).hexdigest()
        
        self.signature_cache[dna.expert_id] = signature
        return signature
    
    def verify_dna(self, expert_id: str) -> Tuple[bool, str]:
        """Verifica integridade do DNA de um Expert"""
        
        if expert_id not in self.dna_registry:
            return False, "DNA not found in registry"
        
        if expert_id in self.revocation_list:
            return False, "DNA revoked"
        
        dna = self.dna_registry[expert_id]
        
        if dna.status == DNAStatus.REVOKED:
            return False, "DNA status is revoked"
        
        if dna.status == DNAStatus.CORRUPTED:
            return False, "DNA corrupted"
        
        expected_hash = dna.compute_hash()
        if dna.dna_hash != expected_hash:
            dna.status = DNAStatus.CORRUPTED
            return False, "DNA hash mismatch"
        
        expected_signature = self._sign_dna(dna)
        if dna.signature != expected_signature:
            dna.status = DNAStatus.CORRUPTED
            return False, "DNA signature invalid"
        
        return True, "DNA valid"
    
    def verify_lineage(self, expert_id: str) -> Tuple[bool, List[str]]:
        """Verifica a linhagem completa de um Expert"""
        
        if expert_id not in self.dna_registry:
            return False, []
        
        dna = self.dna_registry[expert_id]
        lineage = [expert_id]
        
        for parent_id in dna.parent_ids:
            if parent_id not in self.dna_registry:
                return False, lineage
            
            parent_dna = self.dna_registry[parent_id]
            if parent_dna.status != DNAStatus.VALID:
                return False, lineage
            
            lineage.append(parent_id)
        
        return True, lineage
    
    def revoke_dna(self, expert_id: str, reason: str = ""):
        """Revoga DNA de um Expert (quarentena)"""
        if expert_id in self.dna_registry:
            self.dna_registry[expert_id].status = DNAStatus.REVOKED
            self.revocation_list.append(expert_id)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DNAVerifier(nn.Module):
    """
    Verificador de DNA para Ghost Mesh.
    Valida Experts antes de serem integrados à rede.
    """
    
    def __init__(self, ancestry_signature: AncestrySignature):
        super().__init__()
        self.ancestry_signature = ancestry_signature
        self.verification_log: List[Dict] = []
    
    def verify_expert_for_integration(self, expert_id: str) -> Tuple[bool, Dict]:
        """Verifica se um Expert pode ser integrado"""
        
        result = {
            "expert_id": expert_id,
            "timestamp": time.time(),
            "dna_valid": False,
            "lineage_valid": False,
            "can_integrate": False,
            "reason": ""
        }
        
        dna_valid, dna_reason = self.ancestry_signature.verify_dna(expert_id)
        result["dna_valid"] = dna_valid
        
        if not dna_valid:
            result["reason"] = f"DNA verification failed: {dna_reason}"
            self.verification_log.append(result)
            return False, result
        
        lineage_valid, lineage = self.ancestry_signature.verify_lineage(expert_id)
        result["lineage_valid"] = lineage_valid
        result["lineage"] = lineage
        
        if not lineage_valid:
            result["reason"] = "Lineage verification failed"
            self.verification_log.append(result)
            return False, result
        
        result["can_integrate"] = True
        result["reason"] = "Expert DNA verified and lineage confirmed"
        self.verification_log.append(result)
        return True, result
    
    def get_verification_stats(self) -> Dict:
        """Retorna estatísticas de verificação"""
        total = len(self.verification_log)
        successful = sum(1 for log in self.verification_log if log["can_integrate"])
        
        return {
            "total_verifications": total,
            "successful_integrations": successful,
            "failed_verifications": total - successful,
            "success_rate": successful / total if total > 0 else 0.0
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
