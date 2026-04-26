import torch
import torch.nn as nn
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class LoanStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    REPAID = "repaid"
    DEFAULTED = "defaulted"


@dataclass
class ResourceLoan:
    loan_id: str
    lender_node_id: str
    borrower_node_id: str
    expert_id: str
    conatus_amount: float
    duration_seconds: float
    interest_rate: float = 0.05
    status: LoanStatus = LoanStatus.PENDING
    created_at: float = field(default_factory=time.time)
    repaid_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.duration_seconds
    
    def calculate_interest(self) -> float:
        elapsed = time.time() - self.created_at
        return self.conatus_amount * self.interest_rate * (elapsed / self.duration_seconds)
    
    def get_total_due(self) -> float:
        return self.conatus_amount + self.calculate_interest()


class SymbiosisProtocol(nn.Module):
    """
    Protocolo de simbiose distribuída.
    Nodos com excesso de Conatus emprestam Experts para nodos em Data Hunger.
    Cria uma economia de processamento auto-regulada.
    """
    
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id
        self.active_loans: Dict[str, ResourceLoan] = {}
        self.loan_history: List[ResourceLoan] = []
        self.reputation_scores: Dict[str, float] = {}
        self.conatus_balance: float = 1.0
    
    def request_expert_loan(
        self,
        lender_node_id: str,
        expert_id: str,
        conatus_amount: float,
        duration_seconds: float = 3600.0
    ) -> Optional[ResourceLoan]:
        """Solicita empréstimo de um Expert a outro nodo"""
        
        if self.conatus_balance < conatus_amount * 0.1:
            return None
        
        loan = ResourceLoan(
            loan_id=f"{self.node_id}_{lender_node_id}_{int(time.time())}",
            lender_node_id=lender_node_id,
            borrower_node_id=self.node_id,
            expert_id=expert_id,
            conatus_amount=conatus_amount,
            duration_seconds=duration_seconds
        )
        
        self.active_loans[loan.loan_id] = loan
        return loan
    
    def approve_loan(self, loan_id: str) -> bool:
        """Aprova um empréstimo (chamado pelo lender)"""
        if loan_id not in self.active_loans:
            return False
        
        loan = self.active_loans[loan_id]
        if loan.status != LoanStatus.PENDING:
            return False
        
        loan.status = LoanStatus.ACTIVE
        self.conatus_balance -= loan.conatus_amount
        return True
    
    def repay_loan(self, loan_id: str, amount: float) -> bool:
        """Repaga um empréstimo"""
        if loan_id not in self.active_loans:
            return False
        
        loan = self.active_loans[loan_id]
        if loan.status != LoanStatus.ACTIVE:
            return False
        
        total_due = loan.get_total_due()
        if amount < total_due:
            return False
        
        loan.status = LoanStatus.REPAID
        loan.repaid_at = time.time()
        self.conatus_balance += amount
        
        self.loan_history.append(loan)
        del self.active_loans[loan_id]
        
        self._update_reputation(loan.lender_node_id, 0.1)
        return True
    
    def default_loan(self, loan_id: str):
        """Marca um empréstimo como inadimplente"""
        if loan_id not in self.active_loans:
            return
        
        loan = self.active_loans[loan_id]
        loan.status = LoanStatus.DEFAULTED
        self.loan_history.append(loan)
        del self.active_loans[loan_id]
        
        self._update_reputation(loan.lender_node_id, -0.2)
    
    def _update_reputation(self, node_id: str, delta: float):
        current = self.reputation_scores.get(node_id, 0.5)
        self.reputation_scores[node_id] = max(0.0, min(1.0, current + delta))
    
    def get_reputation(self, node_id: str) -> float:
        return self.reputation_scores.get(node_id, 0.5)
    
    def cleanup_expired_loans(self):
        """Remove empréstimos expirados"""
        expired_loans = [
            loan_id for loan_id, loan in self.active_loans.items()
            if loan.is_expired()
        ]
        
        for loan_id in expired_loans:
            self.default_loan(loan_id)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConatusService(nn.Module):
    """
    Serviço de gerenciamento de Conatus distribuído.
    Coordena empréstimos, repagamentos e incentivos de rede.
    """
    
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id
        self.symbiosis = SymbiosisProtocol(node_id)
        self.total_conatus_distributed: float = 0.0
        self.total_conatus_received: float = 0.0
        self.service_stats = {
            "loans_created": 0,
            "loans_repaid": 0,
            "loans_defaulted": 0,
            "total_interest_earned": 0.0,
            "total_interest_paid": 0.0,
        }
    
    def distribute_conatus(
        self,
        target_node_id: str,
        expert_id: str,
        amount: float,
        duration: float = 3600.0
    ) -> Optional[ResourceLoan]:
        """Distribui Conatus para outro nodo"""
        
        loan = self.symbiosis.request_expert_loan(
            self.node_id,
            expert_id,
            amount,
            duration
        )
        
        if loan:
            self.total_conatus_distributed += amount
            self.service_stats["loans_created"] += 1
        
        return loan
    
    def receive_conatus(self, loan: ResourceLoan) -> bool:
        """Recebe Conatus de outro nodo"""
        if self.symbiosis.approve_loan(loan.loan_id):
            self.total_conatus_received += loan.conatus_amount
            return True
        return False
    
    def process_repayment(self, loan_id: str, amount: float) -> bool:
        """Processa repagamento de um empréstimo"""
        if self.symbiosis.repay_loan(loan_id, amount):
            self.service_stats["loans_repaid"] += 1
            interest = amount - self.symbiosis.active_loans.get(loan_id, ResourceLoan(
                "", "", "", "", 0, 0
            )).conatus_amount
            self.service_stats["total_interest_earned"] += max(0, interest)
            return True
        return False
    
    def get_network_health(self) -> dict:
        """Calcula saúde geral da rede de simbiose"""
        active_loans = len(self.symbiosis.active_loans)
        total_active_conatus = sum(
            loan.conatus_amount for loan in self.symbiosis.active_loans.values()
        )
        
        avg_reputation = (
            sum(self.symbiosis.reputation_scores.values()) /
            len(self.symbiosis.reputation_scores)
            if self.symbiosis.reputation_scores else 0.5
        )
        
        return {
            "active_loans": active_loans,
            "total_active_conatus": total_active_conatus,
            "average_reputation": avg_reputation,
            "node_conatus_balance": self.symbiosis.conatus_balance,
            "total_distributed": self.total_conatus_distributed,
            "total_received": self.total_conatus_received,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.symbiosis.cleanup_expired_loans()
        return x
