
import torch
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agi_core import AGICore

def test_simple():
    print("🧪 TESTE SIMPLIFICADO DE PERSISTÊNCIA E HOMEOSTASE")
    device = "cpu"
    agi = AGICore(vocab_size=100, d_model=128, num_experts=2, device=device)
    
    # Homeostase
    for e in agi.core.moe.experts: e.conatus.fill_(0.2)
    status = agi.check_homeostasis()
    print(f"Impulso: {status['impulse']}")
    
    # Evolução Manual
    print("Simulando evolução...")
    for e in agi.core.moe.experts: e.conatus.fill_(2.5)
    # Gatilhar ciclo de vida manualmente via forward pequeno
    from alpha_omega import OuroborosMoE
    agi.core.moe._lifecycle_management(set([0, 1]))
    
    num_before = len(agi.core.moe.experts)
    print(f"Experts antes de salvar: {num_before}")
    agi.save_state()
    
    print("Reiniciando...")
    new_agi = AGICore(vocab_size=100, d_model=128, num_experts=2, device=device)
    num_after = len(new_agi.core.moe.experts)
    print(f"Experts carregados: {num_after}")
    
    if num_before == num_after:
        print("✅ SUCESSO!")
    else:
        print("❌ FALHA!")

if __name__ == "__main__":
    test_simple()
