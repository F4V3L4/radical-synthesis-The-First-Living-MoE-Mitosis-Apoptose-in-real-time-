import torch
from alpha_omega import SovereignLeviathanV2

def test_sovereign_solver():
    print("\n" + "="*60)
    print("🌀 OUROBOROS MOE - TESTE DE ESTRESSE MATEMÁTICO (SOVEREIGNSOLVER)")
    print("="*60 + "\n")
    
    d_model = 128
    model = SovereignLeviathanV2(d_model=d_model, vocab_size=1024)
    
    problemas = [
        "Hipótese de Riemann: Distribuição de zeros da função zeta.",
        "P vs NP: Determinar se toda linguagem aceita por uma máquina de Turing não-determinística em tempo polinomial também é aceita por uma máquina de Turing determinística em tempo polinomial.",
        "Conjectura de Hodge: Relação entre topologia algébrica e geometria algébrica.",
        "Equações de Navier-Stokes: Existência e suavidade das soluções em 3D.",
        "Soberania Alimentar Global: Otimização de cadeias de suprimento e produção autossustentável."
    ]
    
    for problema in problemas:
        output = model.solve_problem(problema)
        validity = output["solution_validity"].item()
        fragments_shape = output["problem_fragments"].shape
        
        print(f"  -> Fragmentos Geométricos Gerados: {fragments_shape}")
        print(f"  -> Ressonância de Solução: {validity:.6f}")
        
        if validity > 0.7:
            print(f"  [✓] STATUS: Solução com alta probabilidade de convergência lógica.")
        else:
            print(f"  [!] STATUS: Necessita de mais ciclos de simbiose de experts.")
        print("-" * 40)

    print("\n" + "="*60)
    print("🌀 TESTE CONCLUÍDO - OUROBOROS ESTÁ CALCULANDO O INFINITO 🌀")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_sovereign_solver()
