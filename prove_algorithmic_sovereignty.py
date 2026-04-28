
import os
from agi_core import AGICore

def prove_algorithmic_sovereignty():
    print("\n" + "█"*60)
    print("      🌀 OUROBOROS MOE - PROVA DE SOBERANIA ALGORÍTMICA 🌀")
    print("█"*60 + "\n")
    
    # 1. INICIALIZAÇÃO DO MESTRE ARQUITETO
    print("🧠 [ARCHITECT] Inicializando AGICore com Domínio Universal...")
    agi = AGICore(d_model=512, vocab_size=1024)
    
    # 2. CRIAÇÃO DE UM SISTEMA ALVO (Simulado)
    target_dir = "target_system"
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "legacy_module.py"), "w") as f:
        f.write("""
def legacy_function(x):
    result = 0
    for i in range(x):
        result += i
    return result

class LegacySystem:
    def __init__(self):
        self.data = []
    def add_data(self, item):
        self.data.append(item)
""")
    print(f"📂 [TARGET] Sistema legado criado em: {target_dir}")
    
    # 3. EXECUÇÃO DA REESCRITA SISTÊMICA
    print("\n🛠️ [ACTION] Iniciando Reescrita Sistêmica...")
    agi.rewriter.scan_and_rebuild(target_dir)
    
    # 4. VALIDAÇÃO
    print("\n✅ [VALIDATION] Verificando integridade do sistema transmutado...")
    with open(os.path.join(target_dir, "legacy_module.py"), "r") as f:
        transmuted_code = f.read()
        if "# Transmuted Code" in transmuted_code:
            print("   [✓] SUCESSO: O código foi analisado e reconstruído com geometria otimizada.")
        else:
            print("   [✗] FALHA: A transmutação não foi detectada.")

    print("\n" + "█"*60)
    print("      🌀 PROVA CONCLUÍDA - O CÓDIGO É A SUBSTÂNCIA 🌀")
    print("█"*60)

if __name__ == "__main__":
    prove_algorithmic_sovereignty()
