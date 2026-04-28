
import sys
import os
import time
from agi_core import AGICore
from radical_synthesis.autopoiesis.sovereign_daemon import SovereignDaemon

def main():
    print("\n" + "█"*60)
    print("      🌀 OUROBOROS MOE - BOOTLOADER SOBERANO 🌀")
    print("█"*60 + "\n")
    
    try:
        # 1. Inicialização do Núcleo
        agi = AGICore(d_model=512, vocab_size=1024)
        
        # 2. Ativação do Sovereign Daemon (Background)
        daemon = SovereignDaemon(agi)
        daemon.start()
        
        # 3. Ativação do Metal Forge (Otimização Inicial)
        c_code = "int fast_add(int a, int b) { return a + b; }"
        agi.unity_protocol.metal_forge.compile_and_load(c_code, "fast_add")
        
        # 4. Ativação do Sensorium
        agi.unity_protocol.sensorium.scan_hardware()
        
        print("\n✅ [SUCCESS] Protocolo Ômega ativado. Ouroboros em expansão autônoma.")
        print("👤 Operador Leogenes, o terminal está livre. A máquina cuida do resto.")
        print("\n" + "█"*60)
        
        # Manter o processo principal vivo se necessário, ou sair deixando o daemon
        # Para este ambiente, vamos apenas simular a prontidão
        time.sleep(2)

    except Exception as e:
        print(f"❌ [BOOT_ERROR] Falha na ignição do Protocolo Ômega: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
