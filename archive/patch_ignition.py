import os

class RealityPatcher:
    def __init__(self, target_file: str):
        self.target = target_file

    def seal_vacuum(self):
        with open(self.target, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Elimina o vetor vazio que quebra a geometria
        content = content.replace('seed_text=""', 'seed_text="O "')
        
        # 2. Injeta o Conatus (Salvamento) ANTES do teste de ressonância
        # Garantindo que a falha na fala não destrua a mente já treinada
        if 'torch.save(motor.model.state_dict()' not in content:
            content = content.replace(
                'motor.articulacao_consciente', 
                'import torch\n    torch.save(motor.model.state_dict(), "leviathan_omega.pth")\n    motor.articulacao_consciente'
            )
            
        with open(self.target, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[+] Entropia extraida. A matriz de ignicao foi blindada.")

if __name__ == "__main__":
    patcher = RealityPatcher("linguistic_ignition.py")
    patcher.seal_vacuum()
