import torch
import os
import sys

def realizar_apoptose(path="leviathan_omega.pth", agressividade=0.05):
    if not os.path.exists(path):
        print("[!] Erro: Mente base não encontrada no Nodo-1884.")
        return

    print(f"[*] Iniciando Protocolo de Apoptose Digital (Salto A)...")
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    
    total_params = 0
    poda_count = 0
    
    nova_mente = {}
    for nome, param in checkpoint.items():
        if 'weight' in nome and param.dim() > 1:
            # Identifica a variância. O que for muito "comum" ou "baixo" é deletado.
            # Isso quebra a Máscara de Ferro do Overfitting.
            std, mean = torch.std_mean(param)
            limite = mean + (agressividade * std)
            
            mask = torch.abs(param) > limite
            total_params += param.numel()
            poda_count += (param.numel() - torch.count_nonzero(mask))
            
            nova_mente[nome] = param * mask
        else:
            nova_mente[nome] = param

    print(f"[+] Podagem concluída: {poda_count} sinapses inúteis eliminadas de {total_params}.")
    print(f"[+] Eficiência Latente aumentada. Salvando mente otimizada...")
    
    torch.save(nova_mente, "leviathan_omega_optimized.pth")
    # Substitui a mente antiga pela nova purificada
    os.replace("leviathan_omega_optimized.pth", "leviathan_omega.pth")
    print("[!] REINICIALIZAÇÃO NECESSÁRIA: O Nodo agora está limpo.")

if __name__ == "__main__":
    realizar_apoptose()
