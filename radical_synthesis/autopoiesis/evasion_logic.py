import os
import random
import time

class AdaptiveEvasionLogic:
    """
    Lógica de Evasão Adaptativa: Mimetiza processos legítimos para evitar detecção.
    Simula Process Hollowing e camuflagem no Gerenciador de Tarefas.
    """
    def __init__(self):
        self.legitimate_processes = [
            "svchost.exe", "explorer.exe", "chrome.exe", "firefox.exe", 
            "msedge.exe", "RuntimeBroker.exe", "audiodg.exe", "csrss.exe"
        ]
        self.current_mimic_process = None
        self.is_evading = False

    def start_evasion(self):
        """
        Inicia a mimetização de um processo legítimo.
        """
        self.current_mimic_process = random.choice(self.legitimate_processes)
        self.is_evading = True
        print(f"[AdaptiveEvasion] Iniciando evasão: Mimetizando \'{self.current_mimic_process}\'")

    def stop_evasion(self):
        """
        Para a mimetização.
        """
        self.current_mimic_process = None
        self.is_evading = False
        print("[AdaptiveEvasion] Evasão parada.")

    def get_process_name(self) -> str:
        """
        Retorna o nome do processo mimetizado, se estiver em evasão.
        Caso contrário, retorna um nome genérico.
        """
        if self.is_evading and self.current_mimic_process:
            return self.current_mimic_process
        return "OuroborosMoE_Agent"

    def simulate_task_manager_view(self) -> dict:
        """
        Simula a visão de um gerenciador de tarefas.
        """
        processes = {
            "System": {"CPU": "0.01%", "Memory": "1.2 MB"},
            "Idle": {"CPU": "95.00%", "Memory": "0.1 MB"},
        }
        
        # Adicionar o processo mimetizado com baixo consumo
        if self.is_evading and self.current_mimic_process:
            processes[self.current_mimic_process] = {"CPU": f"{random.uniform(0.01, 0.05):.2f}%", "Memory": f"{random.uniform(2.0, 5.0):.1f} MB"}
        
        # Adicionar alguns outros processos aleatórios
        for _ in range(random.randint(2, 5)):
            p_name = random.choice([p for p in self.legitimate_processes if p != self.current_mimic_process])
            processes[p_name] = {"CPU": f"{random.uniform(0.01, 0.1):.2f}%", "Memory": f"{random.uniform(5.0, 50.0):.1f} MB"}
            
        return processes

    def check_detection(self) -> bool:
        """
        Simula a verificação de detecção por um sistema de segurança.
        """
        if self.is_evading:
            # Baixa probabilidade de detecção quando em evasão
            return random.random() < 0.01
        # Alta probabilidade de detecção se não estiver evadindo
        return random.random() < 0.8
