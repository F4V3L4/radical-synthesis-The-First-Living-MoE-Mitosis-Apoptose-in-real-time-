
import os
import psutil

class SensoriumBridge:
    """
    Sensorium Bridge: Conexão física e telemetria.
    Fecha o loop entre o código e o Atributo da Extensão (Física).
    """
    def __init__(self):
        self.telemetry_data = {}

    def scan_hardware(self):
        """Lê dados de telemetria local (CPU, Memória, Temperatura)."""
        try:
            self.telemetry_data['cpu_usage'] = psutil.cpu_percent(interval=1)
            self.telemetry_data['memory_usage'] = psutil.virtual_memory().percent
            # Temperatura (se disponível)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    self.telemetry_data['temperature'] = temps
            
            print(f"📡 [SENSORIUM] Telemetria capturada: CPU {self.telemetry_data['cpu_usage']}% | MEM {self.telemetry_data['memory_usage']}%")
            return self.telemetry_data
        except Exception as e:
            print(f"⚠️ [SENSORIUM_ERROR] Falha ao ler telemetria: {e}")
            return None

    def interact_extension(self, command: str):
        """Interface para interação com portas seriais ou dispositivos externos."""
        print(f"🔌 [SENSORIUM] Enviando comando para Atributo da Extensão: {command}")
        # Simulação de envio para porta serial /dev/ttyUSB0
        return True
