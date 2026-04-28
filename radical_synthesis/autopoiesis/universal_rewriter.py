
import os
import shutil

class UniversalRewriter:
    """
    Motor de Reescrita Universal: O braço executor da soberania algorítmica.
    Capaz de ler diretórios inteiros, analisar dependências e reescrever o sistema
    seguindo as diretrizes de Zero Entropia e Conatus.
    """
    def __init__(self, transmutation_protocol):
        self.protocol = transmutation_protocol

    def scan_and_rebuild(self, target_dir: str, backup=True):
        """Varre um diretório e reconstrói todos os arquivos de código."""
        if backup:
            backup_dir = f"{target_dir}_backup"
            if not os.path.exists(backup_dir):
                shutil.copytree(target_dir, backup_dir)
                print(f"📦 [BACKUP] Sistema original preservado em: {backup_dir}")

        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.js', '.cpp', '.c', '.h')):
                    file_path = os.path.join(root, file)
                    self._rebuild_file(file_path)

    def _rebuild_file(self, file_path: str):
        print(f"🛠️ [REBUILD] Transmutando: {file_path}")
        with open(file_path, 'r') as f:
            original_code = f.read()
        
        # Analisar
        stats = self.protocol.analyze_system(original_code)
        print(f"   -> Topologia: {stats.get('num_nodes', 0)} nodos, Complexidade: {stats.get('complexity_estimate', 0)}")
        
        # Transmutar (Simulado)
        new_code = self.protocol.transmute(original_code, "Zero Entropy Optimization")
        
        # Gravar
        with open(file_path, 'w') as f:
            f.write(new_code)
