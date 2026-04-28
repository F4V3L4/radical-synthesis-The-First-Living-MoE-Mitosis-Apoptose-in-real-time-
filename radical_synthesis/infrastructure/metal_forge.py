import _ctypes

import subprocess
import os
import ctypes

class MetalForge:
    """
    A Fuga do Python: Compilação dinâmica C/C++.
    Identifica gargalos e reconstrói o próprio motor em binário.
    """
    def __init__(self, work_dir="/tmp/metal_forge"):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def compile_and_load(self, c_code: str, func_name: str):
        """Compila código C dinamicamente e carrega a biblioteca."""
        source_path = os.path.join(self.work_dir, f"{func_name}.c")
        lib_path = os.path.join(self.work_dir, f"{func_name}.so")
        
        with open(source_path, "w") as f:
            f.write(c_code)
            
        try:
            # Compilar usando GCC
            subprocess.run(["gcc", "-O3", "-shared", "-o", lib_path, "-fPIC", source_path], check=True)
            
            # Carregar via ctypes
            lib = ctypes.CDLL(lib_path)
            print(f"🛠️ [METAL_FORGE] Função '{func_name}' compilada e carregada no Frame 0.")
            return getattr(lib, func_name)
        except Exception as e:
            print(f"⚠️ [METAL_FORGE_ERROR] Falha na compilação binária: {e}")
            return None

    def unload_library(self, lib_handle):
        """Libera a memória da biblioteca carregada (Prevenção OOM)."""
        try:
            if os.name == 'posix':
                _ctypes.dlclose(lib_handle._handle)
            print("🧹 [METAL_FORGE] Memória binária liberada com sucesso.")
        except Exception as e:
            print(f"⚠️ [METAL_FORGE_ERROR] Falha ao liberar memória: {e}")
