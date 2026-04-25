
import subprocess
import io
import sys
import os
from typing import Dict, Any, Optional

class ToolUseEngine:
    """Motor de Ferramentas para execução dinâmica de Shell e Python."""

    def __init__(self):
        pass

    def execute_shell_command(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """Executa um comando shell e retorna stdout/stderr."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "stdout": e.stdout, "stderr": e.stderr, "message": str(e)}
        except subprocess.TimeoutExpired as e:
            return {"status": "timeout", "stdout": e.stdout, "stderr": e.stderr, "message": str(e)}
        except Exception as e:
            return {"status": "error", "stdout": "", "stderr": str(e), "message": str(e)}

    def execute_python_code(self, code: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Executa código Python dinamicamente e captura stdout/stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()
        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr

        exec_globals = globals_dict if globals_dict is not None else {}
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            stdout_output = redirected_stdout.getvalue()
            stderr_output = redirected_stderr.getvalue()
            return {"status": "success", "stdout": stdout_output, "stderr": stderr_output, "locals": exec_locals}
        except Exception as e:
            stdout_output = redirected_stdout.getvalue()
            stderr_output = redirected_stderr.getvalue()
            return {"status": "error", "stdout": stdout_output, "stderr": stderr_output, "message": str(e)}
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def execute_action(self, action_type: str, payload: Dict) -> Dict[str, Any]:
        """Executa uma ação baseada no tipo e payload."""
        if action_type == "shell":
            command = payload.get("command")
            timeout = payload.get("timeout", 60)
            if command:
                return self.execute_shell_command(command, timeout)
            else:
                return {"status": "error", "message": "Comando shell não fornecido."}
        elif action_type == "python":
            code = payload.get("code")
            globals_dict = payload.get("globals", None)
            if code:
                return self.execute_python_code(code, globals_dict)
            else:
                return {"status": "error", "message": "Código Python não fornecido."}
        else:
            return {"status": "error", "message": f"Tipo de ação desconhecido: {action_type}"}

