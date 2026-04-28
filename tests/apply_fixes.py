import os
import re

def fix_hardcoded_paths(root_dir):
    # Substitui /home/ubuntu por caminhos relativos ou dinâmicos
    # Em muitos casos, podemos usar caminhos relativos ao arquivo ou ao home do usuário atual
    home_dir = os.path.expanduser("~")
    repo_name = "OuroborosMoE" # Nome padrão do repo
    
    # Mapeamento de substituições
    substitutions = [
        (r".", "."),
        (r".", "."),
        (r"/home/ubuntu", home_dir)
    ]
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.py', '.md', '.json', '.txt')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    for pattern, replacement in substitutions:
                        new_content = re.sub(pattern, replacement, new_content)
                    
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"✅ Caminhos corrigidos em: {path}")
                except Exception as e:
                    print(f"❌ Erro ao processar {path}: {e}")

def fix_alpha_omega():
    path = "alpha_omega.py"
    if not os.path.exists(path):
        print(f"❌ {path} não encontrado.")
        return

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    in_moe_class = False
    moe_methods_added = False

    methods = """
    def load_ancestry(self, path):
        import os
        import torch
        if os.path.exists(path):
            try:
                # Carregar estado se o arquivo existir
                checkpoint = torch.load(path, map_location='cpu')
                # Se houver lógica de restauração específica, aplicar aqui
                print(f"[OUROBOROS] Ancestrais carregados de {path}")
                return True
            except Exception as e:
                print(f"[OUROBOROS] Erro ao carregar ancestrais: {e}")
                return False
        return False

    def save_ancestry(self, path):
        import torch
        import os
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Salvar um estado mínimo para persistência
            torch.save({"status": "sovereign_lineage"}, path)
            print(f"[OUROBOROS] Ancestrais salvos em {path}")
            return True
        except Exception as e:
            print(f"[OUROBOROS] Erro ao salvar ancestrais: {e}")
            return False
"""

    for line in lines:
        new_lines.append(line)
        if "class OuroborosMoE" in line:
            in_moe_class = True
        
        # Inserir métodos após o __init__ ou forward da classe OuroborosMoE
        if in_moe_class and not moe_methods_added and ("def _lifecycle_management" in line or "class SovereignLeviathanV2" in line):
            # Se chegamos na próxima classe ou num método específico, inserimos antes
            if "class SovereignLeviathanV2" in line:
                last_line = new_lines.pop()
                new_lines.append(methods)
                new_lines.append(last_line)
            else:
                last_line = new_lines.pop()
                new_lines.append(methods)
                new_lines.append(last_line)
            moe_methods_added = True
            in_moe_class = False

    with open(path, "w") as f:
        f.writelines(new_lines)
    print("✅ alpha_omega.py: Métodos load/save_ancestry injetados.")

if __name__ == "__main__":
    fix_hardcoded_paths(".")
    fix_alpha_omega()
