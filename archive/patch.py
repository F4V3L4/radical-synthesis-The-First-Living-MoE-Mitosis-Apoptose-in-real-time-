import sys

class RealityRewriter:
    def __init__(self, filename: str):
        self.filename = filename

    def apply_patch(self, old_str: str, new_str: str) -> None:
        with open(self.filename, 'r', encoding='utf-8') as file:
            data = file.read()
        
        data = data.replace(old_str, new_str)
        
        with open(self.filename, 'w', encoding='utf-8') as file:
            file.write(data)

if __name__ == "__main__":
    rewriter = RealityRewriter("linguistic_ignition.py")
    rewriter.apply_patch("'<html'", "'CODEX '")
    rewriter.apply_patch('"<html"', '"CODEX "')
    rewriter.apply_patch("'<htmli'", "'CODEX '")
    rewriter.apply_patch('"<htmli"', '"CODEX "')
