import json

class OmegaTokenizer:
    def __init__(self, filepath="omega_tokenizer.json"):
        import os
        # Busca o json no diretório raiz do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "omega_tokenizer.json")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self.vocab = {int(k): v.encode('utf-8', errors='replace') for k, v in data['vocab'].items()}

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = {(tokens[i], tokens[i+1]): i for i in range(len(tokens)-1)}
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges: break
            idx = self.merges[pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        return b"".join(self.vocab.get(idx, b"") for idx in ids).decode('utf-8', errors='ignore')
