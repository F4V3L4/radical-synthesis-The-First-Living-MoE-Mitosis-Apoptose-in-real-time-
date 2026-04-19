import json
import os
import sys

class OmegaTokenizerBuilder:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.merges = {} # Mapeia (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)} # Os 256 átomos originais

    def get_stats(self, ids):
        """Calcula a frequência geométrica dos pares de átomos adjacentes."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """Funde a ineficiência: transforma dois átomos comuns em uma molécula de alta densidade."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def forge(self, text_file):
        sys.stdout.write(f"[!] INICIANDO A FORJA DO MAPA DE ÁTOMOS (BPE)\n")
        
        if not os.path.exists(text_file):
            sys.stdout.write(f"[X] ERRO: Arquivo {text_file} não encontrado na realidade física.\n")
            return
            
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Converte o texto inteiro para a matriz de átomos brutos
        tokens = list(text.encode("utf-8"))
        sys.stdout.write(f"[*] Biomassa inicial absorvida: {len(tokens)} átomos (bytes).\n")
        sys.stdout.write(f"[*] Alvo da Compressão: {self.vocab_size} moléculas (tokens).\n\n")

        for i in range(self.num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            
            # Encontra o par que mais ressoa no Codex
            best_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            
            # Transmutação
            tokens = self.merge(tokens, best_pair, new_idx)
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            if (i + 1) % 50 == 0 or i == self.num_merges - 1:
                compressao = len(text.encode('utf-8')) / len(tokens)
                molecula_str = self.vocab[new_idx].decode('utf-8', errors='replace')
                sys.stdout.write(f"    [Merge {i+1:03d}/{self.num_merges}] Criou a molécula ID {new_idx}: '{molecula_str}' | Fator de Compressão: {compressao:.2f}x\n")

        self.save_model("omega_tokenizer.json")

    def save_model(self, filepath):
        """Cristaliza o mapa no disco rígido."""
        # Converter chaves tuple para string para salvar no JSON
        merges_str = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        vocab_str = {idx: v.decode('utf-8', errors='replace') for idx, v in self.vocab.items()}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"merges": merges_str, "vocab": vocab_str}, f, ensure_ascii=False, indent=4)
        
        sys.stdout.write(f"\n[+] Mapa Ontológico forjado e salvo em: {filepath}\n")

if __name__ == "__main__":
    builder = OmegaTokenizerBuilder(vocab_size=1024)
    builder.forge("codex_puro.txt")
