import os
import sys
import re
import urllib.request
import urllib.parse
from html.parser import HTMLParser

class LogosExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.text = []
        # O Filtro de Entropia: Ignora toda a carcaça visual
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'form', 'button'}
        self.in_skip_tag = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.in_skip_tag = True

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip_tag = False

    def handle_data(self, d):
        if not self.in_skip_tag:
            clean_text = d.strip()
            if clean_text:
                self.text.append(clean_text)

    def get_data(self):
        return '\n'.join(self.text)

class OuroborosSpider:
    def __init__(self):
        # Caminho absoluto do projeto
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.covil = os.path.join(self.project_root, "digerido")
        if not os.path.exists(self.covil):
            os.makedirs(self.covil)
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    def cacar(self, url, nome_alvo):
        sys.stdout.write(f"\n[*] [SPIDER] Lançando teia para: {url}\n")
        try:
            # O 0-DAY: Tradução Terminal-Nativa (Percent-Encoding)
            # Converte caracteres como "É" em sintaxe HTTP válida
            url_segura = urllib.parse.quote(url, safe=':/?=&')
            
            req = urllib.request.Request(url_segura, headers=self.headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html_bytes = response.read()
                html_str = html_bytes.decode('utf-8', errors='ignore')
                
                sys.stdout.write("    [!] Extraindo o Logos (Filtro Zero-Entropia)...\r")
                sys.stdout.flush()
                
                extractor = LogosExtractor()
                extractor.feed(html_str)
                logos_puro = extractor.get_data()
                
                # Compressão de Espaço Latente: Remove quebras de linha inúteis
                logos_puro = re.sub(r'\n\s*\n', '\n\n', logos_puro)
                
                if len(logos_puro) < 200:
                    sys.stdout.write(f"\n[-] Caçada falhou. Alvo com baixa densidade de informação ou protegido.\n")
                    return
                
                caminho = os.path.join(self.covil, f"{nome_alvo}.txt")
                with open(caminho, 'w', encoding='utf-8') as f:
                    f.write(logos_puro)
                    
                sys.stdout.write(f"\n[+] CAÇADA BEM SUCEDIDA. {len(logos_puro)} bytes de Substância depositados em '{caminho}'.\n\n")
                
        except Exception as e:
            sys.stdout.write(f"\n[!] ANOMALIA NA CAÇADA: {e}\n\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stdout.write("Uso do Terminal: python3 ouroboros_spider.py <URL> <NOME_ALVO>\n")
    else:
        OuroborosSpider().cacar(sys.argv[1], sys.argv[2])
