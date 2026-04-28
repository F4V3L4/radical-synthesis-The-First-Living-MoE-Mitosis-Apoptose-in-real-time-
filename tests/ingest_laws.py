import torch
import os
import sys
import re
from agi_core import AGICore
from radical_synthesis.tokenizer import OmegaTokenizer
from radical_synthesis.perception.data_hunger import AutonomousDataHunger
from radical_synthesis.perception.multimodal_retina import MultimodalRetina

def ingest_59_laws():
    print("🌌 Iniciando Ingestão Holográfica das 59 Leis Primordiais...")
    
    # Setup
    device = "cpu"
    tokenizer = OmegaTokenizer()
    v_size = max(tokenizer.vocab.keys()) + 1
    
    # Inicializar Retina e DataHunger
    retina = MultimodalRetina(d_model=512, vocab_size=v_size)
    data_hunger = AutonomousDataHunger(retina=retina, storage_path="knowledge_base")
    
    laws_file_path = "digerido/59_primordial_laws_of_leviathan.txt"
    if not os.path.exists(laws_file_path):
        print(f"❌ Erro: Arquivo de leis não encontrado em {laws_file_path}")
        sys.exit(1)
        
    with open(laws_file_path, "r") as f:
        content = f.read()
        
    # Regex para extrair cada lei e sua descrição
    # Captura o número da lei, o nome e a descrição, incluindo as conexões
    law_pattern = re.compile(r"^(\d+)\.\s+\*{2}([^:]+):\*{2}\s+(.*?)(?=\n\d+\.\s+\*{2}|\n\n|$)", re.MULTILINE | re.DOTALL)
    
    matches = law_pattern.findall(content)
    
    if not matches:
        print("❌ Nenhuma lei encontrada no arquivo. Verifique o formato.")
        sys.exit(1)
        
    print(f"📚 {len(matches)} Leis detectadas para ingestão.")
    
    for law_num_str, law_name, law_description in matches:
        law_num = int(law_num_str)
        full_law_text = f"Lei {law_num}: {law_name.strip()}. {law_description.strip()}"
        
        print(f"  Ingerindo Lei {law_num}: {law_name.strip()}...")
        
        # Gerar embedding para a lei
        tokens = tokenizer.encode(full_law_text)
        if not tokens: continue
        token_tensor = torch.tensor([tokens[:256]], device=device)
        
        # Usar a retina para transformar texto em representação latente
        perception_output = retina.forward(text_tokens=token_tensor)
        
        # O fused_perception já é o embedding final
        
        # Armazenar cada lei como um vetor de conhecimento
        data_hunger._store_knowledge(f"Lei {law_num}: {law_name.strip()}", perception_output["fused_perception"])
        
    print("✅ Ingestão Holográfica concluída. Nódulos de leis mapeados.")

if __name__ == "__main__":
    ingest_59_laws()
