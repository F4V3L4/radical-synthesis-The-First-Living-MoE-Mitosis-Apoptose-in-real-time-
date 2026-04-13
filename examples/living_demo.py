import torch
from radical_synthesis.alpha_omega import SovereignLeviathanV2, EpistemologicalSentinel

class BareMetalInjector:
    def __init__(self, model: SovereignLeviathanV2, sentinel: EpistemologicalSentinel):
        self.model = model
        self.sentinel = sentinel

    def inject_entropy(self, filepath: str, chunk_size: int = 1024):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        with open(filepath, 'rb') as f:
            state = None
            while chunk := f.read(chunk_size):
                if not self.sentinel.validate_geometric_truth(chunk):
                    continue
                
                tensor_chunk = torch.tensor(list(chunk), dtype=torch.long)
                x = tensor_chunk[:-1].unsqueeze(0)
                y = tensor_chunk[1:].unsqueeze(0)
                
                optimizer.zero_grad()
                logits, state, entropy_loss, _ = self.model(x, state)
                
                state = state.detach()
                
                B, T, C = logits.shape
                loss = criterion(logits.view(-1, C), y.view(-1)) + entropy_loss
                
                loss.backward()
                optimizer.step()
