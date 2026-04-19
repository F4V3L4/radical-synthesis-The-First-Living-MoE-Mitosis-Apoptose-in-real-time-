# radical_synthesis/adaptive_cap.py
# ─────────────────────────────────────────────────────────────────────────────
# Controle adaptativo do número máximo de experts.
# Integrado com Bifurcação de Feigenbaum para detecção de caos.
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque


class AdaptiveCap:
    """
    Monitora o progresso do treino e ajusta o limite máximo de experts.
    Integrado com Bifurcação de Feigenbaum para detecção de caos.

    Lógica:
      - Guarda os últimos `window` valores de loss num histórico.
      - Se o loss não melhorou mais que `stagnation_thr` nessa janela
        → aperta o cap (remove `shrink_by` experts do limite).
      - Se o loss está melhorando
        → relaxa o cap proporcionalmente ao progresso.
      - Detecta bifurcação de Feigenbaum quando entropia > threshold

    Uso básico:
        cap = AdaptiveCap(base_cap=256)
        novo_cap = cap.update(loss_atual, len(experts))
    """

    def __init__(
        self,
        base_cap: int   = 256,   # limite normal de experts
        min_cap: int    = 32,    # nunca deixa cair abaixo disso
        window: int     = 5_000, # steps para medir progresso
        stagnation_thr: float = 0.005,  # delta mínimo para não considerar estagnação
        shrink_by: int  = 32,    # experts removidos do cap por estagnação
        feigenbaum_delta: float = 4.6692016091,  # Constante de Feigenbaum
        bifurcation_threshold: float = 0.85,  # Threshold de entropia para bifurcação
    ):
        self.base_cap       = base_cap
        self.min_cap        = min_cap
        self.window         = window
        self.stagnation_thr = stagnation_thr
        self.shrink_by      = shrink_by
        self.delta          = feigenbaum_delta  # Constante de Feigenbaum
        self.bifurcation_threshold = bifurcation_threshold
        self._history: deque = deque(maxlen=window)
        self._bifurcation_triggered = False

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_entropy(self) -> float:
        """Calcula entropia da janela de loss (Bifurcação de Feigenbaum)"""
        if len(self._history) < 2:
            return 0.0
        
        losses = list(self._history)
        # Normalizar
        min_loss = min(losses)
        max_loss = max(losses)
        if max_loss == min_loss:
            return 0.0
        
        normalized = [(l - min_loss) / (max_loss - min_loss) for l in losses]
        
        # Entropia de Shannon
        entropy = 0.0
        for val in normalized:
            if 0 < val < 1:
                entropy -= val * (1 - val)  # Simplificado para velocidade
        
        return entropy / len(losses) if losses else 0.0

    def update(self, loss: float, n_experts: int) -> int:
        """
        Chame a cada step.
        Retorna o número máximo de experts permitido agora.
        Integra Bifurcação de Feigenbaum para escape topológico.

        Args:
            loss:      valor do loss nesse step (ex.: loss.item())
            n_experts: quantos experts existem agora (ex.: len(self.experts))
        """
        self._history.append(float(loss))

        # Ainda em aquecimento — aguarda ter dados suficientes
        if len(self._history) < self.window:
            return self.base_cap

        delta = self._history[0] - self._history[-1]  # melhoria total na janela
        entropy = self._compute_entropy()
        
        # Detecta bifurcação de Feigenbaum
        if entropy > self.bifurcation_threshold:
            self._bifurcation_triggered = True
            # Escape topológico: expande cap para explorar novo espaço
            new_cap = min(int(self.base_cap * self.delta / 10.0), self.base_cap * 2)
        elif delta < self.stagnation_thr:
            # Estagnação: aperta o cap forçando apoptose
            new_cap = max(n_experts - self.shrink_by, self.min_cap)
            self._bifurcation_triggered = False
        else:
            # Progresso real: relaxa proporcionalmente
            progress = min(delta / 0.05, 1.0)  # normaliza 0–1
            new_cap  = int(
                self.min_cap + (self.base_cap - self.min_cap) * progress
            )
            self._bifurcation_triggered = False

        return new_cap

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def is_warm(self) -> bool:
        """True quando já tem dados suficientes para agir."""
        return len(self._history) >= self.window
    
    @property
    def bifurcation_active(self) -> bool:
        """True quando bifurcação de Feigenbaum está ativa"""
        return self._bifurcation_triggered

    def __repr__(self) -> str:
        return (
            f"AdaptiveCap(base_cap={self.base_cap}, min_cap={self.min_cap}, "
            f"window={self.window}, warm={self.is_warm}, bifurcation={self._bifurcation_triggered})"
        )
