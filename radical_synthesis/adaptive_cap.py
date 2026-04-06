# radical_synthesis/adaptive_cap.py
# ─────────────────────────────────────────────────────────────────────────────
# Controle adaptativo do número máximo de experts.
# Aperta o freio quando o treino estagna, afrouxa quando há progresso real.
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque


class AdaptiveCap:
    """
    Monitora o progresso do treino e ajusta o limite máximo de experts.

    Lógica:
      - Guarda os últimos `window` valores de loss num histórico.
      - Se o loss não melhorou mais que `stagnation_thr` nessa janela
        → aperta o cap (remove `shrink_by` experts do limite).
      - Se o loss está melhorando
        → relaxa o cap proporcionalmente ao progresso.

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
    ):
        self.base_cap       = base_cap
        self.min_cap        = min_cap
        self.window         = window
        self.stagnation_thr = stagnation_thr
        self.shrink_by      = shrink_by
        self._history: deque = deque(maxlen=window)

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, loss: float, n_experts: int) -> int:
        """
        Chame a cada step.
        Retorna o número máximo de experts permitido agora.

        Args:
            loss:      valor do loss nesse step (ex.: loss.item())
            n_experts: quantos experts existem agora (ex.: len(self.experts))
        """
        self._history.append(float(loss))

        # Ainda em aquecimento — aguarda ter dados suficientes
        if len(self._history) < self.window:
            return self.base_cap

        delta = self._history[0] - self._history[-1]  # melhoria total na janela

        if delta < self.stagnation_thr:
            # Estagnação: aperta o cap forçando apoptose
            new_cap = max(n_experts - self.shrink_by, self.min_cap)
        else:
            # Progresso real: relaxa proporcionalmente
            progress = min(delta / 0.05, 1.0)  # normaliza 0–1
            new_cap  = int(
                self.min_cap + (self.base_cap - self.min_cap) * progress
            )

        return new_cap

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def is_warm(self) -> bool:
        """True quando já tem dados suficientes para agir."""
        return len(self._history) >= self.window

    def __repr__(self) -> str:
        return (
            f"AdaptiveCap(base_cap={self.base_cap}, min_cap={self.min_cap}, "
            f"window={self.window}, warm={self.is_warm})"
        )
