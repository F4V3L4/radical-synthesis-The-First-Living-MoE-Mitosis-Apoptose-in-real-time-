# radical_synthesis/genealogy.py
# ─────────────────────────────────────────────────────────────────────────────
# Rastreamento completo da árvore genealógica de experts.
# Registra quem nasceu de quem, quando morreu, vitality ao longo do tempo.
# ─────────────────────────────────────────────────────────────────────────────

import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Registro de um único expert
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpertRecord:
    """
    Histórico completo de um expert desde o nascimento até a morte (se morreu).
    """
    id:         int
    parent_id:  Optional[int]  # None = expert original (geração 0)
    born_step:  int
    generation: int            # 0 = original, 1 = filho, 2 = neto, ...

    died_step:      Optional[int] = None
    children:       list          = field(default_factory=list)  # IDs dos clones
    total_vitality: float         = 0.0
    steps_alive:    int           = 0

    # ── Propriedades calculadas ──────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        """True se o expert ainda não sofreu apoptose."""
        return self.died_step is None

    @property
    def lifespan(self) -> Optional[int]:
        """Quantos steps o expert viveu (None se ainda vivo)."""
        if self.died_step is None:
            return None
        return self.died_step - self.born_step

    @property
    def avg_vitality(self) -> float:
        """Vitality média ao longo da vida do expert."""
        if self.steps_alive == 0:
            return 0.0
        return self.total_vitality / self.steps_alive

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "parent_id":     self.parent_id,
            "born_step":     self.born_step,
            "died_step":     self.died_step,
            "generation":    self.generation,
            "children":      self.children,
            "lifespan":      self.lifespan,
            "avg_vitality":  round(self.avg_vitality, 5),
            "steps_alive":   self.steps_alive,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Árvore genealógica completa
# ─────────────────────────────────────────────────────────────────────────────

class GenealogyTree:
    """
    Rastreia toda a árvore genealógica dos experts durante o treino.

    Uso mínimo:
        tree = GenealogyTree()

        # No __init__ (experts iniciais):
        for expert in self.experts:
            tree.register_birth(expert.id, parent_id=None, step=0)

        # Na mitose:
        tree.register_birth(clone.id, parent_id=parent.id, step=current_step)

        # Na apoptose:
        tree.register_death(expert.id, step=current_step)

        # A cada step (opcional mas recomendado para análise):
        for expert in self.experts:
            tree.update_vitality(expert.id, expert.vitality)

        # Ao final do treino:
        tree.save("genealogy_final.json")
        print(tree.summary())
    """

    def __init__(self):
        self._records: dict[int, ExpertRecord] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Eventos principais
    # ─────────────────────────────────────────────────────────────────────────

    def register_birth(
        self,
        expert_id: int,
        parent_id: Optional[int],
        step: int,
    ) -> None:
        """
        Registra o nascimento de um expert.

        Args:
            expert_id: ID único do expert recém-criado.
            parent_id: ID do expert-pai (None para experts da geração 0).
            step:      Step atual do treino.
        """
        generation = 0

        if parent_id is not None and parent_id in self._records:
            parent = self._records[parent_id]
            parent.children.append(expert_id)
            generation = parent.generation + 1

        self._records[expert_id] = ExpertRecord(
            id=expert_id,
            parent_id=parent_id,
            born_step=step,
            generation=generation,
        )

    def register_death(self, expert_id: int, step: int) -> None:
        """
        Registra a morte de um expert por apoptose.

        Args:
            expert_id: ID do expert removido.
            step:      Step atual do treino.
        """
        if expert_id in self._records:
            self._records[expert_id].died_step = step

    def update_vitality(self, expert_id: int, vitality: float) -> None:
        """
        Atualiza o histórico de vitality de um expert.
        Chame a cada step para manter avg_vitality preciso.

        Args:
            expert_id: ID do expert.
            vitality:  Valor atual de vitality (ex.: expert.vitality).
        """
        if expert_id in self._records:
            r = self._records[expert_id]
            r.total_vitality += float(vitality)
            r.steps_alive    += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Análise evolutiva
    # ─────────────────────────────────────────────────────────────────────────

    def darwin_fitness(self, expert_id: int) -> int:
        """
        Calcula o fitness darwiniano de um expert:
        número total de descendentes vivos (filhos + netos + ...).

        Um expert com fitness 0 não gerou linhagens sobreviventes.
        Um expert com fitness alto dominou o pool evolucionário.
        """
        def _count(eid: int) -> int:
            r = self._records.get(eid)
            if r is None:
                return 0
            alive = 1 if r.is_alive else 0
            return alive + sum(_count(c) for c in r.children)

        # Subtrai o próprio expert da contagem
        root_alive = 1 if self._records.get(expert_id, ExpertRecord(0, None, 0, 0)).is_alive else 0
        return _count(expert_id) - root_alive

    def top_lineages(self, top_n: int = 5) -> list[dict]:
        """
        Retorna os N experts fundadores (geração 0) com mais descendentes vivos.
        Útil para identificar quais linhagens dominaram o treino.

        Returns:
            Lista de dicts com: id, fitness, avg_vitality, n_children.
        """
        founders = [
            r for r in self._records.values()
            if r.parent_id is None
        ]
        ranked = sorted(
            founders,
            key=lambda r: self.darwin_fitness(r.id),
            reverse=True,
        )
        return [
            {
                "id":           r.id,
                "fitness":      self.darwin_fitness(r.id),
                "avg_vitality": round(r.avg_vitality, 4),
                "n_children":   len(r.children),
                "born_step":    r.born_step,
            }
            for r in ranked[:top_n]
        ]

    def generation_stats(self) -> dict:
        """
        Estatísticas por geração: quantos nasceram, morreram, e a vitality média.

        Returns:
            Dict geração → {'born': N, 'alive': N, 'dead': N, 'avg_vitality': f}
        """
        stats: dict[int, dict] = {}
        for r in self._records.values():
            g = r.generation
            if g not in stats:
                stats[g] = {"born": 0, "alive": 0, "dead": 0, "total_vitality": 0.0}
            stats[g]["born"] += 1
            if r.is_alive:
                stats[g]["alive"] += 1
            else:
                stats[g]["dead"] += 1
            stats[g]["total_vitality"] += r.avg_vitality

        # Calcula médias
        for g, s in stats.items():
            s["avg_vitality"] = round(s["total_vitality"] / max(s["born"], 1), 4)
            del s["total_vitality"]

        return dict(sorted(stats.items()))

    # ─────────────────────────────────────────────────────────────────────────
    # Persistência e resumo
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Salva a árvore genealógica completa em JSON.
        O arquivo pode ser carregado depois para análise e visualização.

        Args:
            path: caminho do arquivo de saída (ex.: 'genealogy_final.json')
        """
        data = {
            "summary": self.summary(),
            "records": {
                str(eid): r.to_dict()
                for eid, r in self._records.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[GenealogyTree] Árvore salva em '{path}' "
              f"({len(self._records)} experts registrados)")

    def summary(self) -> dict:
        """Resumo estatístico da árvore."""
        total   = len(self._records)
        alive   = sum(1 for r in self._records.values() if r.is_alive)
        dead    = total - alive
        max_gen = max((r.generation for r in self._records.values()), default=0)
        return {
            "total_experts_ever":  total,
            "currently_alive":     alive,
            "total_deaths":        dead,
            "max_generation":      max_gen,
            "generation_stats":    self.generation_stats(),
        }

    def print_summary(self) -> None:
        """Imprime o resumo no terminal de forma legível."""
        s = self.summary()
        print("\n" + "=" * 55)
        print("  GENEALOGIA DOS EXPERTS")
        print("=" * 55)
        print(f"  Total de experts criados : {s['total_experts_ever']}")
        print(f"  Vivos agora              : {s['currently_alive']}")
        print(f"  Mortos (apoptose)        : {s['total_deaths']}")
        print(f"  Geração mais profunda    : {s['max_generation']}")
        print()
        print("  Linhagens dominantes (top 5):")
        for row in self.top_lineages(top_n=5):
            print(f"    Expert #{row['id']:>4d} | "
                  f"fitness={row['fitness']:>4d} | "
                  f"vitality média={row['avg_vitality']:.3f} | "
                  f"filhos diretos={row['n_children']}")
        print("=" * 55 + "\n")

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        alive = sum(1 for r in self._records.values() if r.is_alive)
        return f"GenealogyTree(total={len(self._records)}, alive={alive})"
