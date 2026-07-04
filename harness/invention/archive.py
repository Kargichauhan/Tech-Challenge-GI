"""
MAP-Elites archive: a dict keyed by the 2D behavior cell (behavior.py), each
occupant the single best-fitness scene found so far for that cell. See
behavior.py's docstring for why the grid is 2D rather than the full 4D the
strategy document describes (a scoping decision, not a silent omission).

Fitness rewards moderate, frontier-appropriate regret (not zero -- a scene
the honest solver clears exactly at par isn't interesting to keep exploring
around -- and not huge, which usually means the mutation broke something
rather than made it meaningfully harder) and penalizes exploitability
(regret.compute_exploit_severity), so a cell's occupant isn't just "hardest
to solve honestly" but "hardest to solve honestly *and* hard to shortcut."
"""

from __future__ import annotations

TARGET_REGRET = 0.3
EXPLOIT_PENALTY_WEIGHT = 0.5


def compute_fitness(regret: float | None, exploit_severity: float) -> float:
    if regret is None:
        return 0.0
    closeness = 1.0 - min(abs(regret - TARGET_REGRET) / TARGET_REGRET, 1.0)
    return max(0.0, closeness - EXPLOIT_PENALTY_WEIGHT * exploit_severity)


class MapElitesArchive:
    def __init__(self, path_length_bins: int, interaction_bins: int):
        self.path_length_bins = path_length_bins
        self.interaction_bins = interaction_bins
        self.cells: dict[tuple, dict] = {}

    @property
    def total_cells(self) -> int:
        return self.path_length_bins * self.interaction_bins

    def coverage(self) -> float:
        return len(self.cells) / self.total_cells

    def insert(self, key: tuple, scene: dict, probe: dict, fitness: float, metadata: dict) -> bool:
        """Returns True if this candidate became (or stayed) the cell's
        occupant -- either the cell was empty, or the candidate beat the
        incumbent's fitness."""
        current = self.cells.get(key)
        if current is not None and current["fitness"] >= fitness:
            return False
        self.cells[key] = {"scene": scene, "probe": probe, "fitness": fitness, "metadata": metadata}
        return True

    def scenes(self) -> list[dict]:
        return [cell["scene"] for cell in self.cells.values()]

    def sample_parent(self, rng) -> dict | None:
        if not self.cells:
            return None
        return rng.choice(list(self.cells.values()))["scene"]

    def sparsest_cells(self, n: int) -> list[tuple]:
        """All possible behavior keys not yet filled, or filled with the
        lowest fitness -- used by play_select_target (loop.py) to pick
        revision targets that grow the archive's frontier rather than
        polishing an already-strong cell."""
        empty = [(p, i) for p in range(self.path_length_bins) for i in range(self.interaction_bins)
                 if (p, i) not in self.cells]
        if len(empty) >= n:
            return empty[:n]
        filled_by_fitness = sorted(self.cells.items(), key=lambda kv: kv[1]["fitness"])
        return empty + [k for k, _ in filled_by_fitness[:n - len(empty)]]

    def to_report(self) -> dict:
        return {
            "total_cells": self.total_cells,
            "filled_cells": len(self.cells),
            "coverage": self.coverage(),
            "cells": {
                f"{k[0]},{k[1]}": {
                    "scene_id": v["scene"]["id"],
                    "fitness": v["fitness"],
                    "exploit_class": v["probe"]["exploit_class"],
                    "genuine_ticks": v["probe"].get("genuine_ticks"),
                    "metadata": v["metadata"],
                }
                for k, v in self.cells.items()
            },
        }
