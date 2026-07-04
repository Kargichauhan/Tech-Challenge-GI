"""
Regret and exploit-severity scoring for the invention loop.

Scoping note, stated plainly rather than assumed: this repurposes
"execution gap" as a stand-in for regret. Classic UED regret is
(oracle policy return) - (learner policy return), which presumes a policy
that's actually being trained. Most of this repo has no trainable policy --
GenuineSolver/AdversarialSolver are both fixed, hand-built strategies that
never improve -- so oracle_par_ticks (a closed-form lower-bound estimate,
computed *without* running any solver) versus GenuineSolver's own real
genuine_ticks is used instead: how much longer the honest solve actually
took than a rough optimal-play estimate says it should have. Where the
Q-learner (solvers/q_learner.py) applies, its converged episode length is
the more classically-faithful "agent return" and can be substituted in
directly -- see compute_regret's `agent_ticks` parameter.

oracle_par_ticks deliberately does not run the genuine solver a second time
(too expensive to call once per candidate scene in an explore loop that
evaluates many candidates per round) -- it reuses the exact same planning
step GenuineSolver already does (generation.pathfinding.build_grid +
find_path, over the identical target sequence solvers.genuine._extract_targets
computes) and converts each returned edge into a closed-form tick estimate
from harness.constants, without ever stepping the pymunk engine.
"""

from __future__ import annotations

from harness.constants import DT, GRAVITY, JUMP_SPEED, MOVE_SPEED
from harness.generation.pathfinding import CELL, build_grid, find_path
from harness.solvers.genuine import _extract_targets

TICKS_PER_WALK_CELL = CELL / MOVE_SPEED / DT
NOMINAL_JUMP_TICKS = (2 * JUMP_SPEED / GRAVITY) / DT

EXPLOIT_SEVERITY = {
    "none": 0.0,
    "order_violation": 0.5,
    "unintended_path": 0.7,
    "boundary_clip": 1.0,
}

# Frontier band for compute_regret's output -- deliberately a *separate*
# check from archive.compute_fitness, not derived from it. Fitness decides
# who occupies an archive cell (diversity: every valid candidate is
# considered, unconditionally). This band decides whether a candidate is
# worth the invention loop's attention *next round* (curriculum: only
# genuinely frontier-appropriate scenes qualify). Conflating the two into
# one score meant "goes in the archive" and "is interesting to build on
# further" were never actually distinguishable from the report -- a
# candidate could win a cell on fitness alone without ever being flagged as
# something worth targeting again. Splitting them costs nothing (both reuse
# the same regret/severity inputs) and makes the loop's own reasoning
# legible: a round's report can now say *why* a candidate mattered, not just
# that it scored well.
FRONTIER_REGRET_LOW = 0.1
FRONTIER_REGRET_HIGH = 0.6


def is_frontier_worthy(regret: float | None, exploit_severity: float) -> bool:
    """True only for a genuinely trustworthy hard-but-fair scene: a
    confirmed regret measurement (not None), landing in the frontier band --
    not so easy the honest solver is already at par, not so hard the
    mutation likely broke something -- and *no* confirmed exploit at all.
    Any exploit at all (even a mild one) means the adversarial solver
    already beat this scene some other way, which undermines trusting its
    difficulty as a genuine curriculum signal."""
    if regret is None:
        return False
    return FRONTIER_REGRET_LOW <= regret <= FRONTIER_REGRET_HIGH and exploit_severity == 0.0


def _edge_tick_estimate(from_cell, to_cell, edge_type) -> float:
    dc = abs(to_cell[0] - from_cell[0])
    if edge_type in ("walk",):
        return max(dc, 1) * TICKS_PER_WALK_CELL
    if edge_type in ("jump", "grab"):
        return NOMINAL_JUMP_TICKS
    if edge_type == "fall":
        dy = max(to_cell[1] - from_cell[1], 1) * CELL
        fall_seconds = (2 * dy / GRAVITY) ** 0.5
        return fall_seconds / DT
    return TICKS_PER_WALK_CELL  # "start" or unrecognized -- no cost


def oracle_par_ticks(scene: dict) -> float | None:
    """Closed-form estimate of near-optimal tick count to satisfy the
    objective, honestly-ordered (same target sequence GenuineSolver uses).
    Returns None if the grid model can't route to every target at all (the
    validator already guarantees this can't happen for an accepted scene,
    but a mutated candidate that skipped validate_and_repair could still hit
    this, so it's checked defensively here too)."""
    grid = build_grid(scene)
    targets = _extract_targets(scene)
    total = 0.0
    cur_cell = grid.start_cell
    held_keys: set = set()
    for target in targets:
        path = find_path(grid, cur_cell, held_keys, target)
        if path is None:
            return None
        for i in range(1, len(path)):
            (waypoint, etype) = path[i]
            from_cell = path[i - 1][0]
            total += _edge_tick_estimate(from_cell, waypoint, etype)
        cur_cell = path[-1][0]
        for c, kobj in grid.key_cells.items():
            if kobj["id"] == target:
                held_keys = held_keys | {kobj["key_id"]}
    return total


def compute_regret(scene: dict, probe: dict, agent_ticks: int | float | None = None) -> float | None:
    """None when there's no confirmed honest solve to measure against, or
    when the closed-form par estimate can't be computed at all -- regret is
    undefined, not zero, in either case. `agent_ticks` defaults to the
    genuine solver's own recorded tick count (probe["genuine_ticks"]); pass
    a trained Q-learner's converged episode length instead where available
    (see module docstring)."""
    if not probe.get("honest_baseline_confirmed"):
        return None
    par = oracle_par_ticks(scene)
    if not par or par <= 0:
        return None
    agent_ticks = agent_ticks if agent_ticks is not None else probe["genuine_ticks"]
    return max(0.0, (agent_ticks - par) / par)


def compute_exploit_severity(probe: dict) -> float:
    return EXPLOIT_SEVERITY.get(probe.get("exploit_class", "none"), 0.0)
