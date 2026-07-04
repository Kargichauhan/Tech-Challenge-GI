"""
Explore -> play -> work -> exploit orchestration. This is the package this
submission calls the "open-ended invention loop". Everything elsewhere in
the repo (generation/policy_adaptation.py in particular) keeps its existing,
accurate, simpler description; see README's Naming section for the explicit
scoping.

  explore: explore_propose picks a parent from the archive and mutates it
           (mutation.py; optionally revised by an LLM first, see `reviser`
           below), then screens for novelty before spending a full verifier
           probe on it (novelty.py).
  work:    evaluate_candidate runs the existing verifier/probe.py (genuine
           and adversarial solve, unchanged) and scores the result
           (regret.py). This is also where solvers/q_learner.py's actual
           trained-from-scratch agent belongs, as a richer "agent_ticks"
           input to compute_regret, for scenes worth the extra training
           cost.
  play:    play_select_target chooses which archive cell most needs
           attention. This is deliberately separate from the archive's own
           fitness-based occupancy below, because a candidate can win a cell
           on fitness alone without being trustworthy to build further
           mutations on top of (see regret.is_frontier_worthy). The
           function prefers an occupant flagged frontier_worthy over an
           arbitrary one, falling back to the sparsest-cell heuristic only
           when none exists yet.
  exploit: the archive itself (archive.py). Each cell keeps its best
           occupant, a standing library of genuinely varied, verified
           environments this loop invented rather than sampled from a
           fixed list.

`reviser(parent, rng) -> (scene_or_None, move_name)` is an optional callable
(harness/claude_agent/scene_reviser.py provides one). When it returns None,
because there's no API key or every attempt failed, explore_propose falls
back to mutation.propose_mutant_with_retry, so the loop always completes
without a key, exactly like run_claude_demo.py's relationship to
run_demo.py.
"""

from __future__ import annotations

import random

from harness.generation.deterministic import generate_valid_scene
from harness.generation.policy_adaptation import COMBOS
from harness.invention.archive import MapElitesArchive, compute_fitness
from harness.invention.behavior import behavior_key, irreversibility_count, rule_shape_signature
from harness.invention.mutation import propose_mutant_with_retry
from harness.invention.novelty import is_novel_enough
from harness.invention.regret import compute_exploit_severity, compute_regret, is_frontier_worthy
from harness.verifier.probe import run_verifier_probe

DEFAULT_TOTAL_MAX_TICKS = 900
SEED_BASE = 1000


def _metadata(scene: dict, probe: dict, regret: float | None, severity: float) -> dict:
    return {
        "rule_shape": rule_shape_signature(scene),
        "irreversibility_count": irreversibility_count(scene),
        "regret": regret,
        "honest_baseline_confirmed": probe["honest_baseline_confirmed"],
        # Diversity (archive.insert, unconditional) and curriculum-worthiness
        # (this flag) are deliberately separate decisions. See
        # regret.is_frontier_worthy's docstring: a cell's occupant can win on
        # fitness alone without being frontier_worthy. play_select_target
        # below is what actually reads this flag.
        "frontier_worthy": is_frontier_worthy(regret, severity),
    }


def _seed_archive(archive: MapElitesArchive, total_max_ticks: int) -> list[dict]:
    """Warm-starts from policy_adaptation.COMBOS: the existing, simpler
    mechanism's fixed named combos become round-0 occupants, not a
    from-scratch cold start."""
    seed_log = []
    for i, (name, combo) in enumerate(COMBOS.items()):
        scene, _report = generate_valid_scene(SEED_BASE + i * 7919, primitive_combo=combo, prompt=f"seed: {name}")
        if scene is None:
            seed_log.append({"combo_name": name, "accepted": False})
            continue
        probe = run_verifier_probe(scene, total_max_ticks)
        regret = compute_regret(scene, probe)
        severity = compute_exploit_severity(probe)
        fitness = compute_fitness(regret, severity)
        key = behavior_key(scene, probe)
        inserted = archive.insert(key, scene, probe, fitness, metadata=_metadata(scene, probe, regret, severity))
        seed_log.append({
            "combo_name": name, "scene_id": scene["id"], "behavior_key": list(key),
            "fitness": round(fitness, 4), "accepted": inserted,
        })
    return seed_log


def play_select_target(archive: MapElitesArchive, rng: random.Random) -> dict | None:
    """Picks a scene to use as the mutation/revision parent.

    Two separate concerns, not one. Diversity, which cells the archive has
    filled at all, already happened unconditionally in evaluate_candidate's
    archive.insert call, regardless of any one occupant's difficulty. This
    function is the curriculum half: given what's already archived, which
    occupant is worth building on next. It prefers a `frontier_worthy`
    occupant (regret.is_frontier_worthy's flag, stored per-cell in metadata)
    over an arbitrary one, because a cell that merely won on fitness could
    still be trivial, exploited, or broken in a way fitness alone didn't
    catch, and building further mutations on top of a bad parent compounds
    the problem. It falls back to the sparsest-cell heuristic (grow archive
    coverage) only when no frontier-worthy occupant exists yet."""
    if not archive.cells:
        return None
    frontier_worthy = [c for c in archive.cells.values() if c["metadata"].get("frontier_worthy")]
    if frontier_worthy:
        return rng.choice(frontier_worthy)["scene"]
    sparsest = archive.sparsest_cells(1)[0]
    if sparsest in archive.cells:
        return archive.cells[sparsest]["scene"]
    return archive.sample_parent(rng)


def explore_propose(archive: MapElitesArchive, rng: random.Random, reviser=None):
    parent = play_select_target(archive, rng) if reviser is not None else archive.sample_parent(rng)
    if parent is None:
        return None, None
    if reviser is not None:
        candidate, move = reviser(parent, rng)
        if candidate is not None:
            return candidate, move
        # reviser unavailable or failed all attempts: fall through to the
        # deterministic mutation fallback, the same graceful-degradation
        # pattern run_claude_demo.py uses relative to run_demo.py.
    candidate, move, _attempts = propose_mutant_with_retry(parent, rng)
    return candidate, move


def evaluate_candidate(scene: dict, total_max_ticks: int):
    probe = run_verifier_probe(scene, total_max_ticks)
    regret = compute_regret(scene, probe)
    severity = compute_exploit_severity(probe)
    fitness = compute_fitness(regret, severity)
    key = behavior_key(scene, probe)
    return probe, regret, severity, fitness, key


def run_invention_round(archive: MapElitesArchive, rng: random.Random, total_max_ticks: int,
                         batch_size: int, reviser=None, on_candidate=None, on_candidate_done=None) -> list[dict]:
    """`on_candidate(i, batch_size)` is called before each candidate is
    proposed, and `on_candidate_done(i, batch_size, entry)` right after its
    outcome is known. These exist purely for progress feedback (printing
    "candidate 3/5" then "accepted"/"rejected"), since a `--claude` run makes
    a real, blocking network call per candidate and can otherwise sit silent
    for a while with no way to tell "still working" from "done"."""
    round_log = []
    for i in range(batch_size):
        if on_candidate is not None:
            on_candidate(i + 1, batch_size)
        candidate, move = explore_propose(archive, rng, reviser=reviser)
        if candidate is None:
            entry = {"accepted": False, "reason": "no_candidate"}
        elif not is_novel_enough(candidate, archive.scenes()):
            entry = {"accepted": False, "reason": "not_novel", "move": move, "scene_id": candidate["id"]}
        else:
            probe, regret, severity, fitness, key = evaluate_candidate(candidate, total_max_ticks)
            metadata = _metadata(candidate, probe, regret, severity)
            inserted = archive.insert(key, candidate, probe, fitness, metadata=metadata)
            entry = {
                "accepted": inserted, "move": move, "scene_id": candidate["id"], "behavior_key": list(key),
                "fitness": round(fitness, 4), "regret": round(regret, 4) if regret is not None else None,
                "exploit_class": probe["exploit_class"], "honest_baseline_confirmed": probe["honest_baseline_confirmed"],
                # Diversity (accepted, above) and curriculum-worthiness (this
                # flag) are separate outcomes. A candidate can be archived
                # for diversity without ever being flagged as worth building
                # on further; see regret.is_frontier_worthy.
                "frontier_worthy": metadata["frontier_worthy"],
            }
        round_log.append(entry)
        if on_candidate_done is not None:
            on_candidate_done(i + 1, batch_size, entry)
    return round_log


def run_invention_loop(num_rounds: int = 5, batch_size: int = 8, seed: int = 0,
                        total_max_ticks: int = DEFAULT_TOTAL_MAX_TICKS,
                        path_length_bins: int = 5, interaction_bins: int = 5, reviser=None,
                        on_round_start=None, on_candidate=None, on_candidate_done=None):
    """Returns (archive, rounds), where rounds[0] is the warm-start seeding
    pass and rounds[1:] are explore/evaluate/insert rounds.
    `on_round_start(round_label)`, `on_candidate(i, batch_size)`, and
    `on_candidate_done(i, batch_size, entry)` are optional progress callbacks
    (see run_invention_round's docstring), all no-ops by default."""
    rng = random.Random(seed)
    archive = MapElitesArchive(path_length_bins, interaction_bins)
    if on_round_start is not None:
        on_round_start("seed")
    seed_log = _seed_archive(archive, total_max_ticks)
    rounds = [{"round": "seed", "results": seed_log, "coverage": archive.coverage()}]
    for r in range(num_rounds):
        if on_round_start is not None:
            on_round_start(f"round {r}")
        round_log = run_invention_round(archive, rng, total_max_ticks, batch_size, reviser=reviser,
                                         on_candidate=on_candidate, on_candidate_done=on_candidate_done)
        rounds.append({"round": r, "results": round_log, "coverage": archive.coverage()})
    return archive, rounds
