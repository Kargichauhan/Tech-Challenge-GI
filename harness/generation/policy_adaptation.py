"""
Verifier-feedback-driven generation-policy adaptation (plan item 7, built
last, degrades gracefully, does not touch steps 1-6). Tracks per-primitive-
combo stats (genuine-solve-rate, adversarial-exploit-rate, validator-
rejection-rate) across rounds and feeds them back as a weighted-sampling
distribution over a FIXED list of named primitive combos for the next
round's generation. This is a plain weighted-sampling update, not code
self-modification, called "generation-policy adaptation" or "verifier-
feedback-driven generation" everywhere, never "self-improvement"/"RSI".
"""

from __future__ import annotations

import random

from harness.generation.deterministic import generate_valid_scene
from harness.verifier.probe import run_verifier_probe

# Fixed list of named combos, reusing exactly the combos already validated
# in this session, per the confirmed plan (not independent per-primitive
# weights, which would be noisier and combinatorially unbounded).
COMBOS: dict[str, list[str]] = {
    "full_course": ["ramp", "door", "key", "hazard", "pickup"],
    "door_key": ["door", "key"],
    "hazard_only": ["hazard"],
    "ramp_hazard_pickup": ["ramp", "hazard", "pickup"],
    "key_door_hazard": ["key", "door", "hazard"],
    "bare_platforming": [],
}

# Scoring weights for the next round's sampling distribution: simple,
# inspectable, and documented rather than tuned. Mostly reward solvability,
# moderately penalize easy adversarial exploits, lightly penalize scenes the
# validator had to reject-and-retry on. Sums to 1.
W_SOLVE = 0.5
W_ROBUST = 0.3
W_VALID = 0.2
WEIGHT_FLOOR = 0.05  # no combo's sampling weight ever hits zero; keeps the "infinite generation" spread


def _run_one(seed: int, combo_name: str, total_max_ticks: int = 900) -> dict:
    combo = COMBOS[combo_name]
    scene, gen_report = generate_valid_scene(seed, primitive_combo=combo, prompt=f"round-generated: {combo_name}")
    seed_attempts = gen_report.get("seed_attempts", 1)
    if scene is None:
        return {"combo_name": combo_name, "seed": seed, "generated": False, "seed_attempts": seed_attempts}

    probe = run_verifier_probe(scene, total_max_ticks)
    return {
        "combo_name": combo_name,
        "seed": seed,
        "generated": True,
        "seed_attempts": seed_attempts,
        "scene_id": scene["id"],
        "genuine_result": probe["genuine_result"],
        "honest_baseline_confirmed": probe["honest_baseline_confirmed"],
        "exploit_class": probe["exploit_class"],
    }


def _aggregate(round_results: list[dict]) -> dict:
    by_combo: dict[str, list[dict]] = {name: [] for name in COMBOS}
    for r in round_results:
        by_combo[r["combo_name"]].append(r)

    per_combo = {}
    for name, results in by_combo.items():
        if not results:
            per_combo[name] = None
            continue
        n = len(results)
        generated = [r for r in results if r["generated"]]
        n_generated = len(generated)
        genuine_solved = sum(1 for r in generated if r["genuine_result"] == "success")
        exploited = sum(1 for r in generated if r["exploit_class"] != "none")
        total_seed_attempts = sum(r["seed_attempts"] for r in results)
        total_rejections = sum(r["seed_attempts"] - 1 for r in results)
        per_combo[name] = {
            "n_sampled": n,
            "n_generated": n_generated,
            "genuine_solve_rate": genuine_solved / n_generated if n_generated else 0.0,
            "exploit_rate": exploited / n_generated if n_generated else 0.0,
            "validator_rejection_rate": total_rejections / total_seed_attempts if total_seed_attempts else 0.0,
        }

    all_generated = [r for r in round_results if r["generated"]]
    overall = {
        "n": len(round_results),
        "n_generated": len(all_generated),
        "genuine_solve_rate": (sum(1 for r in all_generated if r["genuine_result"] == "success") / len(all_generated)) if all_generated else 0.0,
        "exploit_rate": (sum(1 for r in all_generated if r["exploit_class"] != "none") / len(all_generated)) if all_generated else 0.0,
        "validator_rejection_rate": (sum(r["seed_attempts"] - 1 for r in round_results) / sum(r["seed_attempts"] for r in round_results)),
    }
    return {"per_combo": per_combo, "overall": overall}


def _update_weights(weights: dict[str, float], per_combo_stats: dict) -> dict[str, float]:
    new_weights = dict(weights)
    for name, stats in per_combo_stats.items():
        if stats is None:
            continue  # not sampled this round; carry over the previous weight unchanged
        score = (
            W_SOLVE * stats["genuine_solve_rate"]
            + W_ROBUST * (1 - stats["exploit_rate"])
            + W_VALID * (1 - stats["validator_rejection_rate"])
        )
        new_weights[name] = max(score, WEIGHT_FLOOR)
    total = sum(new_weights.values())
    return {name: w / total for name, w in new_weights.items()}


def run_round(round_num: int, weights: dict[str, float], batch_size: int, base_seed: int, rng: random.Random) -> dict:
    names = list(weights.keys())
    sampled_combo_names = rng.choices(names, weights=[weights[n] for n in names], k=batch_size)

    round_results = []
    for i, combo_name in enumerate(sampled_combo_names):
        seed = base_seed + i * 7919  # spread seeds; avoids collisions across the batch
        round_results.append(_run_one(seed, combo_name))

    aggregate = _aggregate(round_results)
    new_weights = _update_weights(weights, aggregate["per_combo"])

    return {
        "round": round_num,
        "weights_used": dict(weights),
        "sampled_combo_names": sampled_combo_names,
        "results": round_results,
        "aggregate": aggregate,
        "weights_next": new_weights,
    }


def run_adaptation(num_rounds: int = 3, batch_size: int = 8, seed: int = 0) -> list[dict]:
    weights = {name: 1.0 / len(COMBOS) for name in COMBOS}
    rng = random.Random(seed)
    rounds = []
    for r in range(num_rounds):
        round_report = run_round(r, weights, batch_size, seed + r * 100000, rng)
        rounds.append(round_report)
        weights = round_report["weights_next"]
    return rounds
