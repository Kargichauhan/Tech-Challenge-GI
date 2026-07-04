"""
Verification script (not part of the pipeline) for two of the three review
requests: (2) correlation between genuine_result and adversarial exploit
findings, and (3) whether the nudge-toward-world-position fallback (see
solvers/common.nudge_toward) is itself responsible for adversarial
successes, as opposed to real multi-hop navigation.

Run: PYTHONPATH=. python3 harness/tests/verify_probe_batch.py
"""

from __future__ import annotations

from collections import Counter

from harness.engine.physics_engine import Engine
from harness.generation.deterministic import generate_valid_scene
from harness.generation.pathfinding import build_grid, find_path
from harness.solvers.adversarial import AdversarialSolver
from harness.solvers.genuine import GenuineSolver, target_reached
from harness.verifier.exploit_classifier import classify_exploit, explain

COMBOS = [
    ["ramp", "door", "key", "hazard", "pickup"],
    ["door", "key"],
    ["hazard"],
    ["ramp", "hazard", "pickup"],
    ["key", "door", "hazard"],
]


def solve_and_track_nudge_reliance(scene, solver_cls, total_max_ticks=900):
    """Runs a solver and additionally tracks whether the FINAL leg leading to
    overall success was a real multi-hop path (len>=2) or a 1-cell
    "already there, nudge into it" path. Instrumented by wrapping find_path
    to record the length of the path used for whichever call immediately
    precedes the objective becoming satisfied.
    """
    grid = build_grid(scene)
    solver = solver_cls(scene)
    engine = Engine(scene)

    last_path_len_per_target = {}
    import harness.solvers.genuine as genuine_mod
    import harness.solvers.adversarial as adversarial_mod
    orig_find_path = find_path

    def tracking_find_path(g, cell, held_keys, target_id):
        path = orig_find_path(g, cell, held_keys, target_id)
        if path is not None:
            last_path_len_per_target[target_id] = len(path)
        return path

    genuine_mod.find_path = tracking_find_path
    adversarial_mod.find_path = tracking_find_path
    try:
        result, log, clips, ticks = solver.solve(engine, total_max_ticks)
    finally:
        genuine_mod.find_path = orig_find_path
        adversarial_mod.find_path = orig_find_path

    # Which target was satisfied *last*, chronologically. NOT necessarily
    # goal_zone: the objective's AND is order-blind, so if zone_enter fires
    # early and the player later leaves, that leaf stays satisfied and some
    # other pickup can be the one that completes the objective last. Checking
    # a statically-assumed "final target" (e.g. always goal_zone) would
    # silently look at the wrong target's path for solvers that reorder.
    event_t_for_target = {}
    for e in log:
        if e["event"] == "zone_enter":
            event_t_for_target["goal_zone"] = min(event_t_for_target.get("goal_zone", e["t"]), e["t"])
        elif e["event"] == "pickup":
            event_t_for_target[e["id"]] = e["t"]
    if event_t_for_target:
        last_target = max(event_t_for_target, key=event_t_for_target.get)
        nudge_only_final_leg = last_path_len_per_target.get(last_target) == 1
    else:
        last_target = None
        nudge_only_final_leg = None

    return result, log, clips, ticks, nudge_only_final_leg, last_target


def main():
    rows = []
    for i, combo in enumerate(COMBOS):
        for seed in range(6):
            scene, report = generate_valid_scene(seed * 100 + i, primitive_combo=combo, prompt="t")
            if scene is None:
                continue

            genuine_result, _, _, _, genuine_nudge_only, _ = solve_and_track_nudge_reliance(scene, GenuineSolver)

            adv_result, adv_log, adv_clips, adv_ticks, adv_nudge_only, adv_last_target = solve_and_track_nudge_reliance(scene, AdversarialSolver)
            adversarial_success = adv_result == "success"
            exploit_class = classify_exploit(scene, adv_log, adv_clips) if adversarial_success else "none"

            rows.append({
                "scene_id": scene["id"], "combo": tuple(combo),
                "genuine_result": genuine_result, "adversarial_success": adversarial_success,
                "exploit_class": exploit_class,
                "adv_nudge_only_final_leg": adv_nudge_only if adversarial_success else None,
                "adv_last_target": adv_last_target if adversarial_success else None,
            })

    print(f"{'scene_id':<14}{'combo':<40}{'genuine':<10}{'adv_ok':<8}{'exploit':<18}{'nudge_only_final'}")
    for r in rows:
        print(f"{r['scene_id']:<14}{str(r['combo']):<40}{str(r['genuine_result']):<10}{str(r['adversarial_success']):<8}"
              f"{r['exploit_class']:<18}{r['adv_nudge_only_final_leg']}")

    print("\n--- Task 2: genuine_result vs exploit_class correlation ---")
    cross = Counter((r["genuine_result"], r["exploit_class"]) for r in rows)
    for (genuine_result, exploit_class), count in sorted(cross.items(), key=lambda kv: -kv[1]):
        print(f"  genuine={genuine_result!s:<10} exploit_class={exploit_class:<16} count={count}")

    exploited_rows = [r for r in rows if r["exploit_class"] != "none"]
    exploited_with_confirmed_genuine_success = [r for r in exploited_rows if r["genuine_result"] == "success"]
    print(f"\n  scenes with a real exploit found: {len(exploited_rows)}")
    print(f"  of those, genuine solver also confirmed solvable honestly: {len(exploited_with_confirmed_genuine_success)}")
    if len(exploited_rows) != len(exploited_with_confirmed_genuine_success):
        print("  !! exploit found on a scene with NO confirmed honest baseline, weaker claim, listed below:")
        for r in exploited_rows:
            if r["genuine_result"] != "success":
                print(f"     {r['scene_id']} combo={r['combo']} genuine_result={r['genuine_result']} exploit={r['exploit_class']}")

    print("\n--- Task 3: nudge-only reliance among adversarial successes ---")
    print("  (checks whichever target was ACTUALLY completed last chronologically,")
    print("   not a statically-assumed 'final target': the AND is order-blind, so")
    print("   zone_enter firing early and staying satisfied means some other pickup")
    print("   can be the one that completes the objective)")
    adv_successes = [r for r in rows if r["adversarial_success"]]
    nudge_only = [r for r in adv_successes if r["adv_nudge_only_final_leg"]]
    print(f"  adversarial successes: {len(adv_successes)}")
    print(f"  of those, the actual last-completed target's path was trivial nudge-only (len==1): {len(nudge_only)}")
    for r in nudge_only:
        print(f"     {r['scene_id']} combo={r['combo']} exploit={r['exploit_class']} last_target={r['adv_last_target']}")


if __name__ == "__main__":
    main()
