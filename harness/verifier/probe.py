"""Runs the genuine solver and the adversarial probe on the same scene (each
gets its own fresh Engine instance, since both mutate state) and reports a
verifier-robustness result for that scene."""

from __future__ import annotations

from harness.engine.physics_engine import Engine
from harness.solvers.adversarial import AdversarialSolver
from harness.solvers.genuine import GenuineSolver
from harness.verifier.exploit_classifier import classify_exploit, explain


def run_verifier_probe(scene: dict, total_max_ticks: int = 60 * 25) -> dict:
    genuine_engine = Engine(scene)
    genuine_result, genuine_log, genuine_clips, genuine_ticks = GenuineSolver(scene).solve(genuine_engine, total_max_ticks)

    adv_engine = Engine(scene)
    adv_result, adv_log, adv_clips, adv_ticks = AdversarialSolver(scene).solve(adv_engine, total_max_ticks)

    adversarial_success = adv_result == "success"
    exploit_class = classify_exploit(scene, adv_log, adv_clips) if adversarial_success else "none"
    explanation = explain(scene, adv_log, adv_clips, exploit_class) if adversarial_success else None

    return {
        "scene_id": scene["id"],
        "genuine_result": genuine_result,
        "genuine_ticks": genuine_ticks,
        # Whether the genuine solver confirmed this scene is honestly
        # solvable at all. An exploit found alongside genuine_result !=
        # "success" is a weaker claim: there's no confirmed honest
        # baseline the adversary shortcut past, only that it reached success
        # via a route that skipped the intended order. Downstream reporting
        # (GIF captions, aggregate stats) should distinguish the two rather
        # than presenting every exploit_class != "none" the same way.
        "honest_baseline_confirmed": genuine_result == "success",
        "adversarial_success": adversarial_success,
        "adversarial_ticks": adv_ticks,
        "exploit_class": exploit_class,
        "explanation": explanation,
        "genuine_event_log": genuine_log,
        "adversarial_event_log": adv_log,
    }
