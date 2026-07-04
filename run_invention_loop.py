#!/usr/bin/env python3
"""
The invention loop (harness/invention/): explore -> play -> work -> exploit.
Warm-starts from the existing generation-policy adaptation combos, then
mutates/evaluates/archives new scenes round by round via a MAP-Elites
archive keyed on (path-length, interaction-count) behavior bins. Runs fully
deterministically with no API key by default; pass --claude to layer
Claude-driven scene revision on top (falls back to deterministic mutation
automatically if no key is set or a revision attempt fails).

Run:  source .venv/bin/activate && python3 run_invention_loop.py [--claude] [--rounds N] [--batch-size N]
Output: runs/invention/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "invention")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", action="store_true", help="layer Claude scene revision on top of mutation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from harness.generation.archive_heatmap import render_archive_heatmap_grid, render_coverage_trend
    from harness.invention.loop import run_invention_loop

    reviser = None
    if args.claude:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        if not has_key:
            print("--claude given but no ANTHROPIC_API_KEY set. Proceeding with deterministic mutation only.")
        else:
            import anthropic

            from harness.claude_agent.scene_reviser import REQUEST_TIMEOUT_SECONDS, make_reviser
            reviser = make_reviser(anthropic.Anthropic())
            print("Claude scene revision enabled (falls back to deterministic mutation on any failure).")

    print(f"Running the invention loop: {args.rounds} rounds x {args.batch_size} candidates "
          f"(seed={args.seed}, deterministic mutation{' + Claude revision' if reviser else ''})...\n")
    if reviser:
        print(f"(--claude makes one real, blocking network call per candidate, up to "
              f"{args.rounds * args.batch_size} total this run, each capped at "
              f"{REQUEST_TIMEOUT_SECONDS:.0f}s. Progress prints per candidate below, "
              f"since this can take a while.)\n")

    def on_round_start(label):
        print(f"-- {label} --", flush=True)

    def on_candidate(i, n):
        print(f"  candidate {i}/{n}...", end="", flush=True)

    def on_candidate_done(i, n, entry):
        if not entry.get("move"):
            outcome = f"skipped ({entry.get('reason', 'unknown')})"
        elif entry.get("reason") == "not_novel":
            outcome = f"skipped (too similar to an existing archive entry, move={entry['move']})"
        else:
            frontier_tag = " [FRONTIER-WORTHY]" if entry.get("frontier_worthy") else ""
            outcome = (f"{'accepted' if entry['accepted'] else 'evaluated, not accepted'} "
                       f"(move={entry['move']}, exploit={entry.get('exploit_class')}){frontier_tag}")
        print(f" -> {outcome}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    archive, rounds = run_invention_loop(num_rounds=args.rounds, batch_size=args.batch_size,
                                          seed=args.seed, reviser=reviser,
                                          on_round_start=on_round_start, on_candidate=on_candidate,
                                          on_candidate_done=on_candidate_done)
    elapsed = time.time() - t0

    print()
    for round_report in rounds:
        label = round_report["round"] if isinstance(round_report["round"], str) else f"round {round_report['round']}"
        n_accepted = sum(1 for r in round_report["results"] if r.get("accepted"))
        print(f"[{label:>8}] {n_accepted}/{len(round_report['results'])} accepted into the archive, "
              f"coverage now {round_report['coverage']:.2%}")

    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump({
            "elapsed_seconds": round(elapsed, 2),
            "rounds": rounds,
            "archive": archive.to_report(),
        }, f, indent=2, default=str)

    heatmap_path = os.path.join(OUTPUT_DIR, "archive_heatmap.png")
    render_archive_heatmap_grid(archive, heatmap_path)
    trend_path = os.path.join(OUTPUT_DIR, "coverage_trend.png")
    render_coverage_trend(rounds, trend_path)

    print(f"\n{elapsed:.1f}s total. Final archive coverage: {archive.coverage():.2%} "
          f"({len(archive.cells)}/{archive.total_cells} cells filled).")
    print(f"Report written to {report_path}")
    print(f"Archive heatmap: {heatmap_path}")
    print(f"Coverage trend: {trend_path}")


if __name__ == "__main__":
    main()
