#!/usr/bin/env python3
"""
Verifier-feedback-driven generation-policy adaptation (plan item 7): runs
N=8 environments/round for 3 rounds over a fixed list of named primitive
combos, scoring each combo on genuine-solve-rate, adversarial-exploit-rate,
and validator-rejection-rate, and feeding that back as a weighted-sampling
distribution for the next round. Deterministic, no API key.

Run:  source .venv/bin/activate && python3 run_generation_adaptation.py
Output: runs/adaptation/report.json, runs/adaptation/trend.png
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "adaptation")
NUM_ROUNDS = 3
BATCH_SIZE = 8


def main():
    from harness.generation.adaptation_chart import render_trend_chart
    from harness.generation.policy_adaptation import COMBOS, run_adaptation

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Running generation-policy adaptation: {NUM_ROUNDS} rounds x {BATCH_SIZE} environments, "
          f"fixed combo list = {list(COMBOS.keys())}\n")

    rounds = run_adaptation(num_rounds=NUM_ROUNDS, batch_size=BATCH_SIZE, seed=0)

    for r in rounds:
        agg = r["aggregate"]["overall"]
        print(f"--- round {r['round']} ---")
        print(f"  sampled: {r['sampled_combo_names']}")
        print(f"  overall: genuine_solve_rate={agg['genuine_solve_rate']:.2f} "
              f"exploit_rate={agg['exploit_rate']:.2f} validator_rejection_rate={agg['validator_rejection_rate']:.2f}")
        for name, stats in r["aggregate"]["per_combo"].items():
            if stats is None:
                print(f"    {name:<20} not sampled this round")
            else:
                print(f"    {name:<20} n={stats['n_generated']:<2} solve={stats['genuine_solve_rate']:.2f} "
                      f"exploit={stats['exploit_rate']:.2f} reject={stats['validator_rejection_rate']:.2f}")
        print(f"  weights for next round: " + ", ".join(f"{k}={v:.2f}" for k, v in r["weights_next"].items()))
        print()

    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(rounds, f, indent=2, default=str)

    chart_path = os.path.join(OUTPUT_DIR, "trend.png")
    render_trend_chart(rounds, chart_path)

    print(f"Report: {report_path}")
    print(f"Trend chart: {chart_path}")


if __name__ == "__main__":
    main()
