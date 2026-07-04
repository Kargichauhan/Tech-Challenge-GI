"""
Trend chart for generation-policy adaptation: genuine-solve-rate,
adversarial-exploit-rate, and validator-rejection-rate across rounds.
Static matplotlib PNG, dark aesthetic matching render/scene_renderer.py.

Palette: fixed-order categorical slots 1-3 (blue/aqua/yellow) from the
dataviz skill's validated dark-mode palette (references/palette.md) -- these
were chosen and validated as a set for adjacent-pair colorblind-safe
separation; using the first three in their documented order is the safe
subset. (Could not run the JS validator directly in this environment --
`node` isn't installed -- so this relies on the pre-validated documented
values rather than a fresh validation run.)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#1a1a19"
PRIMARY_INK = "#ffffff"
SECONDARY_INK = "#c3c2b7"
MUTED_INK = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"

SERIES = [
    ("genuine_solve_rate", "genuine solve rate", "#3987e5"),   # slot 1: blue
    ("exploit_rate", "adversarial exploit rate", "#199e70"),   # slot 2: aqua
    ("validator_rejection_rate", "validator rejection rate", "#c98500"),  # slot 3: yellow
]


def render_trend_chart(rounds: list[dict], output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = [r["round"] for r in rounds]

    for key, label, color in SERIES:
        y = [r["aggregate"]["overall"][key] for r in rounds]
        ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", marker="o", markersize=8, label=label)
        # direct end-of-line label, since with 3 rounds a legend alone is a bit sparse
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(8, 0),
                    color=color, fontsize=10, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"round {i}" for i in x], color=SECONDARY_INK, fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="y", colors=MUTED_INK, labelsize=9)
    ax.tick_params(axis="x", colors=SECONDARY_INK)

    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE)

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_title("Generation-policy adaptation across rounds", color=PRIMARY_INK, fontsize=13, pad=14, loc="left")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
