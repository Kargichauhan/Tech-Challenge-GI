"""
Learning-curve chart for the tabular Q-learner (solvers/q_learner.py): episode
return over training. Static matplotlib PNG, same dark aesthetic and palette
discipline as generation/adaptation_chart.py (categorical slot 1, blue).
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
LINE_COLOR = "#3987e5"  # slot 1: blue, same as adaptation_chart.py's genuine_solve_rate series


def _moving_average(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def render_learning_curve(returns: list[float], scene_id: str, output_path: str, window: int = 25):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = list(range(len(returns)))
    ax.plot(x, returns, color=LINE_COLOR, linewidth=0.6, alpha=0.35)
    smoothed = _moving_average(returns, window)
    ax.plot(x, smoothed, color=LINE_COLOR, linewidth=2.2, solid_capstyle="round",
             label=f"{window}-episode moving average")

    ax.tick_params(axis="y", colors=MUTED_INK, labelsize=9)
    ax.tick_params(axis="x", colors=SECONDARY_INK, labelsize=9)
    ax.set_xlabel("training episode", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("episode return", color=SECONDARY_INK, fontsize=10)

    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE)

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_title(f"Q-learner training curve: {scene_id}", color=PRIMARY_INK, fontsize=13, pad=14, loc="left")
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
