"""
MAP-Elites archive heatmap: path-length bin x interaction-count bin, cell
color = occupant fitness. Static matplotlib PNG, same dark aesthetic as
generation/adaptation_chart.py and solvers/q_learner_chart.py.

Sequential palette: the dataviz skill's validated default sequential hue
(blue, references/palette.md's 100->700 ramp), reversed for this dark chart
surface (#1a1a19) rather than used as documented, since the reference ramp is
specified for a light surface where "near zero recedes toward the (light)
surface." On a dark surface the analogous "recede" color is the darkest
step, not the lightest, so low-fitness cells map to the ramp's dark end
(blending toward the surface) and high-fitness cells map to its light end
(popping against it), a stated, reasoned adaptation, not a guessed palette.
Could not run the skill's node-based validator directly in this environment
(node isn't installed, same limitation noted in adaptation_chart.py), so
this relies on the documented reference values rather than a fresh
validation run.

Unfilled cells (never reached by the invention loop, not merely a genuine
0.0-fitness occupant) get a distinct flat neutral, not a very-dark blue,
so "coverage filling in over rounds" stays legible as a real gauge instead of
looking identical to "filled but bad."
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SURFACE = "#1a1a19"
PRIMARY_INK = "#ffffff"
SECONDARY_INK = "#c3c2b7"
MUTED_INK = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
EMPTY_CELL_COLOR = "#242422"

# references/palette.md's sequential blue ramp, 100->700 (light->dark on a
# light surface); reversed below for use against our dark surface.
_RAMP_100_TO_700 = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_DARK_SURFACE_RAMP = list(reversed(_RAMP_100_TO_700))  # index 0 = darkest (near-zero), -1 = lightest (max)


def _fitness_color(fitness: float) -> str:
    idx = round(max(0.0, min(1.0, fitness)) * (len(_DARK_SURFACE_RAMP) - 1))
    return _DARK_SURFACE_RAMP[idx]


def _label_color(fitness: float) -> str:
    # Dark ramp steps need light text; light ramp steps (high fitness) need dark text.
    idx = round(max(0.0, min(1.0, fitness)) * (len(_DARK_SURFACE_RAMP) - 1))
    return "#0b0b0b" if idx >= len(_DARK_SURFACE_RAMP) - 5 else "#ffffff"


def render_archive_heatmap_grid(archive, output_path: str):
    """`archive` is a harness.invention.archive.MapElitesArchive instance."""
    rows, cols = archive.path_length_bins, archive.interaction_bins
    fig, ax = plt.subplots(figsize=(1.4 * cols + 2, 1.4 * rows + 1.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for p in range(rows):
        for i in range(cols):
            cell = archive.cells.get((p, i))
            x, y = i, rows - 1 - p  # path-length increases upward
            if cell is None:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=EMPTY_CELL_COLOR, edgecolor=GRIDLINE, linewidth=1))
                ax.text(x + 0.5, y + 0.5, "--", ha="center", va="center", color=MUTED_INK, fontsize=9)
            else:
                fitness = cell["fitness"]
                color = _fitness_color(fitness)
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=1))
                ax.text(x + 0.5, y + 0.5, f"{fitness:.2f}", ha="center", va="center",
                        color=_label_color(fitness), fontsize=10, fontweight="bold")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([c + 0.5 for c in range(cols)])
    ax.set_xticklabels([f"{c}" if c < cols - 1 else f"{c}+" for c in range(cols)], color=SECONDARY_INK, fontsize=10)
    ax.set_yticks([r + 0.5 for r in range(rows)])
    ax.set_yticklabels([f"bin {rows - 1 - r}" for r in range(rows)], color=SECONDARY_INK, fontsize=10)
    ax.set_xlabel("interaction-count bin (distinct non-platform primitive types)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("path-length bin (longer honest solve →)", color=SECONDARY_INK, fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    coverage = archive.coverage()
    ax.set_title(f"MAP-Elites archive: {len(archive.cells)}/{archive.total_cells} cells filled ({coverage:.0%} coverage)",
                 color=PRIMARY_INK, fontsize=13, pad=14, loc="left")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)


# Categorical slot 4 (green) per references/palette.md's fixed hue order.
# Slots 1-3 (blue/aqua/yellow) are already used by generation/adaptation_chart.py's
# trend chart. That chart plots a different report shape (policy_adaptation's
# per-combo aggregate rounds), so archive coverage gets its own single-series
# chart here rather than being spliced in as a fourth line on an unrelated
# dataset: same style tokens, separate figure, since "one axis, one job"
# argues against merging two different loops' round-numbering onto one plot.
COVERAGE_COLOR = "#008300"


def render_coverage_trend(rounds: list[dict], output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = list(range(len(rounds)))
    y = [r["coverage"] for r in rounds]
    labels = [str(r["round"]) if isinstance(r["round"], str) else f"round {r['round']}" for r in rounds]

    ax.plot(x, y, color=COVERAGE_COLOR, linewidth=2, solid_capstyle="round", marker="o", markersize=8)
    ax.annotate(f"{y[-1]:.0%}", (x[-1], y[-1]), textcoords="offset points", xytext=(8, 0),
                color=COVERAGE_COLOR, fontsize=10, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=SECONDARY_INK, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="y", colors=MUTED_INK, labelsize=9)

    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE)

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Invention loop: archive coverage over rounds", color=PRIMARY_INK, fontsize=13, pad=14, loc="left")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
