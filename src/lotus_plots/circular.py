"""Circular genus-level taxonomy tree with NPClassifier pathway rings.

The tree is the Open Tree of Life lineage from the metadata (domain → genus).
Around it: a kingdom ring, one ring per pathway whose opacity is the share of
the genus's molecules in that pathway, and an outer bar of the genus's total
unique molecules (log scale).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from . import data
from .plot import KINGDOM_COLORS, PATHWAY_COLORS
from .tree import Path, tree_order

GAP = np.deg2rad(6)
"""Empty wedge at the top so the circle has a visible start and end."""
TREE_RADIUS = 1.0
RING_WIDTH = 0.055
RING_GAP = 0.008


def genus_profiles(
    pathway_tokens: pl.DataFrame, organisms: pl.DataFrame, min_molecules: int
) -> tuple[list[Path], dict[Path, int], np.ndarray]:
    """Genera with ≥ `min_molecules` pathway-labelled molecules, taxonomy-ordered.

    Returns the paths, their molecule totals, and a genus × pathway matrix of
    shares (molecules in the pathway / molecules of the genus).
    """
    counts = data.count_table(pathway_tokens, organisms, "pathway", "genus")
    _, totals = data.totals(pathway_tokens, organisms, "pathway", "genus")
    weights = {tuple(r[:-1]): r[-1] for r in totals.iter_rows() if r[-1] >= min_molecules}
    paths = tree_order(list(weights), weights)
    index = {p: i for i, p in enumerate(paths)}
    pathways = list(PATHWAY_COLORS)
    shares = np.zeros((len(paths), len(pathways)))
    for pathway, *path, n in counts.iter_rows():
        i = index.get(tuple(path))
        if i is not None:
            shares[i, pathways.index(pathway)] = n / weights[tuple(path)]
    return paths, weights, shares


def _tree_segments(paths: Sequence[Path], theta: np.ndarray) -> list[np.ndarray]:
    """Polar (θ, r) polylines drawing the rank-based tree from root to leaves."""
    depth = len(paths[0])
    span: dict[Path, list[float]] = defaultdict(lambda: [np.inf, -np.inf])
    for path, angle in zip(paths, theta, strict=True):
        for k in range(depth + 1):
            lo_hi = span[path[:k]]
            lo_hi[0], lo_hi[1] = min(lo_hi[0], angle), max(lo_hi[1], angle)

    def angle_of(node: Path) -> float:
        lo, hi = span[node]
        return (lo + hi) / 2

    def radius(level: int) -> float:
        return TREE_RADIUS * level / depth

    segments = []
    for node, (lo, hi) in span.items():
        level = len(node)
        if level == depth:
            continue
        r = radius(level)
        if hi > lo:
            arc = np.linspace(lo, hi, max(2, int(np.degrees(hi - lo) * 2)))
            segments.append(np.column_stack([arc, np.full_like(arc, r)]))
    for node in span:
        if node:
            a = angle_of(node)
            segments.append(np.array([[a, radius(len(node) - 1)], [a, radius(len(node))]]))
    return segments


def plot_circular(
    paths: Sequence[Path],
    weights: Mapping[Path, int],
    shares: np.ndarray,
    title: str,
    out_stem: FilePath,
) -> None:
    n = len(paths)
    step = (2 * np.pi - GAP) / n
    theta = np.pi / 2 - GAP / 2 - step * (np.arange(n) + 0.5)  # clockwise from the top

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(projection="polar")
    ax.set_axis_off()

    ax.add_collection(LineCollection(_tree_segments(paths, theta), colors="#555555", linewidths=0.35))

    r = TREE_RADIUS + 0.02
    ax.bar(theta, RING_WIDTH, width=step, bottom=r, color=[KINGDOM_COLORS[p[1]] for p in paths], linewidth=0)
    r += RING_WIDTH + 2 * RING_GAP
    for j, (pathway, color) in enumerate(PATHWAY_COLORS.items()):
        rgba = np.array([to_rgba(color, alpha) for alpha in shares[:, j]])
        ax.bar(theta, RING_WIDTH, width=step, bottom=r, color="#f2f2f2", linewidth=0)
        ax.bar(theta, RING_WIDTH, width=step, bottom=r, color=rgba, linewidth=0)
        r += RING_WIDTH + RING_GAP

    totals = np.array([weights[p] for p in paths], dtype=float)
    bar_max = 0.35
    log_totals = np.log10(totals)
    heights = bar_max * log_totals / log_totals.max()
    r += RING_GAP
    ax.bar(theta, heights, width=step, bottom=r, color="#6b6b6b", linewidth=0)
    for decade in range(1, int(log_totals.max()) + 1):
        ring = r + bar_max * decade / log_totals.max()
        arc = np.linspace(theta[0], theta[-1], 400)
        ax.plot(arc, np.full_like(arc, ring), color="#bbbbbb", lw=0.4, zorder=0)
        ax.text(np.pi / 2, ring, f"$10^{decade}$", ha="center", va="center", fontsize=7, color="#777777")

    # Phylum labels outside the bars for phyla spanning ≥ 1.5 % of the circle.
    label_r = r + bar_max + 0.05
    phyla: dict[Path, list[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        phyla[p[:3]].append(i)
    for phylum, members in phyla.items():
        if len(members) / n < 0.015:
            continue
        a = theta[members].mean()
        arc = np.linspace(theta[members[0]] + step / 2, theta[members[-1]] - step / 2, 50)
        ax.plot(arc, np.full_like(arc, label_r - 0.02), color=KINGDOM_COLORS[phylum[1]], lw=2.5)
        degrees = np.degrees(a) % 360
        flip = 90 < degrees < 270
        ax.text(
            a, label_r, phylum[-1], rotation=degrees + 180 if flip else degrees,
            rotation_mode="anchor", ha="right" if flip else "left", va="center", fontsize=10,
        )
    ax.set_ylim(0, label_r + 0.05)

    kingdoms = list(dict.fromkeys(p[1] for p in paths))
    fig.legend(
        handles=[Patch(color=KINGDOM_COLORS[k], label=k) for k in kingdoms],
        title="kingdom (inner ring)", loc="upper left", bbox_to_anchor=(0.02, 0.98),
        fontsize=10, title_fontsize=11, frameon=False,
    )
    fig.legend(
        handles=[Patch(color=c, label=p) for p, c in PATHWAY_COLORS.items()],
        title="pathway rings, inside → out\n(opacity = share of genus molecules)",
        loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10, title_fontsize=11, frameon=False,
    )
    fig.text(0.98, 0.02, "outer grey bars: unique molecules per genus (log scale)",
             ha="right", va="bottom", fontsize=10)
    fig.suptitle(title, x=0.02, y=1.0, ha="left", va="top", fontsize=14)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
