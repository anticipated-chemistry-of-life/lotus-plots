"""Heatmaps of unique 2D molecules per NPClassifier label × organism taxon."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path as FilePath

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch

from .tree import Path, taxonomy_linkage, tree_order

# Fixed colours so a pathway / kingdom looks the same in every figure.
_tab10 = sns.color_palette("tab10")
_set2 = sns.color_palette("Set2")
PATHWAY_COLORS = {
    "Terpenoids": _tab10[0],
    "Shikimates and Phenylpropanoids": _tab10[1],
    "Alkaloids": _tab10[2],
    "Polyketides": _tab10[3],
    "Amino acids and Peptides": _tab10[4],
    "Fatty acids": _tab10[5],
    "Carbohydrates": _tab10[6],
}
KINGDOM_COLORS = {
    "Archaeplastida": _set2[0],
    "Fungi": _set2[1],
    "Metazoa": _set2[2],
    "Other eukaryotes": _set2[3],
    "Bacteria": _set2[4],
    "Archaea": _set2[5],
}
ANNOTATE_MAX_CELLS = 400


def select_leaves(weights: Mapping[Path, int], limit: int) -> list[Path]:
    """The `limit` heaviest leaves, in taxonomy order."""
    kept = sorted(weights, key=lambda p: (-weights[p], p))[:limit]
    return tree_order(kept, weights)


def leaf_labels(paths: Sequence[Path]) -> list[str]:
    """Leaf names, suffixed with their parent when a name repeats."""
    repeats = Counter(p[-1] for p in paths)
    return [
        f"{p[-1]} ({p[-2]})" if repeats[p[-1]] > 1 and len(p) > 1 else p[-1] for p in paths
    ]


def _strip(paths: Sequence[Path], labels: Sequence[str], level: int, colors, name: str):
    """Colour strip marking the ancestor at `level` of every leaf."""
    groups = list(dict.fromkeys(p[level] for p in paths))
    strip = pd.Series([colors[p[level]] for p in paths], index=labels, name=name)
    return strip, [Patch(color=colors[g], label=g) for g in groups], name


def _cell_size(n: int) -> float:
    """Cell edge in inches for an axis with `n` leaves."""
    return 0.55 if n <= 12 else 0.32 if n <= 50 else 0.2


ENRICHMENT_PSEUDOCOUNT = 1.0


def count_matrix(
    counts: pl.DataFrame, chem_paths: Sequence[Path], org_paths: Sequence[Path]
) -> np.ndarray:
    """Observed unique molecules for the shown leaves (rows: chemistry, cols: organisms)."""
    chem_index = {p[-1]: i for i, p in enumerate(chem_paths)}
    org_index = {p: i for i, p in enumerate(org_paths)}
    matrix = np.zeros((len(chem_paths), len(org_paths)), dtype=np.int64)
    for row in counts.iter_rows():
        i = chem_index.get(row[0])
        j = org_index.get(tuple(row[1:-1]))
        if i is not None and j is not None:
            matrix[i, j] = row[-1]
    return matrix


def expected_matrix(
    counts: pl.DataFrame, chem_paths: Sequence[Path], org_paths: Sequence[Path]
) -> np.ndarray:
    """Counts expected if chemistry and organisms were independent.

    E = row total × column total / grand total, with totals over the whole
    table (not only the shown leaves), as in a chi-square test.
    """
    chem_col, *org_cols, value = counts.columns
    rows = dict(counts.group_by(chem_col).agg(pl.col(value).sum()).iter_rows())
    cols = {
        tuple(r[:-1]): r[-1] for r in counts.group_by(org_cols).agg(pl.col(value).sum()).iter_rows()
    }
    total = counts[value].sum()
    row_totals = np.array([rows.get(p[-1], 0) for p in chem_paths], dtype=float)
    col_totals = np.array([cols.get(p, 0) for p in org_paths], dtype=float)
    return np.outer(row_totals, col_totals) / total


SCALES = ("log", "linear", "enrichment")


@dataclass(frozen=True)
class ColorScale:
    values: np.ndarray
    mask: np.ndarray
    norm: Normalize
    cmap: str
    fmt: str
    label: str


def color_scale(scale: str, observed: np.ndarray, expected: np.ndarray) -> ColorScale:
    """Cell values and colour mapping for one of `SCALES`."""
    label = "unique molecules (2D InChIKey)"
    vmax = max(int(observed.max()), 2)
    if scale == "log":
        return ColorScale(
            observed, observed == 0, LogNorm(vmin=1, vmax=vmax), "viridis", "d", f"{label}, log scale"
        )
    if scale == "linear":
        return ColorScale(
            observed, observed == 0, Normalize(vmin=0, vmax=vmax), "viridis", "d", f"{label}, linear scale"
        )
    if scale == "enrichment":
        # Pseudocount keeps zero / near-zero cells finite and shrinks noisy small cells to 0.
        ratio = np.log2((observed + ENRICHMENT_PSEUDOCOUNT) / (expected + ENRICHMENT_PSEUDOCOUNT))
        limit = max(float(np.quantile(np.abs(ratio), 0.99)), 1.0)
        return ColorScale(
            ratio,
            np.zeros_like(observed, dtype=bool),
            Normalize(vmin=-limit, vmax=limit),
            "RdBu_r",
            ".1f",
            f"log2(observed / expected) {label}",
        )
    raise ValueError(f"unknown scale {scale!r}")


def plot_heatmap(
    colors: ColorScale,
    chem_paths: Sequence[Path],
    org_paths: Sequence[Path],
    chem_level: str,
    org_rank: str,
    title: str,
    out_stem: FilePath,
) -> None:
    """Draw a matrix over the given (taxonomy-ordered) leaves; save PNG and PDF."""
    data = pd.DataFrame(
        colors.values, index=leaf_labels(chem_paths), columns=leaf_labels(org_paths)
    )
    n_rows, n_cols = colors.values.shape
    cell_w, cell_h = _cell_size(n_cols), _cell_size(n_rows)
    row_tree = len(chem_paths[0]) > 1
    col_tree = len(org_paths[0]) > 1

    strips = []
    row_colors = col_colors = None
    if len(chem_paths[0]) > 1:
        row_colors, handles, name = _strip(chem_paths, data.index, 0, PATHWAY_COLORS, "pathway")
        strips.append((handles, name))
    # Organism paths start at domain, so index 1 is the kingdom.
    if len(org_paths[0]) > 2:
        col_colors, handles, name = _strip(org_paths, data.columns, 1, KINGDOM_COLORS, "kingdom")
        strips.append((handles, name))

    # Layout in inches; tick labels, legends and colour bar spill outside the
    # figure and are kept by bbox_inches="tight".
    row_dendro = 1.6 if row_tree else 0.0
    col_dendro = 1.4 if col_tree else 0.0
    row_strip = 0.2 if row_colors is not None else 0.0
    col_strip = 0.2 if col_colors is not None else 0.0
    gap = 0.03
    heat_w, heat_h = n_cols * cell_w, n_rows * cell_h
    width = row_dendro + row_strip + heat_w + 2 * gap
    height = col_dendro + col_strip + heat_h + 2 * gap

    cell = min(cell_w, cell_h)
    font = 11 if cell >= 0.5 else 8 if cell >= 0.3 else 6.5
    annotate = n_rows * n_cols <= ANNOTATE_MAX_CELLS
    # clustermap's internal tight_layout is irrelevant: axes are placed below.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied")
        grid = sns.clustermap(
            data,
            row_linkage=taxonomy_linkage(chem_paths) if row_tree else None,
            col_linkage=taxonomy_linkage(org_paths) if col_tree else None,
            row_cluster=row_tree,
            col_cluster=col_tree,
            row_colors=row_colors,
            col_colors=col_colors,
            mask=colors.mask,
            norm=colors.norm,
            cmap=colors.cmap,
            annot=annotate,
            fmt=colors.fmt,
            annot_kws={"fontsize": font - 2.5},
            linewidths=0.4 if annotate else 0,
            linecolor="white",
            figsize=(width, height),
            dendrogram_ratio=(0.1, 0.1),
            colors_ratio=(0.05, 0.05),
            cbar_pos=None,
            xticklabels=True,
            yticklabels=True,
            tree_kws={"linewidths": 0.8},
        )
    fig = grid.fig

    def place(ax, x: float, y: float, w: float, h: float) -> None:
        if ax is not None:
            ax.set_position((x / width, y / height, max(w, 1e-3) / width, max(h, 1e-3) / height))

    heat_x = row_dendro + row_strip + 2 * gap
    place(grid.ax_row_dendrogram, 0.0, 0.0, row_dendro, heat_h)
    place(grid.ax_row_colors, row_dendro + gap, 0.0, row_strip, heat_h)
    place(grid.ax_heatmap, heat_x, 0.0, heat_w, heat_h)
    place(grid.ax_col_colors, heat_x, heat_h + gap, heat_w, col_strip)
    place(grid.ax_col_dendrogram, heat_x, heat_h + col_strip + 2 * gap, heat_w, col_dendro)

    heat = grid.ax_heatmap
    heat.set_facecolor("#efefef")
    heat.set_xlabel(f"organism {org_rank} (Open Tree of Life)", fontsize=font + 1)
    heat.set_ylabel(f"NPClassifier {chem_level}", fontsize=font + 1)
    heat.tick_params(axis="x", labelsize=font, labelrotation=90)
    heat.tick_params(axis="y", labelsize=font, labelrotation=0)
    for ax in (grid.ax_row_colors, grid.ax_col_colors):
        if ax is not None:
            ax.tick_params(labelsize=font)

    # Colour bar to the right of the row labels, aligned with the heatmap top.
    fig.canvas.draw()
    labels_right = heat.get_tightbbox().transformed(fig.dpi_scale_trans.inverted()).x1
    bar_h = min(2.5, heat_h)
    cax = fig.add_axes(
        ((labels_right + 0.3) / width, (heat_h - bar_h) / height, 0.2 / width, bar_h / height)
    )
    bar = fig.colorbar(heat.collections[0], cax=cax)
    bar.set_label(colors.label, fontsize=font + 1)
    bar.ax.tick_params(labelsize=font)

    anchor_y = 1.0
    for handles, name in strips:
        legend = fig.legend(
            handles=handles,
            title=name,
            loc="upper right",
            bbox_to_anchor=(0.0, anchor_y),
            fontsize=font,
            title_fontsize=font + 1,
            frameon=False,
        )
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        anchor_y = box.y0 - 0.02
    fig.suptitle(title, x=0.0, y=1.0, ha="left", va="bottom", fontsize=font + 3)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
