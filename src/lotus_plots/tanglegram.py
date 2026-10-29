"""Taxonomy versus chemistry: do related organisms make similar molecules?

Each organism taxon gets a chemical profile (share of its molecules in every
NPClassifier class). Taxa are clustered on Bray–Curtis distance between
profiles (average linkage) and that tree is drawn facing the taxonomy tree.
Agreement is measured with a Mantel test (Spearman, permutations) between
profile distances and taxonomic distances, and with the correlation between
the cophenetic distances of the two trees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from . import data
from .plot import KINGDOM_COLORS, leaf_labels
from .tree import Path, taxonomy_linkage, tree_order

MANTEL_PERMUTATIONS = 999


@dataclass(frozen=True)
class Agreement:
    rank: str
    taxa: int
    mantel_r: float
    mantel_p: float
    cophenetic_r: float


def chemical_profiles(
    class_tokens: pl.DataFrame, organisms: pl.DataFrame, rank: str, min_molecules: int, max_taxa: int
) -> tuple[list[Path], np.ndarray]:
    """Taxa (taxonomy-ordered paths) and their row-normalised class profiles.

    Keeps taxa with at least `min_molecules` class-labelled molecules, at most
    the `max_taxa` richest ones.
    """
    counts = data.count_table(class_tokens, organisms, "class", rank)
    _, org_totals = data.totals(class_tokens, organisms, "class", rank)
    weights = {tuple(r[:-1]): r[-1] for r in org_totals.iter_rows() if r[-1] >= min_molecules}
    kept = sorted(weights, key=lambda p: (-weights[p], p))[:max_taxa]
    paths = tree_order(kept, weights)

    index = {p: i for i, p in enumerate(paths)}
    labels = sorted(counts[data.chem_column("class")].unique())
    column = {c: j for j, c in enumerate(labels)}
    matrix = np.zeros((len(paths), len(labels)))
    for label, *path, n in counts.iter_rows():
        i = index.get(tuple(path))
        if i is not None:
            matrix[i, column[label]] = n
    return paths, matrix / matrix.sum(axis=1, keepdims=True)


def mantel(a: np.ndarray, b: np.ndarray, permutations: int, rng: np.random.Generator):
    """Spearman Mantel statistic between two square distance matrices, one-sided p."""
    upper = np.triu_indices_from(a, k=1)
    observed = spearmanr(a[upper], b[upper]).statistic
    exceed = 0
    for _ in range(permutations):
        p = rng.permutation(len(b))
        exceed += spearmanr(a[upper], b[p][:, p][upper]).statistic >= observed
    return float(observed), float((exceed + 1) / (permutations + 1))


def _untangle(tree: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    """Swap children of every merge so leaves follow `target_position` where possible.

    Each child is placed by the mean target position of its leaves; this
    reduces crossing lines without changing the tree itself.
    """
    n = len(tree) + 1
    total = list(target_position) + [0.0] * (n - 1)
    size = [1] * n + [0] * (n - 1)
    out = tree.copy()
    for k, (a, b, *_rest) in enumerate(tree):
        a, b = int(a), int(b)
        if total[a] / size[a] > total[b] / size[b]:
            out[k, 0], out[k, 1] = b, a
        total[n + k] = total[a] + total[b]
        size[n + k] = size[a] + size[b]
    return out


def compare(paths: Sequence[Path], profiles: np.ndarray, rank: str, seed: int = 0):
    """Both linkages (taxonomy, chemistry) and their agreement statistics."""
    taxonomy = taxonomy_linkage(paths)
    chemistry = linkage(pdist(profiles, "braycurtis"), method="average")
    chemistry = _untangle(chemistry, np.arange(len(paths), dtype=float))
    taxonomic_distance = cophenet(taxonomy)
    r, p = mantel(
        squareform(pdist(profiles, "braycurtis")),
        squareform(taxonomic_distance),
        MANTEL_PERMUTATIONS,
        np.random.default_rng(seed),
    )
    cophenetic_r = float(np.corrcoef(taxonomic_distance, cophenet(chemistry))[0, 1])
    return taxonomy, chemistry, Agreement(rank, len(paths), r, p, cophenetic_r)


def plot_tanglegram(
    paths: Sequence[Path],
    taxonomy: np.ndarray,
    chemistry: np.ndarray,
    agreement: Agreement,
    title: str,
    out_stem: FilePath,
) -> None:
    n = len(paths)
    labels = leaf_labels(paths)
    colors = [KINGDOM_COLORS[p[1]] for p in paths]
    leaf_h, font = 0.17, 7
    height = n * leaf_h + 0.6
    fig = plt.figure(figsize=(11, height))
    left = fig.add_axes((0.0, 0.0, 0.25, 1.0))
    middle = fig.add_axes((0.25, 0.0, 0.5, 1.0))
    right = fig.add_axes((0.75, 0.0, 0.25, 1.0))

    black = {"link_color_func": lambda _: "black"}
    tax = dendrogram(taxonomy, orientation="left", no_labels=True, ax=left, **black)
    chem = dendrogram(chemistry, orientation="right", no_labels=True, ax=right, **black)
    # dendrogram puts leaf i of its order at y = 5 + 10 i.
    y_tax = {leaf: 5 + 10 * i for i, leaf in enumerate(tax["leaves"])}
    y_chem = {leaf: 5 + 10 * i for i, leaf in enumerate(chem["leaves"])}

    for ax in (left, middle, right):
        ax.set_ylim(10 * n, 0)
    for ax in (left, right):
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(labelsize=font)
    left.set_xlabel("taxonomic distance (ranks)", fontsize=font + 1)
    right.set_xlabel("Bray–Curtis distance of class profiles (average linkage)", fontsize=font + 1)
    left.set_title("Open Tree of Life taxonomy", fontsize=font + 2)
    right.set_title("chemical-profile clustering", fontsize=font + 2)

    middle.set_xlim(0, 1)
    middle.set_axis_off()
    for leaf in range(n):
        middle.text(0.01, y_tax[leaf], labels[leaf], ha="left", va="center", fontsize=font, color="black")
        middle.text(0.99, y_chem[leaf], labels[leaf], ha="right", va="center", fontsize=font, color="black")
        middle.plot([0.34, 0.66], [y_tax[leaf], y_chem[leaf]], color=colors[leaf], lw=1.1, alpha=0.9)

    kingdoms = list(dict.fromkeys(p[1] for p in paths))
    handles = [plt.Line2D([], [], color=KINGDOM_COLORS[k], lw=3, label=k) for k in kingdoms]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.4 / height), ncol=len(handles),
               fontsize=font + 1, frameon=False)
    stats = (
        f"{agreement.taxa} taxa · Mantel r (Spearman) = {agreement.mantel_r:.2f}, "
        f"p = {agreement.mantel_p:.3f} ({MANTEL_PERMUTATIONS} permutations) · "
        f"cophenetic correlation between trees = {agreement.cophenetic_r:.2f}"
    )
    fig.suptitle(f"{title}\n{stats}", x=0.0, y=1.0 + 0.3 / height, ha="left", va="bottom", fontsize=font + 3)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
