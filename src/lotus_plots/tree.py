"""Turn taxonomy paths into a leaf order and a scipy linkage matrix.

The dendrograms drawn next to the heatmaps are not computed from the counts:
they encode the classification already present in the metadata. Leaves that
share their first k path elements merge at height `depth - k`, so siblings join
at height 1 and the root at height `depth`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

Path = tuple[str, ...]


def tree_order(paths: Sequence[Path], weights: Mapping[Path, int]) -> list[Path]:
    """Sort leaf paths so every subtree is contiguous, heaviest subtree first."""
    subtree: dict[Path, int] = defaultdict(int)
    for path in paths:
        for k in range(1, len(path) + 1):
            subtree[path[:k]] += weights[path]

    def key(path: Path):
        return [(-subtree[path[: k + 1]], path[k]) for k in range(len(path))]

    return sorted(paths, key=key)


def taxonomy_linkage(paths: Sequence[Path]) -> np.ndarray:
    """Linkage matrix whose dendrogram reproduces the taxonomy of `paths`.

    `paths` must come from `tree_order` (subtrees contiguous) and have equal
    length. Multifurcations become chains of binary merges at the same height,
    which keeps the given leaf order in `scipy.cluster.hierarchy.dendrogram`.
    """
    if len(paths) < 2:
        raise ValueError("a dendrogram needs at least two leaves")
    depth = len(paths[0])
    rows: list[list[float]] = []
    next_id = len(paths)

    def build(lo: int, hi: int, level: int) -> tuple[int, int]:
        nonlocal next_id
        if hi - lo == 1:
            return lo, 1
        if level == depth:
            raise ValueError(f"duplicate leaf path {paths[lo]}")
        bounds = [lo] + [i for i in range(lo + 1, hi) if paths[i][level] != paths[i - 1][level]] + [hi]
        clusters = [build(a, b, level + 1) for a, b in zip(bounds, bounds[1:])]
        node, size = clusters[0]
        for other, other_size in clusters[1:]:
            size += other_size
            rows.append([node, other, depth - level, size])
            node, next_id = next_id, next_id + 1
        return node, size

    build(0, len(paths), 0)
    return np.asarray(rows, dtype=float)
