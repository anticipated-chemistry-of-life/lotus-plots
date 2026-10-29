"""Plots (heatmaps, tanglegrams, circular tree) of NPClassifier chemistry versus organism taxa in LOTUS."""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import polars as pl

from . import data
from .circular import genus_profiles, plot_circular
from .plot import SCALES, color_scale, count_matrix, expected_matrix, plot_heatmap, select_leaves
from .tanglegram import chemical_profiles, compare, plot_tanglegram

DEFAULT_ORG_RANKS = ["kingdom", "phylum", "class", "order", "family"]
DEFAULT_TANGLE_RANKS = ["phylum", "class", "order", "family"]
VERSION = data.METADATA_FILE.split("_")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument(
        "--chem-levels", nargs="+", choices=list(data.CHEM_LEVELS), default=list(data.CHEM_LEVELS)
    )
    parser.add_argument(
        "--org-ranks", nargs="+", choices=list(data.ORG_RANKS), default=DEFAULT_ORG_RANKS
    )
    parser.add_argument(
        "--max-rows", type=int, default=100, help="keep the N chemical labels with most molecules"
    )
    parser.add_argument(
        "--max-cols", type=int, default=100, help="keep the N organism taxa with most molecules"
    )
    parser.add_argument(
        "--tangle-ranks", nargs="+", choices=list(data.ORG_RANKS)[2:], default=DEFAULT_TANGLE_RANKS
    )
    parser.add_argument(
        "--tangle-min-molecules", type=int, default=100,
        help="tanglegram taxa need at least N class-labelled molecules",
    )
    parser.add_argument("--tangle-max-taxa", type=int, default=60)
    parser.add_argument(
        "--circular-min-molecules", type=int, default=50,
        help="circular tree genera need at least N pathway-labelled molecules",
    )
    args = parser.parse_args()

    metadata = data.load_metadata(data.fetch_metadata(args.data_dir))
    organisms = data.organism_table(metadata)
    heatmaps(args, metadata, organisms)
    tanglegrams(args, metadata, organisms)
    circular_tree(args, metadata, organisms)


def heatmaps(args: argparse.Namespace, metadata: pl.DataFrame, organisms: pl.DataFrame) -> None:
    chem_paths = data.chemical_paths(metadata)
    for chem_level in args.chem_levels:
        tokens = data.chemical_tokens(metadata, chem_level)
        for org_rank in args.org_ranks:
            name = f"{chem_level}_x_{org_rank}"
            counts = data.count_table(tokens, organisms, chem_level, org_rank)
            chem_totals, org_totals = data.totals(tokens, organisms, chem_level, org_rank)

            chem_weights = {
                chem_paths[chem_level][label]: n for label, n in chem_totals.iter_rows()
            }
            org_weights = {tuple(row[:-1]): row[-1] for row in org_totals.iter_rows()}
            rows = select_leaves(chem_weights, args.max_rows)
            cols = select_leaves(org_weights, args.max_cols)

            table = args.out / "tables" / f"{name}.csv"
            table.parent.mkdir(parents=True, exist_ok=True)
            counts.sort("molecules", descending=True).write_csv(table)

            shown = []
            if len(rows) < len(chem_weights):
                shown.append(f"top {len(rows)} of {len(chem_weights)} {chem_level} labels")
            if len(cols) < len(org_weights):
                shown.append(f"top {len(cols)} of {len(org_weights)} {org_rank} taxa")
            title = (
                f"LOTUS {VERSION}: NPClassifier {chem_level} × organism {org_rank}"
                + (f"\n({'; '.join(shown)}, by unique molecules)" if shown else "")
            )
            observed = count_matrix(counts, rows, cols)
            expected = expected_matrix(counts, rows, cols)
            for scale in SCALES:
                colors = color_scale(scale, observed, expected)
                out_stem = args.out / "heatmaps" / scale / name
                plot_heatmap(colors, rows, cols, chem_level, org_rank, title, out_stem)
            print(f"{name}: {len(rows)}×{len(cols)} shown, {counts.height} non-zero cells in table")


def tanglegrams(args: argparse.Namespace, metadata: pl.DataFrame, organisms: pl.DataFrame) -> None:
    class_tokens = data.chemical_tokens(metadata, "class")
    agreements = []
    for rank in args.tangle_ranks:
        paths, profiles = chemical_profiles(
            class_tokens, organisms, rank, args.tangle_min_molecules, args.tangle_max_taxa
        )
        taxonomy, chemistry, agreement = compare(paths, profiles, rank)
        title = (
            f"LOTUS {VERSION}: organism {rank} — taxonomy vs NPClassifier class profiles\n"
            f"(taxa with ≥ {args.tangle_min_molecules} class-labelled molecules, "
            f"at most the {args.tangle_max_taxa} richest)"
        )
        plot_tanglegram(paths, taxonomy, chemistry, agreement, title, args.out / "tanglegrams" / rank)
        agreements.append(agreement)
        print(f"tanglegram {rank}: {agreement}")

    with (args.out / "tanglegrams" / "agreement.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(agreements[0])])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(a) for a in agreements)


def circular_tree(args: argparse.Namespace, metadata: pl.DataFrame, organisms: pl.DataFrame) -> None:
    tokens = data.chemical_tokens(metadata, "pathway")
    paths, weights, shares = genus_profiles(tokens, organisms, args.circular_min_molecules)
    title = (
        f"LOTUS {VERSION}: {len(paths)} genera with ≥ {args.circular_min_molecules} unique molecules — "
        "taxonomy and NPClassifier pathway composition"
    )
    plot_circular(paths, weights, shares, title, args.out / "circular" / "genus_pathways")
    print(f"circular tree: {len(paths)} genera")
