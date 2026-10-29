"""Download the LOTUS frozen metadata export and tidy it into count tables.

Molecules are counted by 2D structure: the first InChIKey block (14 characters,
the connectivity hash), so stereoisomers collapse into one molecule.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import polars as pl

METADATA_FILE = "260413_frozen_metadata.csv.gz"
METADATA_MD5 = "b17048b3b77daae9ab1e480b6591aabd"
METADATA_URL = f"https://zenodo.org/records/19360665/files/{METADATA_FILE}?download=1"

MOLECULE = "molecule_2d"
"""Column holding the 2D molecule key (InChIKey connectivity block)."""

NPC_SEPARATOR = " $ "
"""NPClassifier joins multiple predicted labels with this separator."""

CHEM_LEVELS: dict[str, str] = {
    "pathway": "structure_taxonomy_npclassifier_01pathway",
    "superclass": "structure_taxonomy_npclassifier_02superclass",
    "class": "structure_taxonomy_npclassifier_03class",
}

ORG_RANKS: dict[str, str] = {
    "domain": "organism_taxonomy_01domain",
    "kingdom": "organism_taxonomy_02kingdom",
    "phylum": "organism_taxonomy_03phylum",
    "class": "organism_taxonomy_04class",
    "order": "organism_taxonomy_05order",
    "family": "organism_taxonomy_06family",
    "genus": "organism_taxonomy_08genus",
}

OTHER_EUKARYOTES = "Other eukaryotes"


def fetch_metadata(data_dir: Path) -> Path:
    """Return the local metadata file, downloading it from Zenodo if absent."""
    path = data_dir / METADATA_FILE
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(METADATA_URL, tmp)
        tmp.rename(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != METADATA_MD5:
        raise ValueError(f"{path}: md5 {digest} != expected {METADATA_MD5}")
    return path


def load_metadata(path: Path) -> pl.DataFrame:
    columns = ["structure_inchikey", *CHEM_LEVELS.values(), *ORG_RANKS.values()]
    return pl.read_csv(path, columns=columns, infer_schema_length=0).with_columns(
        pl.col("structure_inchikey").str.slice(0, 14).alias(MOLECULE)
    )


def chem_column(level: str) -> str:
    """Name of the label column in `chemical_tokens` output (avoids clashing with ranks)."""
    return f"npclassifier_{level}"


def chemical_tokens(df: pl.DataFrame, level: str) -> pl.DataFrame:
    """Unique (molecule, NPClassifier label) pairs at `level`.

    Multi-label predictions are exploded: a molecule counts once under each
    label it carries (across all of its stereoisomers). Unclassified molecules
    are dropped.
    """
    column = chem_column(level)
    return (
        df.select(MOLECULE, pl.col(CHEM_LEVELS[level]).str.split(NPC_SEPARATOR).alias(column))
        .drop_nulls(column)
        .explode(column)
        .unique()
    )


def _parent_map(df: pl.DataFrame, child: str, parent: str) -> dict[str, str]:
    """Most consistent NPClassifier parent of every `child` label.

    The metadata stores the three levels independently, so multi-label rows do
    not say which parent goes with which child. Prefer the parent seen most
    often on single-label rows, then the most frequent co-occurring parent.
    """
    pairs = (
        df.select(
            pl.col(CHEM_LEVELS[child]).str.split(NPC_SEPARATOR).alias("child"),
            pl.col(CHEM_LEVELS[parent]).str.split(NPC_SEPARATOR).alias("parent"),
        )
        .drop_nulls()
        .with_columns(
            single=(pl.col("child").list.len() == 1) & (pl.col("parent").list.len() == 1)
        )
        .explode("child")
        .explode("parent")
        .group_by("child", "parent")
        .agg(n_single=pl.col("single").sum(), n=pl.len())
        .sort(["child", "n_single", "n", "parent"], descending=[False, True, True, False])
        .unique("child", keep="first")
    )
    return dict(zip(pairs["child"], pairs["parent"], strict=True))


def chemical_paths(df: pl.DataFrame) -> dict[str, dict[str, tuple[str, ...]]]:
    """Map every label at every NPClassifier level to its pathway→label path."""
    superclass_parent = _parent_map(df, "superclass", "pathway")
    class_parent = _parent_map(df, "class", "superclass")
    pathways = chemical_tokens(df, "pathway")[chem_column("pathway")].unique()
    return {
        "pathway": {p: (p,) for p in pathways},
        "superclass": {s: (p, s) for s, p in superclass_parent.items()},
        "class": {
            c: (superclass_parent[s], s, c) for c, s in class_parent.items()
        },
    }


def organism_table(df: pl.DataFrame) -> pl.DataFrame:
    """Unique (molecule, domain…genus) rows with gaps in the taxonomy filled.

    Prokaryotes have no kingdom in the Open Tree of Life export, so their domain
    stands in; eukaryotes without kingdom become "Other eukaryotes". Any other
    missing rank becomes "<nearest named ancestor> [no <rank>]" so every
    organism keeps a complete path. Organisms without domain are dropped.
    """
    domain = pl.col(ORG_RANKS["domain"])
    kingdom = pl.coalesce(
        pl.col(ORG_RANKS["kingdom"]),
        pl.when(domain == "Eukaryota").then(pl.lit(OTHER_EUKARYOTES)).otherwise(domain),
    )
    filled = [domain.alias("domain"), kingdom.alias("kingdom")]
    named = kingdom
    for rank in list(ORG_RANKS)[2:]:
        raw = pl.col(ORG_RANKS[rank])
        filled.append(pl.coalesce(raw, pl.format("{} [no {}]", named, pl.lit(rank))).alias(rank))
        named = pl.coalesce(raw, named)
    return df.filter(domain.is_not_null()).select(MOLECULE, *filled).unique()


def count_table(
    chem_tokens: pl.DataFrame, organisms: pl.DataFrame, chem_level: str, org_rank: str
) -> pl.DataFrame:
    """Unique 2D molecules per (chemical label, organism taxon) pair."""
    ranks = list(ORG_RANKS)[: list(ORG_RANKS).index(org_rank) + 1]
    return (
        organisms.select(MOLECULE, *ranks)
        .unique()
        .join(chem_tokens, on=MOLECULE)
        .group_by(chem_column(chem_level), *ranks)
        .agg(pl.len().alias("molecules"))
    )


def totals(chem_tokens: pl.DataFrame, organisms: pl.DataFrame, chem_level: str, org_rank: str):
    """Unique 2D molecules per chemical label and per organism path (marginals)."""
    ranks = list(ORG_RANKS)[: list(ORG_RANKS).index(org_rank) + 1]
    placed = organisms.select(MOLECULE, *ranks).unique()
    chem = (
        chem_tokens.join(placed.select(MOLECULE).unique(), on=MOLECULE)
        .group_by(chem_column(chem_level))
        .agg(pl.len().alias("molecules"))
    )
    org = (
        placed.join(chem_tokens.select(MOLECULE).unique(), on=MOLECULE)
        .group_by(*ranks)
        .agg(pl.len().alias("molecules"))
    )
    return chem, org
