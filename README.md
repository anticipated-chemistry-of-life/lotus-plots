# lotus-plots: plots of LOTUS chemistry versus organism taxonomy

Visualisation only: draws figures of NPClassifier chemistry versus organism taxa in the
[LOTUS frozen export (Zenodo 19360665, v11, `260413_frozen_metadata.csv.gz`)](https://zenodo.org/records/19360665)
(heatmaps, taxonomy-vs-chemistry tanglegrams, a circular genus tree). It does not curate,
predict or modify the data.

```sh
uv run lotus-plots                      # everything
uv run lotus-plots --chem-levels class --org-ranks family --max-rows 150 --max-cols 150
uv run lotus-plots --help               # thresholds for tanglegrams / circular tree
```

The metadata file is downloaded to `data/` on first run (md5-checked).

## Outputs

- `output/heatmaps/{log,linear,enrichment}/{chem}_x_{org}.png|pdf`: chemical level (`pathway`,
  `superclass`, `class`) × organism rank (`kingdom`, `phylum`, `class`, `order`, `family`).
  - `log`, `linear`: unique molecules on a log or linear (from 0) colour scale; grey = none.
  - `enrichment`: log₂((observed + 1) / (expected + 1)); red = more molecules than expected,
    blue = fewer. Expected = row total × column total / grand total over the whole table,
    so it corrects for how much each taxon and class has been studied overall. Colour limits
    are symmetric at the 99th percentile of |value| (at least 1).
- `output/tables/{chem}_x_{org}.csv`: complete long-format counts (no top-N cut), with the
  organism lineage from domain down to the rank.
- `output/tanglegrams/{phylum,class,order,family}.png|pdf`: taxonomy tree (left) facing a
  clustering of the same taxa by chemical profile (right); lines join the same taxon,
  coloured by kingdom. `agreement.csv` holds the statistics.
- `output/circular/genus_pathways.png|pdf`: genus-level taxonomy tree with a kingdom ring,
  one ring per pathway and an outer bar of molecules per genus.

## How to read the plots

All counts come from the literature compiled in LOTUS: they show what has been *reported*,
which mixes biology with research effort (plants, and terpenoids in particular, are by far the
most studied). A grey or blue cell means "rarely or never reported", not "absent".

### Heatmaps (`heatmaps/log`, `heatmaps/linear`, `heatmaps/enrichment`)

- **Rows** are NPClassifier labels, **columns** organism taxa. The coloured strip left of the
  rows gives the pathway; the strip above the columns gives the kingdom (legends top left).
- **Trees** are the classifications, not a similarity clustering. Branch height = how many
  ranks up two leaves meet; a flat bracket with many branches is a set of siblings at the same
  rank. Siblings are sorted by molecule count, so the largest groups come first.
- **A cell** = number of distinct 2D structures of that chemical label reported from that
  taxon. A molecule reported from several taxa counts in each column, and a molecule with
  several NPClassifier labels counts in each row, so cells do not add up to the number of
  unique molecules. Use `output/tables/` for exact numbers.
- **Titles** say when only the top 100 rows or columns are shown.
- **`log`**: equal colour steps are equal ratios (10 → 100 looks like 1,000 → 10,000).
  Best for comparing small and large groups in one view.
- **`linear`**: colour is proportional to the count. Shows where most of the known chemistry
  is (e.g. 47,612 terpenoids in Archaeplastida); everything much smaller looks equally dark.
- **`enrichment`**: colour is log₂ of observed / expected, where expected assumes that
  chemistry and taxonomy are unrelated given each row's and column's total. White (0) = as
  expected; +1 = twice as many as expected, +2 = four times; −1 = half. It answers "what is
  characteristic of this taxon?" independently of how well studied the taxon is: e.g.
  Bacteria are red for β-lactams, macrolides and aminosugars, Fungi blue for flavonoids and
  lignans. Cells with few molecules are pulled towards white by the +1 pseudocount; strong
  colours in small taxa (e.g. Archaea) still rest on few molecules, so check the `log` plot.

### Tanglegrams (`tanglegrams/`)

- **Left tree**: the taxonomy (x axis = number of ranks apart). **Right tree**: the same taxa
  clustered by what they make, i.e. their share of molecules in each NPClassifier class
  (x axis = Bray–Curtis dissimilarity: 0 = identical profiles, 1 = no class in common).
- **Lines** join each taxon's position in the two trees, coloured by kingdom. Branches of the
  right tree were flipped to reduce crossings, so the crossings that remain are real.
  - Parallel lines of one colour ending in one block on the right: that kingdom is also a
    chemically coherent group (e.g. Bacteria at phylum level).
  - A line crossing to another kingdom's block: that taxon's chemistry resembles distant
    taxa more than its relatives (e.g. at phylum level Streptophyta clusters with Arthropoda,
    Chordata with Chlorophyta).
- **Statistics** (title and `agreement.csv`): Mantel r is the rank correlation between the
  two distance matrices (0 = unrelated, 1 = same ordering of pairs); p = 0.001 is the
  smallest value 999 permutations can give. The cophenetic correlation compares the two
  trees directly. Both depend on which taxa are included: at order and family level the 60
  richest taxa are mostly plants plus a few fungal, animal and bacterial groups, so much of
  the agreement comes from separating kingdoms.

### Circular tree (`circular/genus_pathways`)

- **Centre to edge**: the root at the centre, then one tree level per rank (domain, kingdom,
  phylum, class, order, family) out to the genera at the edge. Genera run clockwise from the
  gap at the top. Too dense to label genera: open the PDF and zoom.
- **Rings**, from the inside out:
  1. kingdom colour;
  2. seven pathway rings (Terpenoids innermost, Carbohydrates outermost; legend top right).
     Colour intensity = share of the genus's molecules in that pathway (pale = none,
     full colour = all). Shares can add up to more than 1 because of multi-label molecules;
  3. grey bars = unique molecules per genus on a log scale (faint circles at 10, 100, 1,000).
- **Reading it**: follow one genus outward to see its chemical profile; follow one ring around
  the circle to see in which lineages a pathway concentrates, e.g. the terpenoid and
  shikimate rings are dense across the plant sector, polyketides stand out in fungi.

## Method

- **Count** = unique 2D structures: distinct InChIKey connectivity blocks (first 14
  characters), so stereoisomers count once. Not number of references.
- **Chemistry**: NPClassifier pathway → superclass → class from the metadata. Multi-label
  predictions (`A $ B`) count the molecule under each label. Unclassified molecules at a level
  are left out of that level.
- **Organisms**: Open Tree of Life lineage from the metadata (domain → kingdom → phylum →
  class → order → family → genus). Prokaryotes carry no kingdom, so `Bacteria`/`Archaea`
  stand in; eukaryotes without kingdom become `Other eukaryotes`. Other missing ranks become
  `<nearest named ancestor> [no <rank>]`. Organisms without domain are dropped.
- **Taxonomy trees** are the classifications themselves, not clustering on the counts: leaves
  sharing more ancestors join lower (height = number of ranks apart). Siblings are ordered by
  molecule count. The parent of an NPClassifier superclass/class is the one it co-occurs with
  most often (single-label rows first), since multi-label rows do not pair levels explicitly.
- **Heatmap size**: axes with more than 100 leaves keep the 100 with most molecules
  (`--max-rows`, `--max-cols`); the title states the cut.
- **Tanglegrams**: taxa with ≥ 100 class-labelled molecules, at most the 60 richest
  (`--tangle-min-molecules`, `--tangle-max-taxa`). Chemical profile = share of the taxon's
  molecules in each NPClassifier class; distance = Bray–Curtis; tree = average linkage, with
  children swapped (tree unchanged) to reduce line crossings. Agreement: Mantel test
  (Spearman between profile distance and taxonomic rank distance, 999 permutations,
  one-sided) and Pearson correlation between the cophenetic distances of the two trees.
  The taxonomic distance has only a few distinct values, so read both as rank agreement,
  not phylogenetic signal.
- **Circular tree**: genera with ≥ 50 pathway-labelled molecules (`--circular-min-molecules`).
  Pathway ring opacity = share of the genus's molecules in that pathway (shares can add up to
  more than 1 because of multi-label molecules). Outer bars: log₁₀ unique molecules. Phyla
  spanning ≥ 1.5 % of the circle are labelled.
