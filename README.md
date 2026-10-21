# Screen selection and coupling-partner evidence

This package contains scientific analysis code, cleaned reaction tables and
condition and compound catalogues for reproducing the screen-selection analysis.

## Contents

- `cleaned_corpus/data/`: 20 reaction CSVs, 410,587 records, including 406,909
  records in the two-substrate primary scope.
- `cleaned_corpus/manifest.json`: CSV hashes, record counts and input checks.
- `code/scripts/`: scientific analysis and the portable runner.
- `code/data/catalogues/`: reaction-condition and compound workbooks.
- `code/data/metadata/EXCLUDE_wells.csv`: excluded measurement identities for
  validation; the 396 excluded measurements are absent from the reaction CSVs.
- `code/requirements.txt`: pinned Python dependencies.
- `reference_outputs.json`: hashes of numerical scientific outputs.
- `release_manifest.json`: hashes of every packaged file.

Catalogue operational notes are faithfully translated to English at explicitly
reviewed cells. Every numerical value, chemical identity and formula is unchanged;
the notes do not change record inclusion. Display fonts and application metadata
are normalized, and unused saved strings and original save paths are removed.

## Reproduce

Use Python 3.13 in a virtual environment. From this directory:

```sh
python -m pip install -r code/requirements.txt
python code/scripts/run_release.py --out ../screen_selection_reproduced --stage primary --verify-reference
```

The output directory must be empty and outside this package. `--stage ingest`
recomputes input classification, qualification, recipe comparisons and measurement
loss without the primary random controls. `--stage primary` additionally computes
the collective witnesses and complete-comparison-block budget schedules under
`science_extensions/`, after completing the core scientific analysis. Verify the
packaged files alone with:

```sh
python code/scripts/run_release.py --out ../unused --verify-only
```

## Endpoints and inclusion

The runner verifies the cleaned tables and restores `raw_yield` for independent
endpoint interpretation, retaining `source_file` and `source_row`. A numeric raw
yield is retained. A blank yield with DESI call 0 is observed zero; blanks with
DESI call 1 or no classifier flag remain missing. No endpoint is imputed. Recorded
SMILES, including stereochemical notation, are preserved.

Excluded measurement identities are checked before endpoint handling. The cleaned
tables already omit these 396 measurements, so replay records zero new removals.
The original-corpus record count is the cleaned record count plus 396.

Qualification Q uses numerical, unique reaction cells and excludes Finkelstein.
Substrate-pair condition menus are grouped at Jaccard similarity 0.90. A candidate
screen has at least 50 substrate pairs and 12 conditions; its dense core keeps rows
and columns with at least 70% available endpoints and must meet the same size
limits. First-substrate structures must parse. Secondary repeated QC conditions
are removed, and overlapping condition campaigns are resolved by substrate count.
All pair membership and evidence targets are recomputed from the cleaned inputs.

The two screen-construction families use numerical-endpoint condition menus and
attempted-condition menus. Each spans 180 settings: minimum substrate counts
20/30/50/75/100, minimum condition counts 6/12/24, completeness 50/70/90%, with
overlap suppression and repeated-QC-condition removal each on or off.

Valid numerical cells outside Q form the addition pool. Connected completion
(R1-6, R1-12, R1-24) retains whole outside pairs sharing at least 6, 12 or 24
measured conditions with a qualified screen. Four-cell completion (R2) adds pool
records from two-recipe comparisons when another substrate pair measured under
both recipes in Q shares a coupling partner in the same role. R3 retains the
two-family rule union. These rules read identities and measurement availability,
without reading yields. Preference contradictions use a 5 percentage-point margin.
Exact restoration uses independently checked mixed-integer optimization and
enumeration of minimum-size families.

Equal-budget controls sample records (R4) or whole pair units with a final partial
unit (R4p). Each uses 100 draws with seeds 20261018 plus draw index 0 through 99.
Molecule-level Poisson bootstrap intervals use 2,000 replicates and seed 20261018.
The qualification provenance records seed 20261016. Hash verification compares
the numerical scientific outputs; ingestion provenance differs because the
released inputs have already been cleaned. Supporting analyses are available as
individual stages of `code/scripts/m2/run_analysis.py`.

## Collective witnesses and budget schedules

Collective witnesses are selected separately for acylation, Heck and Ullmann
coupling. Eligible recipe contrasts require at least two omitted comparisons.
Within each class, selection first minimizes the omitted-comparison count, then
cycle length, then maximizes the smallest unrounded absolute yield difference
among the witness comparisons. Recipe-contrast and full source-record identities
break remaining ties. All simple directed cycles on the resulting minimum-cost,
shortest-length boundary are eligible, including cycles longer than four edges.
This maximin rule describes selected chemical examples; it does not alter the
full collection of evidence targets.

Budget schedules order complete comparison blocks using identities alone. Each
block combines one qualified-core comparison and one non-qualified comparison
under the same two recipes, sharing exactly one substrate partner. The four
endpoint records complete the block; only valid non-core records cost additions.
Each draw randomly permutes the distinct block inventory and takes the longest
complete prefix whose union of added records fits the nominal budget. Reused
records are counted once and individual blocks are never truncated. This order
does not read yields, preference directions or graph outcomes.

The nominal budgets are 0, 5,000, 10,000, ..., 55,000 and 57,149 added records.
Each uses 100 schedules with seeds 20261018 through 20261117. Random-record and
random-pair controls use the actual deduplicated added-record count of each draw.
Evaluation checks every retained comparison in each full target graph, rather
than only a selected witness. The reported 2.5th and 97.5th percentiles describe
variation across schedules; they are not confidence intervals.

The deterministic reference files are `science_extensions/Figure_3a.csv`,
`collective_examples.json`, `collective_examples.csv`,
`collective_candidate_audit.csv`, `block_dual_identities.parquet`,
`complete_blocks.parquet`, `Figure_4_draws.csv`, `block_schedule_audit.csv`,
`Figure_4_summary.csv` and `budget_curve_metadata.json`, all in that directory.
They contain numerical results and scientific identities; catalogue operational
note translations do not enter witness selection, scheduling or these outputs.
