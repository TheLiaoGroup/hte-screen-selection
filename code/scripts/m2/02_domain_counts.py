"""Domain and qualification counts (section 2, rows 1-3) for one molecule-identity variant."""
import argparse, json
import pandas as pd
from m2common import data_dir, out_dir

ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
v = ap.parse_args().variant
u = pd.read_parquet(data_dir(v) / 'observations.parquet')
val = u.loc[u.analysis_valid]
pairs = val.drop_duplicates(['rxn_class', 'subpair'])
res = dict(
    variant=v, raw_records=len(u), reaction_tables=int(u.source_file.nunique()),
    valid_cells=len(val), classes=int(val.rxn_class.nunique()),
    qualified_screens=int(u.loc[u.qualified_row, 'screen'].nunique()),
    Q_pairs=int(val.loc[val.qualified_row].drop_duplicates(['rxn_class', 'subpair']).shape[0]),
    Q_cells=int(val.qualified_row.sum()),
    outside_only_pairs=int(pairs.pair_group.eq('outside_only').sum()),
    outside_only_cells=int(val.pair_group.eq('outside_only').sum()),
    mixed_pairs=int(pairs.pair_group.eq('mixed').sum()),
    mixed_noncore_cells=int((val.pair_group.eq('mixed') & ~val.qualified_row).sum()),
    distinct_partner1=int(val.sub_1_smiles.nunique()), distinct_partner2=int(val.sub_2_smiles.nunique()),
    duplicate_cell_records=int(u.duplicate_cell.sum()))
print(json.dumps(res, indent=1))
(out_dir(v) / 'domain_counts.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
