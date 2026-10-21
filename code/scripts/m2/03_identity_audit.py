"""Does keying molecules by canonical SMILES change any identity? (PLAN section 3, item 4)."""
import argparse
import json
import pandas as pd
from rdkit import Chem, RDLogger
from m2common import data_dir, out_dir
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument('--variant', choices=['clfzero'], default='clfzero')
variant = ap.parse_args().variant
canonical_variant = 'canonical' if variant == 'raw' else 'canonical_clfzero'
u = pd.read_parquet(data_dir(variant) / 'observations.parquet')
c = pd.read_parquet(data_dir(canonical_variant) / 'observations.parquet', columns=['record_id', 'subpair', 'analysis_valid', 'qualified_row', 'pair_group', 'screen'])
assert (u.record_id.values == c.record_id.values).all()
raw = pd.unique(pd.concat([u.sub_1_smiles, u.sub_2_smiles]).dropna())
rows = []
for s in raw:
    m = Chem.MolFromSmiles(s)
    rows.append(dict(recorded=s, parses=m is not None,
                     canonical=Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m is not None else s))
t = pd.DataFrame(rows)
collide = t.groupby('canonical').recorded.nunique()
val = u.loc[u.analysis_valid]
pairs = set(zip(val.rxn_class, val.sub_1_smiles, val.sub_2_smiles))
swapped = sorted((k, a, b) for k, a, b in pairs if (k, b, a) in pairs and a < b)
res = dict(recorded_strings=len(t), unparseable=int((~t.parses).sum()),
           canonical_identities=int(t.canonical.nunique()),
           canonical_collisions=int((collide > 1).sum()),
           strings_changed_by_canonicalization=int((t.recorded != t.canonical).sum()),
           records_with_changed_membership=int(((u.analysis_valid.values != c.analysis_valid.values) |
                                               (u.qualified_row.values != c.qualified_row.values) |
                                               (u.screen.values != c.screen.values) |
                                               (u.pair_group.astype(str).values != c.pair_group.astype(str).values)).sum()),
           role_swapped_pair_twins_within_class=len(swapped))
res.update(variant=variant, canonical_variant=canonical_variant,
           measured_cells_before=int(u.analysis_valid.sum()), measured_cells_after=int(c.analysis_valid.sum()),
           qualified_cells_before=int(u.qualified_row.sum()), qualified_cells_after=int(c.qualified_row.sum()))
before = out_dir(variant) / 'partner_score_edges.csv'
after = out_dir(canonical_variant) / 'partner_score_edges.csv'
if before.exists() and after.exists():
    original_edges, canonical_edges = pd.read_csv(before), pd.read_csv(after)
    original_new = set(original_edges.loc[original_edges.new_contradiction, 'edge_id'])
    canonical_new = set(canonical_edges.loc[canonical_edges.new_contradiction, 'edge_id'])
    res.update(new_counterexamples_before=len(original_new), new_counterexamples_after=len(canonical_new),
               new_counterexample_count_change_fraction=abs(len(canonical_new) - len(original_new)) / max(len(original_new), 1),
               changed_new_counterexample_ids=len(original_new ^ canonical_new))
print(json.dumps(res, indent=1))
(out_dir(variant) / 'identity_audit.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
t.loc[t.recorded != t.canonical].to_csv(out_dir(variant) / 'identity_audit_changed_strings.csv', index=False)
