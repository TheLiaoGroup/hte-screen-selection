"""Rebuild cells and qualification after measurement-ID deletion and endpoint handling.

Classifier-negative blank yields are observed zero; other blanks remain missing.
The canonical variant keys molecules by canonical isomeric SMILES.
"""
import argparse
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
from m2common import RAW, SCOPE, apply_cleanup, assert_clean, data_dir
from release_inputs import read_release_table
RULES = dict(min_substrates=50, min_conditions=12, completeness=0.70,
             signature_jaccard=0.90, seed=20261016)
TWINS = {'Hiyama': [('C43', 'C46')], 'Suzuki': [('C74', 'C77')],
         'Finkelstein': [('C34', 'C37')]}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def signatures(d):
    sig = d.groupby('subpair')['condition_id'].apply(lambda s: frozenset(s.unique()))
    counts = Counter(sig.values)
    ordered = [s for s, _ in counts.most_common()]
    used, groups = set(), []
    for i, a in enumerate(ordered):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(ordered)):
            if j not in used and len(a & ordered[j]) / max(len(a | ordered[j]), 1) >= 0.90:
                group.append(j)
                used.add(j)
        keep = {ordered[k] for k in group}
        groups.append(([sp for sp, s in sig.items() if s in keep],
                       sorted(set().union(*keep))))
    groups.sort(key=lambda x: -len(x[0]))
    return groups


def canonical(s, cache={}):
    if s not in cache:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        cache[s] = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m is not None else s
    return cache[s]


def prepare_measurements(frame, *, exclusions=None):
    u, cleanup = apply_cleanup(frame, reaction_column='rxn_class', exclusions=exclusions)
    u = u.reset_index(drop=True)
    u['raw_yield'] = u['yield']
    u['yield'] = pd.to_numeric(u['yield'], errors='coerce').astype(float)
    u['yield_clf'] = pd.to_numeric(u['yield_clf'], errors='coerce').astype(float)
    blank = u['raw_yield'].isna() | u['raw_yield'].str.strip().eq('').fillna(False)
    assert not (u['yield'].isna() & ~blank).any(), 'Unparseable nonblank yield'
    u['measurement_status'] = np.select(
        [~blank, u['yield_clf'].eq(0), u['yield_clf'].eq(1)],
        ['quantified', 'zero_by_rule', 'detected_not_quantified'], 'no_screen_record')
    u.loc[blank & u['yield_clf'].eq(0), 'yield'] = 0.0
    assert u.loc[blank & ~u['yield_clf'].eq(0), 'yield'].isna().all()
    return u, cleanup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['clfzero', 'canonical_clfzero'], default='clfzero')
    variant = ap.parse_args().variant
    DATA = data_dir(variant)
    frames, inventory, provenance = [], [], []
    for p in sorted(RAW.iterdir()):
        if p.suffix.lower() not in ('.csv', '.xlsx') or p.name.startswith('0_'):
            continue
        release_frame = read_release_table(p)
        d = (release_frame if release_frame is not None else
             pd.read_csv(p, low_memory=False) if p.suffix.lower() == '.csv' else pd.read_excel(p))
        assert {'condition_id', 'sub_1_smiles', 'yield', 'yield_clf', 'name0'} <= set(d), p
        cls = re.sub(r'[_ ]?\d{6}$', '', p.stem).strip('_ ')
        if cls != 'Finkelstein':
            assert 'sub_2_smiles' in d, p
        elif 'sub_2_smiles' not in d:
            d['sub_2_smiles'] = pd.NA
        inventory.append(dict(file=p.name, rxn_class=cls, rows=len(d), columns=len(d.columns),
                              numeric_yields=pd.to_numeric(d['yield'], errors='coerce').notna().sum()))
        provenance.append(dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p)))
        d = d.drop(columns=['idx'], errors='ignore')
        if release_frame is not None:
            original_file = d.pop('source_file')
            original_row = d.pop('source_row')
        for c in d:
            d[c] = d[c].astype('string')
        if release_frame is None:
            d['source_file'] = p.name
        else:
            d['source_file'] = original_file
        d['rxn_class'] = cls
        d['source_row'] = (np.arange(2, len(d) + 2) if release_frame is None else
                           pd.to_numeric(original_row, errors='raise').astype(np.int64))
        frames.append(d)
        print(f'Loaded {p.name}: {len(d):,}', flush=True)
    assert len(frames) == 20, f'Expected 20 reaction tables, found {len(frames)}'
    u = pd.concat(frames, ignore_index=True)
    input_records = len(u)
    u, cleanup = prepare_measurements(u)
    names = u['name0'].fillna('')
    u['plate_id'] = names.str.extract(r'^(.*?-PL\d+)(?:-|$)', expand=False).fillna(
        names.str.replace(r'-LC\d+.*$', '', regex=True))
    assert_clean(u, reaction_column='rxn_class')
    u['record_id'] = u['source_file'] + ':' + u['source_row'].astype(str)
    u['recorded_sub_1_smiles'], u['recorded_sub_2_smiles'] = u['sub_1_smiles'], u['sub_2_smiles']
    if variant == 'canonical_clfzero':
        for c in ('sub_1_smiles', 'sub_2_smiles'):
            u[c] = u[c].map(lambda s: canonical(s) if isinstance(s, str) else s).astype('string')
    u['subpair'] = u['sub_1_smiles'].fillna('nan') + ' >> ' + u['sub_2_smiles'].fillna('nan')
    u['numeric'] = np.isfinite(u['yield'])
    u['qc_pass'] = u['check_passed'].str.lower().isin(['true', '1', '1.0'])
    u['yield_out_of_range'] = u['numeric'] & ~u['yield'].between(0, 1)
    names = u['name0'].fillna('')
    u['source_campaign'] = names.str.replace(r'-PL\d+.*$', '', regex=True)
    u['date_token'] = names.str.extract(r'^(DP\d{6})', expand=False).fillna('unparsed')
    key = ['rxn_class', 'subpair', 'condition_id']
    all_counts = u.groupby(key)['record_id'].transform('size')
    numeric_counts = u.loc[u['numeric']].groupby(key)['record_id'].transform('size')
    u['duplicate_cell'] = False
    u.loc[numeric_counts.index, 'duplicate_cell'] = numeric_counts.gt(1)
    u['legacy_duplicate_export'] = all_counts.gt(1) & u['numeric']
    u['qc_twin_secondary'] = False
    for cls, twins in TWINS.items():
        for _, b in twins:
            u.loc[u['rxn_class'].eq(cls) & u['condition_id'].eq(b), 'qc_twin_secondary'] = True
    eligible = u['numeric'] & ~u['duplicate_cell'] & u['rxn_class'].ne('Finkelstein')
    u['screen'], u['block_id'] = '', ''
    u['outside_reason'] = 'not_numeric_or_excluded_class_or_duplicate'
    u.loc[eligible, 'outside_reason'] = 'unclassified'
    blocks, candidates = [], {}
    for cls, d in u.loc[eligible].groupby('rxn_class'):
        for bi, (subs, conds) in enumerate(signatures(d), 1):
            bid = f'{cls}|{bi}'
            m = u['rxn_class'].eq(cls) & u['subpair'].isin(subs)
            u.loc[m, 'block_id'] = bid
            p = d.loc[d['subpair'].isin(subs)].pivot_table(
                index='subpair', columns='condition_id', values='yield', aggfunc='mean')
            core = p.loc[p.notna().mean(axis=1) >= .70, p.notna().mean(axis=0) >= .70]
            row = dict(block_id=bid, rxn_class=cls, n_substrates=len(subs), n_conditions=len(conds),
                       n_numeric=int(p.notna().sum().sum()), completeness=float(p.notna().mean().mean()),
                       n_core_substrates=core.shape[0], n_core_conditions=core.shape[1],
                       qualifies=False, reason='')
            if len(subs) < 50 or len(conds) < 12:
                reason = 'too_few_substrates' if len(subs) < 50 else 'too_few_conditions'
            elif core.shape[0] < 50 or core.shape[1] < 12:
                reason = 'dense_core_below_threshold'
            else:
                valid = [sp for sp in core.index if Chem.MolFromSmiles(sp.split(' >> ')[0]) is not None]
                if len(valid) < 50:
                    reason = 'valid_first_substrate_below_threshold'
                else:
                    cs = list(core.columns)
                    for a, b in TWINS.get(cls, []):
                        if a in cs and b in cs:
                            cs.remove(b)
                    candidates[bid] = dict(cls=cls, subs=valid, conds=cs)
                    reason = 'candidate'
            row['reason'] = reason
            u.loc[m & eligible, 'outside_reason'] = reason
            blocks.append(row)
    drop = set()
    keys = sorted(candidates)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = candidates[a], candidates[b]
            if sa['cls'] != sb['cls'] or a in drop or b in drop:
                continue
            if set(sa['conds']) & set(sb['conds']):
                drop.add(a if len(sa['subs']) < len(sb['subs']) else b)
    for b in blocks:
        bid = b['block_id']
        if bid in candidates:
            b['qualifies'] = bid not in drop
            b['reason'] = 'overlapping_condition_campaign_removed' if bid in drop else 'qualified'
            m = u['block_id'].eq(bid) & eligible
            u.loc[m, 'outside_reason'] = b['reason'] if bid in drop else 'outside_dense_core_or_qc_twin'
    for bid, s in candidates.items():
        if bid not in drop:
            m = u['rxn_class'].eq(s['cls']) & u['subpair'].isin(s['subs']) & u['condition_id'].isin(s['conds'])
            u.loc[m, 'screen'] = bid
            u.loc[m & eligible, 'outside_reason'] = ''
    u['qualified_row'] = u['screen'].ne('') & eligible
    # Reproduce the legacy outside label independently of subsequent analysis choices.
    u['legacy_outside_numeric'] = (u['screen'].eq('') & u['numeric'] &
                                    u['rxn_class'].ne('Finkelstein') & ~u['legacy_duplicate_export'])
    valid = eligible & ~u['qc_twin_secondary']
    u['analysis_valid'] = valid
    pair_has = u.loc[valid].groupby(['rxn_class', 'subpair'])['qualified_row'].agg(['any', 'all'])
    pair_has['pair_group'] = np.where(~pair_has['any'], 'outside_only',
                                    np.where(pair_has['all'], 'qualified_only', 'mixed'))
    u = u.merge(pair_has[['pair_group']], how='left', on=['rxn_class', 'subpair'], validate='many_to_one')
    pair = u.loc[valid].groupby(['rxn_class', 'subpair', 'pair_group']).agg(
        n_conditions=('condition_id', 'nunique'), n_records=('record_id', 'size'),
        best_yield=('yield', 'max'), mean_yield=('yield', 'mean'),
        n_campaigns=('source_campaign', 'nunique'), n_plates=('plate_id', 'nunique'),
        n_qualified_records=('qualified_row', 'sum'), qc_pass_all=('qc_pass', 'all'),
        sub_1_smiles=('sub_1_smiles', 'first'), sub_2_smiles=('sub_2_smiles', 'first'),
        source_campaign=('source_campaign', 'first'), block_id=('block_id', 'first')).reset_index()
    assert u['record_id'].is_unique
    assert input_records == sum(x['rows'] for x in inventory)
    assert len(u) > 0 and int(valid.sum()) > 0 and int(u.qualified_row.sum()) > 0
    missing = u.measurement_status.isin(['detected_not_quantified', 'no_screen_record'])
    assert u.loc[missing, 'yield'].isna().all() and not u.loc[missing, 'analysis_valid'].any()
    assert u.loc[u.measurement_status.eq('zero_by_rule'), 'yield'].eq(0).all()
    assert not u.loc[valid].duplicated(key).any()
    assert u.loc[u['qualified_row'], 'outside_reason'].eq('').all()
    for p in RAW.glob('0_*.xlsx'):
        provenance.append(dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p)))
    pd.DataFrame(inventory).to_csv(DATA / 'source_inventory.csv', index=False)
    pd.DataFrame(cleanup['by_plate']).to_csv(DATA / 'cleanup_by_plate.csv', index=False)
    pd.DataFrame(blocks).to_csv(DATA / 'block_inventory.csv', index=False)
    pair.to_csv(DATA / 'pair_inventory.csv', index=False)
    u.to_parquet(DATA / 'observations.parquet', index=False)
    u.groupby(['rxn_class', 'pair_group'], dropna=False).agg(
        records=('record_id', 'size'), numeric=('numeric', 'sum'), qualified=('qualified_row', 'sum'),
        valid=('analysis_valid', 'sum')).to_csv(DATA / 'qualification_inventory.csv')
    legacy = u.loc[u['legacy_outside_numeric']].groupby('rxn_class').agg(
        records=('record_id', 'size'), pairs=('subpair', 'nunique'))
    legacy.to_csv(DATA / 'legacy_outside_reconstruction.csv')
    manifest = dict( python=sys.version, rules=RULES, qc_twins=TWINS,
                    reaction_files=len(inventory), input_records=input_records, scope=SCOPE,
                    excluded_plate_ids=cleanup['excluded_plate_ids'],
                    excluded_records=cleanup['excluded_records'], cleanup=cleanup,
                    raw_records=len(u), numeric_records=int(u['numeric'].sum()),
                    measurement_status_counts=u.measurement_status.value_counts().to_dict(),
                    qualified_screens=len(candidates)-len(drop),
                    legacy_outside_records=int(u['legacy_outside_numeric'].sum()),
                    legacy_outside_class_pairs=int(legacy['pairs'].sum()),
                    variant=variant,
                    source_files=provenance, code_sha256=digest(Path(__file__)))
    (DATA / 'reconstruction_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_files','qc_twins')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
