"""Failure measures restricted to conditions measured for both molecule
groups, condition-adjusted measures, and pair menus (length and quality)."""
import argparse
from importlib import import_module
import json

import numpy as np
import pandas as pd
from m2common import data_dir, out_dir

mf = import_module('71_molecule_failure')
SEED, N_BOOT = 20261018, 2000
BINS = [(1, 11), (12, 23), (24, 47), (48, 10 ** 6)]


def long_records(v):
    """One row per (valid record, molecule) with the molecule's group."""
    u = pd.read_parquet(data_dir(v) / 'observations.parquet')
    _, group, _ = mf.molecule_tables(u)
    r = u.loc[u.analysis_valid, ['rxn_class', 'subpair', 'condition_id', 'yield', 'yield_clf', 'sub_1_smiles', 'sub_2_smiles', 'pair_group']]
    r['dead'] = r.yield_clf.eq(0).astype(float)
    cond = r.groupby(['rxn_class', 'condition_id']).agg(cy=('yield', 'mean'), cd=('dead', 'mean'))
    r = r.join(cond, on=['rxn_class', 'condition_id'])
    r['ry'], r['rd'] = r['yield'] - r.cy, r.dead - r.cd
    rows = pd.concat([r.assign(mol=r.sub_1_smiles), r.assign(mol=r.sub_2_smiles)], ignore_index=True)
    return rows.join(group, on=['rxn_class', 'mol']), r


def molecule_measures(rows):
    common = {}
    for cls, x in rows.groupby('rxn_class'):
        a = set(x.loc[x.group.eq('outside'), 'condition_id']); b = set(x.loc[x.group.eq('Q_only'), 'condition_id'])
        common[cls] = a & b
    rows = rows.assign(in_common=[c in common[k] for k, c in zip(rows.rxn_class, rows.condition_id)])
    cm = rows.loc[rows.in_common]
    pair_dead = cm.groupby(['rxn_class', 'mol', 'subpair']).dead.min().rename('all_dead')  # all C_c records dead
    g = rows.groupby(['rxn_class', 'mol'])
    m = pd.DataFrame({'group': g.group.first(),
                      'common_a_all_clf_zero': pair_dead.groupby(level=[0, 1]).mean(),
                      'common_b_mean_yield': cm.groupby(['rxn_class', 'mol'])['yield'].mean(),
                      'adj_dead_residual': g.rd.mean(), 'adj_yield_residual': g.ry.mean()})
    return m, {k: len(v) for k, v in common.items()}


def menus(r, v):
    p = r.groupby(['rxn_class', 'subpair']).agg(n_cond=('condition_id', 'nunique'), quality=('cy', 'mean'),
                                                pair_group=('pair_group', 'first'))
    p = p.loc[p.pair_group.isin(['outside_only', 'qualified_only'])].copy()
    p['group'] = np.where(p.pair_group.eq('outside_only'), 'outside', 'Q_only')
    p['stratum'] = pd.cut(p.n_cond, [b[0] - 1 for b in BINS] + [BINS[-1][1]], labels=[f'{a}-{b}' if b < 10 ** 6 else f'>={a}' for a, b in BINS])
    rng = np.random.default_rng(SEED)
    out = {}
    for col in ('n_cond', 'quality'):
        est = mf.pooled(p, col, 'outside')
        lo, hi = mf.boot_pooled(p, col, 'outside', rng)
        out[f'outside_minus_Q_only|{col}'] = dict(estimate=est, lo=lo, hi=hi)
    strata = p.groupby(['stratum', 'group'], observed=False).agg(pairs=('quality', 'size'), mean_quality=('quality', 'mean'),
                                                                  mean_n_cond=('n_cond', 'mean')).unstack('group')
    return out, strata


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    v = ap.parse_args().variant
    out = out_dir() if v == 'canonical' else out_dir(v)
    rows, r = long_records(v)
    m, n_common = molecule_measures(rows)
    print(m.group.value_counts().to_dict(), 'common conditions per class', n_common, flush=True)
    assert m.group.value_counts().reindex(['Q_only', 'outside', 'both']).gt(0).all()
    rng = np.random.default_rng(SEED)
    res = dict(variant=v, common_conditions=n_common, pooled={})
    for col in ['common_a_all_clf_zero', 'common_b_mean_yield', 'adj_dead_residual', 'adj_yield_residual']:
        sub = m.loc[m[col].notna()]
        est = mf.pooled(sub, col, 'outside')
        lo, hi = mf.boot_pooled(sub, col, 'outside', rng)
        res['pooled'][f'outside_minus_Q_only|{col}'] = dict(estimate=est, lo=lo, hi=hi, molecules=int(len(sub)))
        print(col, round(est, 4), [round(lo, 4), round(hi, 4)], flush=True)
    menu, strata = menus(r, v)
    res['menus'] = menu
    print(menu, flush=True)
    print(strata.round(3).to_string(), flush=True)
    strata.to_csv(out / 'failure_menu_strata.csv')
    (out / 'failure_common_conditions.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
