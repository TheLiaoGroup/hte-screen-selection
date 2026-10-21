"""Historical failure of molecules removed by qualification.

Failure measures use yield values of valid records only; whether a yield is present (the input of
qualification completeness) enters only a descriptive column.
"""
import argparse
import json

import numpy as np
import pandas as pd
from m2common import data_dir, out_dir

SEED = 20261018
N_BOOT = 2000
MEASURES = ['a_all_clf_zero', 'a2_all_yield_zero', 'b_mean_yield']


def molecule_tables(u):
    v = u.loc[u.analysis_valid].copy()
    pair = v.groupby(['rxn_class', 'subpair']).agg(clf0=('yield_clf', lambda s: bool((s == 0).all())),
                                                  y0=('yield', lambda s: bool((s == 0).all())),
                                                  ysum=('yield', 'sum'), n=('yield', 'size'),
                                                  s1=('sub_1_smiles', 'first'), s2=('sub_2_smiles', 'first'))
    pair = pair.reset_index()
    long = pd.concat([pair.assign(mol=pair.s1, role=1), pair.assign(mol=pair.s2, role=2)], ignore_index=True)
    rec = pd.concat([v[['rxn_class', 'sub_1_smiles', 'qualified_row']].rename(columns={'sub_1_smiles': 'mol'}),
                     v[['rxn_class', 'sub_2_smiles', 'qualified_row']].rename(columns={'sub_2_smiles': 'mol'})])
    q = rec.groupby(['rxn_class', 'mol']).qualified_row.agg(['all', 'any'])
    group = pd.Series(np.where(q['all'], 'Q_only', np.where(~q['any'], 'outside', 'both')), index=q.index, name='group')
    att = u.loc[u.rxn_class.ne('Finkelstein') & ~u.duplicate_cell]
    arec = pd.concat([att[['rxn_class', 'sub_1_smiles', 'numeric', 'yield_clf']].rename(columns={'sub_1_smiles': 'mol'}),
                      att[['rxn_class', 'sub_2_smiles', 'numeric', 'yield_clf']].rename(columns={'sub_2_smiles': 'mol'})])
    arec['blank'] = ~arec.numeric
    arec['blank_clf0'] = arec.blank & arec.yield_clf.eq(0)
    desc = arec.groupby(['rxn_class', 'mol']).agg(attempted=('blank', 'size'), blank=('blank', 'sum'), blank_clf0=('blank_clf0', 'sum'))
    return long, group, desc


def measures(long):
    g = long.groupby(['rxn_class', 'mol'])
    m = pd.DataFrame({'a_all_clf_zero': g.clf0.mean(), 'a2_all_yield_zero': g.y0.mean(),
                      'b_mean_yield': g.ysum.sum() / g.n.sum(), 'n_pairs': g.size(),
                      'roles': g.role.nunique()})
    return m


def pooled(mol, col, a, b='Q_only'):
    num = den = 0.0
    for cls, x in mol.groupby(level=0):
        xa, xb = x.loc[x.group.eq(a), col], x.loc[x.group.eq(b), col]
        if len(xa) and len(xb):
            w = len(xa) * len(xb) / (len(xa) + len(xb))
            num += w * (xa.mean() - xb.mean()); den += w
    return num / den if den else np.nan


def boot_pooled(mol, col, a, rng, b='Q_only'):
    """Molecules resampled with replacement within class; pooled difference recomputed per replicate."""
    num = np.zeros(N_BOOT); den = 0.0
    for cls, x in mol.groupby(level=0):
        g, v = x.group.to_numpy(), x[col].to_numpy(float)
        if not ((g == a).any() and (g == b).any()):
            continue
        idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
        gs, vs = g[idx], v[idx]
        na, nb = (gs == a).sum(1), (gs == b).sum(1)
        ma = np.where(gs == a, vs, 0).sum(1) / np.maximum(na, 1)
        mb = np.where(gs == b, vs, 0).sum(1) / np.maximum(nb, 1)
        ok = (na > 0) & (nb > 0)
        w = np.where(ok, na * nb / np.maximum(na + nb, 1), 0.0)
        num = num + w * np.where(ok, ma - mb, 0.0)
        den = den + w
    return np.nanpercentile(num / np.where(den > 0, den, np.nan), [2.5, 97.5]).tolist()


def leave_pair_out(long):
    """Per (pair, molecule) row: the molecule's measures without that pair."""
    g = long.groupby(['rxn_class', 'mol'])
    tot = pd.DataFrame({'np': g.size(), 'nc': g.clf0.sum(), 'ny': g.y0.sum(), 'ys': g.ysum.sum(), 'yn': g.n.sum()})
    x = long.join(tot, on=['rxn_class', 'mol'])
    k = x.np - 1
    x['a_all_clf_zero'] = np.where(k > 0, (x.nc - x.clf0) / k.where(k > 0), np.nan)
    x['a2_all_yield_zero'] = np.where(k > 0, (x.ny - x.y0) / k.where(k > 0), np.nan)
    rest = x.yn - x.n
    x['b_mean_yield'] = np.where(rest > 0, (x.ys - x.ysum) / rest.where(rest > 0), np.nan)
    return x.groupby(['rxn_class', 'subpair'])[MEASURES].mean()  # mean over the pair's two molecules


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    v = ap.parse_args().variant
    out = out_dir() if v == 'canonical' else out_dir(v)  # canonical identities are the primary 4.4 unit
    u = pd.read_parquet(data_dir(v) / 'observations.parquet')
    long, group, desc = molecule_tables(u)
    mol = measures(long).join(group).join(desc)
    mol['blank_fraction'] = mol.blank / mol.attempted
    mol['blank_clf0_fraction'] = mol.blank_clf0 / mol.blank.where(mol.blank > 0)
    counts = mol.groupby([mol.index.get_level_values(0), 'group']).size().unstack(fill_value=0)
    print(counts.to_string(), flush=True)
    assert counts.sum().gt(0).all()
    rng = np.random.default_rng(SEED)
    res = dict(molecules=int(len(mol)), molecules_in_both_roles=int((mol.roles > 1).sum()),
               group_counts=mol.group.value_counts().to_dict(), pooled={})
    for a in ('outside', 'both'):
        for col in MEASURES + ['blank_fraction', 'blank_clf0_fraction']:
            sub = mol.loc[mol[col].notna()]  # blank_clf0_fraction is undefined for molecules without blank attempts
            est = pooled(sub, col, a)
            lo, hi = boot_pooled(sub, col, a, rng)
            res['pooled'][f'{a}_minus_Q_only|{col}'] = dict(estimate=est, lo=lo, hi=hi)
            print(a, col, round(est, 4), [round(lo, 4), round(hi, 4)], flush=True)
    by_class = mol.groupby([mol.index.get_level_values(0), 'group'])[MEASURES + ['blank_fraction']].mean().unstack('group')
    by_class.to_csv(out / 'molecule_failure_by_class.csv')
    mol.reset_index().to_csv(out / 'molecule_failure.csv', index=False)

    # Counterexample records versus all outside-only records (leave-pair-out scores).
    lpo = leave_pair_out(long)
    edges = pd.read_csv((out_dir() if v == 'canonical' else out) / 'partner_score_edges.csv')
    new = edges.loc[edges.new_contradiction].reset_index(drop=True)
    assert v != 'canonical' or len(new) == 179
    rid = u.set_index('record_id')[['rxn_class', 'subpair', 'pair_group']]
    cert = []
    for k, ids in enumerate(new.certificate_dual_ids):
        for did in json.loads(ids):
            for r in did.split('|'):
                cls, sp, pg = rid.loc[r]
                if pg == 'outside_only':
                    cert.append((k, r, cls, sp))
    cert = pd.DataFrame(cert, columns=['contrast', 'record_id', 'rxn_class', 'subpair'])
    assert len(cert) > 0
    cs = cert.join(lpo, on=['rxn_class', 'subpair'])
    pool = u.loc[u.analysis_valid & u.pair_group.eq('outside_only'), ['rxn_class', 'subpair']].join(lpo, on=['rxn_class', 'subpair'])
    res['certificate'] = dict(certificate_records=int(cs.record_id.nunique()), pool_records=int(len(pool)))
    contrasts = cs.contrast.unique()
    for col in MEASURES:
        d = cs.drop_duplicates('record_id')[col].mean() - pool[col].mean()
        per = cs.groupby('contrast')[col].agg(['sum', 'count']).reindex(contrasts)
        pick = rng.integers(0, len(contrasts), size=(N_BOOT, len(contrasts)))
        boots = per['sum'].to_numpy()[pick].sum(1) / per['count'].to_numpy()[pick].sum(1) - pool[col].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res['certificate'][col] = dict(certificate_mean=float(cs.drop_duplicates('record_id')[col].mean()),
                                       pool_mean=float(pool[col].mean()), difference=float(d), lo=float(lo), hi=float(hi))
        print('certificate', col, round(d, 4), [round(lo, 4), round(hi, 4)], flush=True)
    (out / 'molecule_failure.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
