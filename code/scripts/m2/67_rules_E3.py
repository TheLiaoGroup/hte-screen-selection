"""Secondary endpoint E3: of partner-background groups with
opposed directions at 5 pp in the full data, how many keep a positive and a negative event on S.
Comparison enumeration and grouping are those of 30_fixed_change_backgrounds.py.
"""
from importlib import import_module

import numpy as np
import pandas as pd
from m2common import out_dir
from retention import Corpus, N_DRAWS, anchor_R4, anchor_R4p

fc = import_module('30_fixed_change_backgrounds')
e1 = import_module('61_rules_E1')
M = 5 / 100


def catalogue():
    """Strong events of every full-data opposed partner-background group: (dual_low, dual_high, group, sign)."""
    out = out_dir()
    edges = fc.edge_dimensions(pd.read_csv(out / 'recipe_edges.csv', dtype=str, keep_default_na=False))
    duals = pd.read_parquet(out / 'recipe_duals.parquet', columns=[
        'rxn_class', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'edge_id', 'record_position_A', 'record_position_B', 'd'])
    duals = duals.merge(edges, on='edge_id', validate='many_to_one')
    duals['edge_code'] = pd.factorize(duals.edge_id, sort=True)[0]
    for role in [1, 2]:
        keys = duals.rxn_class.astype(str) + '\t' + duals[f'sub_{role}_smiles'].astype(str)
        duals[f'mol{role}'] = pd.factorize(keys, sort=True)[0]
    cd = (duals.d * duals.canonical_sign).to_numpy()
    n_mol = int(max(duals.mol1.max(), duals.mol2.max())) + 1
    lows, highs, groups, signs, mols = [], [], [], [], []
    offset = 0
    for cls in sorted(duals.rxn_class.unique()):
        part = duals.loc[duals.rxn_class.eq(cls), ['edge_code', 'mol1', 'mol2']]
        for role in [1, 2]:
            lo, hi = fc.comparisons(part, role)
            focal, common = duals[f'mol{role}'].to_numpy(), duals[f'mol{3 - role}'].to_numpy()
            pair = focal[lo].astype(np.int64) * n_mol + focal[hi]
            key = np.stack([pair, duals.replacement.to_numpy()[lo], duals.held.to_numpy()[lo]], axis=1)
            _, code = np.unique(key, axis=0, return_inverse=True)
            code = code.ravel()
            a, b = cd[lo], cd[hi]
            sign = np.where((a >= M - fc.TOL) & (b <= -M + fc.TOL), 1, np.where((a <= -M + fc.TOL) & (b >= M - fc.TOL), -1, 0))
            n = int(code.max()) + 1 if len(code) else 0
            pos, neg = np.bincount(code[sign == 1], minlength=n), np.bincount(code[sign == -1], minlength=n)
            opposed = (pos > 0) & (neg > 0)
            keep = (sign != 0) & opposed[code]
            lows.append(lo[keep]); highs.append(hi[keep]); signs.append(sign[keep])
            remap = -np.ones(n, np.int64); remap[opposed] = np.arange(opposed.sum()) + offset
            groups.append(remap[code[keep]]); offset += int(opposed.sum())
            # molecules: the two focal molecules and every common partner in the group's strong events
            g = remap[code[keep]]
            mols.append(pd.DataFrame({'g': np.r_[g, g, g], 'm': np.r_[(role - 1) * n_mol + focal[lo[keep]], (role - 1) * n_mol + focal[hi[keep]],
                                                               (2 - role) * n_mol + common[lo[keep]]]}))
    cat = dict(low=np.concatenate(lows), high=np.concatenate(highs), group=np.concatenate(groups),
               sign=np.concatenate(signs), n=offset,
               pos=duals[['record_position_A', 'record_position_B']].to_numpy())
    m = pd.concat(mols).drop_duplicates().groupby('g').m.agg(list)
    cat['sets'] = [m[i] for i in range(offset)]
    return cat


def evaluate(cat, mask):
    dual_keep = mask[cat['pos']].all(axis=1)
    k = dual_keep[cat['low']] & dual_keep[cat['high']]
    p = np.bincount(cat['group'][k & (cat['sign'] == 1)], minlength=cat['n'])
    q = np.bincount(cat['group'][k & (cat['sign'] == -1)], minlength=cat['n'])
    return (p > 0) & (q > 0)


def main():
    c = Corpus()
    cat = catalogue()
    assert cat['n'] > 0
    assert evaluate(cat, c.valid).all()
    assert len(evaluate(cat, c.rule_union & c.valid)) == cat['n']
    rules = dict(np.load(out_dir() / 'retention_rules.npz'))
    rows = []
    for name, added in rules.items():
        if name == 'R0':
            continue
        r = evaluate(cat, c.mask(added))
        row = dict(rule=name, budget=len(added), retained=int(r.sum()))
        for anchor, draw in (('R4', anchor_R4), ('R4p', anchor_R4p)):
            hits = np.array([evaluate(cat, c.mask(draw(c, len(added), i))) for i in range(N_DRAWS)])
            x = r.astype(float) - hits.mean(axis=0)
            lo, hi = e1.bootstrap(x, cat['sets'])
            row.update({f'{anchor}_mean': float(hits.sum(axis=1).mean()), f'D_vs_{anchor}': float(x.sum()),
                        f'D_vs_{anchor}_lo': lo, f'D_vs_{anchor}_hi': hi})
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir() / 'rules_E3.csv', index=False)


if __name__ == '__main__':
    main()
