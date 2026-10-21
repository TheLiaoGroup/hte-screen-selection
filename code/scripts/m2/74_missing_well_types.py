"""Composition of missing wells in pairs included in or excluded from Q."""
import argparse
import json

import numpy as np
import pandas as pd
from m2common import data_dir, out_dir

SEED, N_BOOT, MIN_WELLS = 20261018, 2000, 20
TYPES = ['quantified', 'zero_by_rule', 'detected_not_quantified', 'no_screen_record']
RAW_TYPES = ['numeric', 'blank_detected', 'blank_no_product', 'blank_unflagged']
RAW_STATUS_NAMES = dict(zip(TYPES, ['numeric', 'blank_no_product', 'blank_detected', 'blank_unflagged']))


def measurement_status(raw_yield, clf):
    """Well status from the exported yield field and the DESI flag (independent of the analysis coding)."""
    blank = pd.to_numeric(raw_yield, errors='coerce').isna().to_numpy()
    return np.select([~blank, clf.eq(0).to_numpy(), clf.eq(1).to_numpy()], TYPES[:3], TYPES[3])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    v = ap.parse_args().variant
    u = pd.read_parquet(data_dir(v) / 'observations.parquet',
                        columns=['rxn_class', 'subpair', 'raw_yield', 'yield_clf', 'duplicate_cell', 'qualified_row'])
    a = u.loc[u.rxn_class.ne('Finkelstein') & ~u.duplicate_cell].copy()
    a['type'] = measurement_status(a.raw_yield, a.yield_clf)
    types = TYPES
    if v == 'raw':
        # Preserve the archived numeric-only labels and bootstrap column order.
        types = RAW_TYPES
        a['type'] = a['type'].map(RAW_STATUS_NAMES)
    p = pd.crosstab([a.rxn_class, a.subpair], a.type).reindex(columns=types, fill_value=0)
    p['n'] = p.sum(axis=1)
    p['included'] = a.groupby(['rxn_class', 'subpair']).qualified_row.any()
    p = p.loc[p.n >= MIN_WELLS].copy()
    for t in types:
        p[f'share_{t}'] = p[t] / p.n
    counts = p.groupby([p.index.get_level_values(0), 'included']).size().unstack(fill_value=0)
    print(counts.to_string(), flush=True)
    classes = [c for c in counts.index if counts.loc[c].get(True, 0) and counts.loc[c].get(False, 0)]
    assert classes
    rng = np.random.default_rng(SEED)
    res = dict(pairs={'included': int(p.included.sum()), 'excluded': int((~p.included).sum())}, classes=classes, pooled={},
               blank_wells_by_group={str(k): {t: int(v) for t, v in r.items()} for k, r in p.groupby('included')[types[1:]].sum().iterrows()})
    for col in [f'share_{t}' for t in types]:
        parts = [(p.loc[c].loc[lambda x: x[col].notna()]) for c in classes]
        def diff(xs):
            return np.mean([x.loc[~x.included, col].mean() - x.loc[x.included, col].mean() for x in xs])
        est = diff(parts)
        boots = []
        for _ in range(N_BOOT):
            boots.append(diff([x.iloc[rng.integers(0, len(x), len(x))] for x in parts]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        inc = float(np.mean([x.loc[x.included, col].mean() for x in parts])); exc = float(np.mean([x.loc[~x.included, col].mean() for x in parts]))
        res['pooled'][col] = dict(included=inc, excluded=exc, difference=float(est), lo=float(lo), hi=float(hi))
        print(f'{col}: included {inc:.4f} excluded {exc:.4f} diff {est:.4f} [{lo:.4f}, {hi:.4f}]', flush=True)
    (out_dir(v) / 'missing_well_types.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
