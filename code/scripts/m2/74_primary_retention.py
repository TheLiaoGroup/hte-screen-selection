"""Replay the frozen retention definitions on the authorized classifier-zero domain.

Scripts 60-67 and retention.py remain unchanged. Only domain-specific historical
counts are replaced; candidate rules, random draws and endpoint functions are reused.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
import json

import networkx as nx
import numpy as np
import pandas as pd

from m2common import out_dir
from retention import Corpus, N_DRAWS, anchor_R4, anchor_R4p, build_rules
from variant_support import activate, init_e2, e2, Judge

e1 = import_module('61_rules_E1')
e3 = import_module('67_rules_E3')
ANCHORS = [('R4', anchor_R4), ('R4p', anchor_R4p)]


def rules(variant):
    c = Corpus(variant)
    selected = build_rules(c)
    rows = {}
    for name, added in selected.items():
        assert c.pool[added].all() and len(np.unique(added)) == len(added)
        rows[name] = dict(added_records=int(len(added)),
                          added_pairs=int(c.u.iloc[added][['rxn_class', 'subpair']].drop_duplicates().shape[0]))
    assert np.array_equal(c.mask(selected['R3']), c.Q | (c.rule_union & c.valid))
    for _, draw in ANCHORS:
        assert len(draw(c, len(selected['R1-12']), 0)) == len(selected['R1-12'])
    np.savez_compressed(out_dir(variant) / 'retention_rules.npz', **selected)
    (out_dir(variant) / 'retention_rule_budgets.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print(json.dumps(rows, indent=1), flush=True)


def inputs(variant):
    return Corpus(variant), dict(np.load(out_dir(variant) / 'retention_rules.npz'))


def primary(variant):
    c, selected = inputs(variant)
    targets = e1.targets_E1()
    ids = list(targets.edge_id)
    d = e1.strong_duals(ids)
    groups = d.groupby('edge_id', sort=False).indices
    witnesses = e1.witness_weights(targets, d)
    assert e1.evaluate(c.valid, d, groups, ids).all()
    assert not e1.evaluate(c.Q, d, groups, ids).any()
    rows, indicators = [], {}
    for name, added in selected.items():
        if name == 'R0':
            continue
        r = e1.evaluate(c.mask(added), d, groups, ids)
        indicators[name] = r
        row = dict(rule=name, budget=len(added), retained=int(r.sum()))
        for anchor, draw in ANCHORS:
            hits = np.array([e1.evaluate(c.mask(draw(c, len(added), i)), d, groups, ids)
                             for i in range(N_DRAWS)])
            p = hits.mean(axis=0)
            indicators[f'{name}|{anchor}'] = p
            x = r.astype(float) - p
            lo, hi = e1.bootstrap(x, witnesses)
            row.update({f'{anchor}_mean': float(hits.sum(axis=1).mean()),
                        f'{anchor}_min': int(hits.sum(axis=1).min()),
                        f'{anchor}_max': int(hits.sum(axis=1).max()),
                        f'D_vs_{anchor}': float(x.sum()), f'D_vs_{anchor}_lo': lo,
                        f'D_vs_{anchor}_hi': hi, f'beats_{anchor}': bool(x.sum() > 0 and lo > 0)})
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir(variant) / 'rules_E1.csv', index=False)
    pd.DataFrame({'edge_id': ids, **indicators}).to_csv(out_dir(variant) / 'rules_E1_targets.csv', index=False)


def strength(variant, workers):
    init_e2(variant)
    c, g, full, rho_full = (e2._G[k] for k in ('c', 'g', 'full', 'rho'))
    witnesses = []
    for rho, pos in zip(full.rho, e2._G['pos']):
        pos = pos[g['weight'][pos] >= rho - e2.TOL]
        graph = nx.DiGraph()
        graph.add_edges_from(zip(g['source'][pos], g['target'][pos]))
        witnesses.append(sorted({n for arc in nx.find_cycle(graph) for n in arc}))
    _, selected = inputs(variant)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_e2, initargs=(variant,)) as pool:
        for name, added in selected.items():
            if name == 'R0':
                continue
            r, values = e2.evaluate(c.mask(added))
            row = dict(rule=name, budget=len(added), retained=int(r.sum()),
                       rho_ratio=float(values.sum() / rho_full.sum()))
            for anchor, _ in ANCHORS:
                results = list(pool.map(e2._draw, [(anchor, len(added), i) for i in range(N_DRAWS)]))
                hits = np.array([h for h, _ in results])
                x = r.astype(float) - hits.mean(axis=0)
                lo, hi = e1.bootstrap(x, witnesses)
                row.update({f'{anchor}_mean': float(hits.sum(axis=1).mean()),
                            f'{anchor}_rho_ratio': float(np.mean([v.sum() / rho_full.sum() for _, v in results])),
                            f'D_vs_{anchor}': float(x.sum()), f'D_vs_{anchor}_lo': lo, f'D_vs_{anchor}_hi': hi})
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir(variant) / 'rules_E2.csv', index=False)


def backgrounds(variant):
    c, selected = inputs(variant)
    catalogue = e3.catalogue()
    assert catalogue['n'] > 0 and e3.evaluate(catalogue, c.valid).all()
    rows = []
    for name, added in selected.items():
        if name == 'R0':
            continue
        r = e3.evaluate(catalogue, c.mask(added))
        row = dict(rule=name, budget=len(added), retained=int(r.sum()))
        for anchor, draw in ANCHORS:
            hits = np.array([e3.evaluate(catalogue, c.mask(draw(c, len(added), i))) for i in range(N_DRAWS)])
            x = r.astype(float) - hits.mean(axis=0)
            lo, hi = e1.bootstrap(x, catalogue['sets'])
            row.update({f'{anchor}_mean': float(hits.sum(axis=1).mean()),
                        f'D_vs_{anchor}': float(x.sum()), f'D_vs_{anchor}_lo': lo, f'D_vs_{anchor}_hi': hi})
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir(variant) / 'rules_E3.csv', index=False)


def heldout(variant):
    c, selected = inputs(variant)
    judge = Judge(variant)
    witnesses = judge.molecules()
    rows = []
    for name, added in selected.items():
        if name == 'R0':
            continue
        a, k = judge.per_target(c.mask(added)[None, :])
        row = dict(rule=name, budget=len(added), answered=int(a.sum()), correct=int(k.sum()))
        for anchor, draw in ANCHORS:
            aa, kk = judge.per_target(np.vstack([c.mask(draw(c, len(added), i)) for i in range(N_DRAWS)]))
            for label, mine, draws in [('answered', a[0], aa), ('correct', k[0], kk)]:
                x = mine.astype(float) - draws.mean(axis=0)
                lo, hi = e1.bootstrap(x, witnesses)
                row.update({f'{anchor}_{label}_mean': float(draws.sum(axis=1).mean()),
                            f'D_{label}_vs_{anchor}': float(x.sum()), f'D_{label}_vs_{anchor}_lo': lo,
                            f'D_{label}_vs_{anchor}_hi': hi})
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir(variant) / 'rules_E4.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['clfzero'], required=True)
    parser.add_argument('--stage', choices=['rules', 'e1', 'e2', 'e3', 'e4', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    activate(args.variant)
    stages = ['rules', 'e1', 'e2', 'e3', 'e4'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        {'rules': rules, 'e1': primary, 'e2': lambda v: strength(v, args.workers),
         'e3': backgrounds, 'e4': heldout}[stage](args.variant)


if __name__ == '__main__':
    main()
