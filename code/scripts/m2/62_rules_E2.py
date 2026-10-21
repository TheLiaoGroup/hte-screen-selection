"""Secondary endpoint E2: of contrasts with full-data rho > 0
(extended domain), how many keep their full rho on S = Q | A, for each rule and its anchors.
Also reports sum rho(S) / sum rho(full). Uses load_graph and whole_rho of 20_constraint_strength.py.
"""
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module

import networkx as nx
import numpy as np
import pandas as pd
from m2common import out_dir
from retention import Corpus, N_DRAWS, anchor_R4, anchor_R4p

cs = import_module('20_constraint_strength')
e1 = import_module('61_rules_E1')
TOL = 1e-12


_G = {}


def _init():
    c = Corpus()
    g = cs.load_graph()
    full = pd.read_csv(out_dir() / 'constraint_strength_edges.csv')
    full = full.loc[full.universe.eq('extended') & full.recordset_id.eq('full') & full.rho.gt(0)]
    assert len(full) > 0 and full.edge_id.is_unique
    _G.update(c=c, g=g, full=full, pos=[g['groups'][int(e)] for e in full.edge_position], rho=full.rho.to_numpy())


def evaluate(mask):
    g, pos, rho_full = _G['g'], _G['pos'], _G['rho']
    active = mask[g['a']] & mask[g['b']]
    r = np.array([cs.whole_rho(p[active[p]], g['source'], g['target'], g['weight']) if active[p].any() else 0.0 for p in pos])
    return np.abs(r - rho_full) <= TOL, r


def _draw(job):
    anchor, budget, i = job
    c = _G['c']
    added = (anchor_R4 if anchor == 'R4' else anchor_R4p)(c, budget, i)
    return evaluate(c.mask(added))


def main():
    _init()
    c, g, full, rho_full = _G['c'], _G['g'], _G['full'], _G['rho']
    w = g['weight']
    # Witness molecules: one directed cycle among full-data duals with weight >= rho.
    sets = []
    for rho, p in zip(full.rho, _G['pos']):
        p = p[w[p] >= rho - TOL]
        G = nx.DiGraph(); G.add_edges_from(zip(g['source'][p], g['target'][p]))
        sets.append(sorted({n for arc in nx.find_cycle(G) for n in arc}))
    assert evaluate(c.valid)[0].all()
    rules = dict(np.load(out_dir() / 'retention_rules.npz'))
    assert len(evaluate(c.mask(rules['R3']))[0]) == len(full)
    rows = []
    with ProcessPoolExecutor(max_workers=10, initializer=_init) as pool:
        for name, added in rules.items():
            if name == 'R0':
                continue
            r, val = evaluate(c.mask(added))
            row = dict(rule=name, budget=len(added), retained=int(r.sum()), rho_ratio=float(val.sum() / rho_full.sum()))
            for anchor in ('R4', 'R4p'):
                res = list(pool.map(_draw, [(anchor, len(added), i) for i in range(N_DRAWS)]))
                hits = np.array([h for h, _ in res])
                ratios = [v.sum() / rho_full.sum() for _, v in res]
                x = r.astype(float) - hits.mean(axis=0)
                lo, hi = e1.bootstrap(x, sets)
                row.update({f'{anchor}_mean': float(hits.sum(axis=1).mean()), f'{anchor}_rho_ratio': float(np.mean(ratios)),
                            f'D_vs_{anchor}': float(x.sum()), f'D_vs_{anchor}_lo': lo, f'D_vs_{anchor}_hi': hi})
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir() / 'rules_E2.csv', index=False)


if __name__ == '__main__':
    main()
