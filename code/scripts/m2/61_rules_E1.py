"""Primary endpoint E1: how many newly contradicted contrasts
remain cyclic at 5 pp on S = Q | A, for each rule and its equal-budget random anchors.

Retention follows the rule-union evaluation of 10_partner_score_counterexamples.py: duals in the
core/outside domain, kept only when both records are in S.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from m2common import out_dir
from retention import Corpus, N_DRAWS, SEED, anchor_R4, anchor_R4p
from importlib import import_module

ps = import_module('10_partner_score_counterexamples')
MARGIN = ps.MARGIN / 100 - ps.TOL
N_BOOT = 2000


def targets_E1():
    res = pd.read_csv(out_dir() / 'partner_score_edges.csv')
    new = res.loc[res.new_contradiction].reset_index(drop=True)
    assert len(new) > 0 and new.edge_id.is_unique
    return new


def strong_duals(edge_ids):
    d = pd.read_parquet(out_dir() / 'recipe_duals.parquet')
    d = d.loc[d.core_outside_domain & d.edge_id.isin(edge_ids) & d.d.abs().ge(MARGIN)].copy()
    d['dual_id'] = d.record_id_A.astype(str) + '|' + d.record_id_B.astype(str)
    return d


def evaluate(mask, d, groups, edge_ids):
    keep = mask[d.record_position_A.to_numpy()] & mask[d.record_position_B.to_numpy()]
    out = np.zeros(len(edge_ids), dtype=bool)
    for k, e in enumerate(edge_ids):
        z = d.iloc[groups[e]]
        z = z.loc[keep[groups[e]]]
        out[k] = len(z) > 0 and not nx.is_directed_acyclic_graph(ps.build_graph(z))
    return out


def witness_weights(new, d):
    """Distinct molecules (class, role, SMILES) of each target's cycle certificate."""
    by_id = d.set_index('dual_id')
    sets = []
    for cls, ids in zip(new.rxn_class, new.certificate_dual_ids):
        z = by_id.loc[json.loads(ids)]
        sets.append(sorted({(cls, 1, s) for s in z.sub_1_smiles} | {(cls, 2, s) for s in z.sub_2_smiles}))
    return sets


def bootstrap(x, sets, seed=SEED):
    mols = sorted({m for s in sets for m in s})
    index = {m: i for i, m in enumerate(mols)}
    member = [np.array([index[m] for m in s]) for s in sets]
    rng = np.random.default_rng(seed)
    w = rng.poisson(1.0, size=(N_BOOT, len(mols))).astype(float)
    wt = np.column_stack([w[:, m].mean(axis=1) for m in member])  # N_BOOT x targets
    denom = wt.sum(axis=1)
    stat = len(x) * (wt * x).sum(axis=1) / np.where(denom > 0, denom, np.nan)
    return np.nanpercentile(stat, [2.5, 97.5]).tolist()


def main():
    c = Corpus()
    rules = dict(np.load(out_dir() / 'retention_rules.npz'))
    new = targets_E1()
    edge_ids = list(new.edge_id)
    d = strong_duals(edge_ids)
    groups = d.groupby('edge_id', sort=False).indices
    sets = witness_weights(new, d)
    # Sanity anchors: full data retains all 179; Q retains none; the rule union retains 14.
    assert evaluate(c.valid, d, groups, edge_ids).all()
    assert not evaluate(c.Q, d, groups, edge_ids).any()
    assert evaluate(c.mask(rules['R3']), d, groups, edge_ids).sum() == int(new.rule_union_cyclic.sum())
    rows, per_target = [], {}
    for name, added in rules.items():
        if name == 'R0':
            continue
        r = evaluate(c.mask(added), d, groups, edge_ids)
        per_target[name] = r
        row = dict(rule=name, budget=len(added), retained=int(r.sum()))
        for anchor, draw in (('R4', anchor_R4), ('R4p', anchor_R4p)):
            hits = np.array([evaluate(c.mask(draw(c, len(added), i)), d, groups, edge_ids) for i in range(N_DRAWS)])
            p = hits.mean(axis=0)
            per_target[f'{name}|{anchor}'] = p
            x = r.astype(float) - p
            lo, hi = bootstrap(x, sets)
            row.update({f'{anchor}_mean': float(hits.sum(axis=1).mean()),
                        f'{anchor}_min': int(hits.sum(axis=1).min()), f'{anchor}_max': int(hits.sum(axis=1).max()),
                        f'D_vs_{anchor}': float(x.sum()), f'D_vs_{anchor}_lo': lo, f'D_vs_{anchor}_hi': hi,
                        f'beats_{anchor}': bool(x.sum() > 0 and lo > 0)})
        rows.append(row)
        print(row, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir() / 'rules_E1.csv', index=False)
    pd.DataFrame({'edge_id': edge_ids, **per_target}).to_csv(out_dir() / 'rules_E1_targets.csv', index=False)


if __name__ == '__main__':
    main()
