"""Yield-blind retention rules and equal-budget random anchors.

No function here reads a yield. Every retained set is S = Q | A, returned as a boolean mask over the
rows of data/observations.parquet; A is drawn from the pool of valid non-Q records.
"""
import numpy as np
import pandas as pd
from m2common import data_dir, out_dir

SEED = 20261018
N_DRAWS = 100


class Corpus:
    def __init__(self, variant=None):
        cols = ['rxn_class', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'condition_id', 'screen',
                'analysis_valid', 'qualified_row', 'pair_group', 'record_id']
        u = pd.read_parquet(data_dir(variant) / 'observations.parquet', columns=cols)
        self.u = u
        self.variant = variant
        self.n = len(u)
        self.valid = u.analysis_valid.to_numpy()
        self.Q = (u.qualified_row & u.analysis_valid).to_numpy()
        self.pool = self.valid & ~self.Q
        masks = np.load(out_dir(variant) / 'rule_masks.npz')
        self.rule_union = np.unpackbits(masks['rule_union'], bitorder='big')[:self.n].astype(bool)
        # Pool units for R4p: one outside-only pair, or the noncore records of one mixed pair.
        p = u.loc[self.pool, ['rxn_class', 'subpair']]
        self.unit_of = p.groupby(['rxn_class', 'subpair'], sort=True).ngroup().to_numpy()
        self.pool_positions = np.flatnonzero(self.pool)

    def mask(self, added):
        s = self.Q.copy()
        s[added] = True
        assert not (s & ~self.valid).any()
        return s


def rule_R1(c, k):
    """Whole outside-only pairs sharing >= k measured conditions with one qualified screen of their class."""
    u = c.u
    screens = u.loc[c.Q].groupby(['rxn_class', 'screen']).condition_id.agg(lambda s: frozenset(s))
    out = u.loc[c.valid & u.pair_group.eq('outside_only').to_numpy()]
    menus = out.groupby(['rxn_class', 'subpair']).condition_id.agg(lambda s: frozenset(s))
    keep = []
    for (cls, sp), menu in menus.items():
        conds = screens.loc[cls] if cls in screens.index.get_level_values(0) else []
        if max((len(menu & s) for s in conds), default=0) >= k:
            keep.append((cls, sp))
    keep = pd.MultiIndex.from_tuples(keep, names=['rxn_class', 'subpair'])
    sel = pd.MultiIndex.from_frame(u[['rxn_class', 'subpair']]).isin(keep) & c.pool
    return np.flatnonzero(sel)


def rule_R2(c, edges, duals):
    """Pool records completing a 2-substrate x 2-recipe comparison with a Q dual sharing exactly one partner."""
    q = duals.loc[duals.identity_A.eq('Q') & duals.identity_B.eq('Q')]
    other = duals.loc[~(duals.identity_A.eq('Q') & duals.identity_B.eq('Q'))]
    p1 = set(zip(q.edge_id, q.sub_1_smiles))
    p2 = set(zip(q.edge_id, q.sub_2_smiles))
    hit = np.fromiter(((e, a) in p1 or (e, b) in p2 for e, a, b in
                       zip(other.edge_id, other.sub_1_smiles, other.sub_2_smiles)), bool, len(other))
    chosen = other.loc[hit]
    pos = np.unique(np.concatenate([chosen.record_position_A.to_numpy(), chosen.record_position_B.to_numpy()]))
    return pos[c.pool[pos]]


def rule_R3(c):
    return np.flatnonzero(c.rule_union & c.pool)


def anchor_R4(c, budget, i):
    rng = np.random.default_rng(SEED + i)
    return np.sort(rng.choice(c.pool_positions, size=budget, replace=False))


def anchor_R4p(c, budget, i):
    rng = np.random.default_rng(SEED + i)
    order = rng.permutation(c.unit_of.max() + 1)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    by_unit = c.pool_positions[np.argsort(rank[c.unit_of], kind='stable')]
    sizes = np.bincount(c.unit_of)[order]
    whole = np.searchsorted(np.cumsum(sizes), budget, side='right')
    taken = int(sizes[:whole].sum())
    chosen = by_unit[:taken]
    if taken < budget:
        nxt = by_unit[taken:taken + sizes[whole]]
        chosen = np.concatenate([chosen, rng.choice(nxt, size=budget - taken, replace=False)])
    assert len(chosen) == budget
    return np.sort(chosen)


def build_rules(c):
    edges = pd.read_csv(out_dir(c.variant) / 'recipe_edges.csv', usecols=['rxn_class', 'edge_id'])
    duals = pd.read_parquet(out_dir(c.variant) / 'recipe_duals.parquet',
                            columns=['edge_id', 'sub_1_smiles', 'sub_2_smiles', 'identity_A', 'identity_B',
                                     'record_position_A', 'record_position_B'])
    return {'R0': np.array([], dtype=np.int64), 'R1-6': rule_R1(c, 6), 'R1-12': rule_R1(c, 12),
            'R1-24': rule_R1(c, 24), 'R2': rule_R2(c, edges, duals), 'R3': rule_R3(c)}
