"""Secondary endpoint E4: fixed held-out directional judgments on the
1,109 strong targets at 5 pp, training on S = Q | A, for each rule and its anchors.
Uses leaveout_graph / leaveout of 43_heldout_crossgoal_replay.py; also exports per-target judgments
for 66_identity_composition_E4.py.
"""
from importlib import import_module

import numpy as np
import pandas as pd
from m2common import data_dir, out_dir
from retention import Corpus, N_DRAWS, anchor_R4, anchor_R4p

ho = import_module('43_heldout_crossgoal_replay')
e1 = import_module('61_rules_E1')


class Judge:
    def __init__(self):
        refs = pd.read_parquet(out_dir() / 'structural_refs.parquet')
        self.targets, self.endpoint, self.adj = ho.leaveout_graph(refs)
        t = self.targets
        self.truth = ho.events(t.d_variable_low.to_numpy(), t.d_variable_high.to_numpy(), 5)
        self.strong = self.truth != 0
        assert self.strong.any()

    def per_target(self, masks):
        keep = masks[:, self.endpoint].all(axis=2)
        n_train = np.asarray(self.adj @ keep.T.astype(np.int32)).T
        pos = np.asarray(self.adj @ (keep & (self.truth == 1)).T.astype(np.int32)).T
        neg = np.asarray(self.adj @ (keep & (self.truth == -1)).T.astype(np.int32)).T
        pred = np.where(n_train >= 2, np.sign(pos - neg), 0)
        answered = (pred != 0)[:, self.strong]
        correct = answered & (pred == self.truth)[:, self.strong]
        return answered, correct

    def molecules(self):
        u = pd.read_parquet(data_dir() / 'observations.parquet', columns=['rxn_class', 'sub_1_smiles', 'sub_2_smiles'])
        cls, s1, s2 = (u[c].to_numpy(object) for c in ('rxn_class', 'sub_1_smiles', 'sub_2_smiles'))
        return [sorted({(cls[p], 1, s1[p]) for p in row} | {(cls[p], 2, s2[p]) for p in row})
                for row in self.endpoint[self.strong]]


def main():
    c = Corpus()
    j = Judge()
    sets = j.molecules()
    a, k = j.per_target(np.vstack([c.rule_union & c.valid, c.valid]))
    assert (k <= a).all() and a[1].sum() > 0
    rules = dict(np.load(out_dir() / 'retention_rules.npz'))
    rows = []
    for name, added in rules.items():
        if name == 'R0':
            continue
        a, k = j.per_target(c.mask(added)[None, :])
        row = dict(rule=name, budget=len(added), answered=int(a.sum()), correct=int(k.sum()))
        for anchor, draw in (('R4', anchor_R4), ('R4p', anchor_R4p)):
            aa, kk = j.per_target(np.vstack([c.mask(draw(c, len(added), i)) for i in range(N_DRAWS)]))
            for label, mine, draws in (('answered', a[0], aa), ('correct', k[0], kk)):
                x = mine.astype(float) - draws.mean(axis=0)
                lo, hi = e1.bootstrap(x, sets)
                row.update({f'{anchor}_{label}_mean': float(draws.sum(axis=1).mean()),
                            f'D_{label}_vs_{anchor}': float(x.sum()), f'D_{label}_vs_{anchor}_lo': lo,
                            f'D_{label}_vs_{anchor}_hi': hi})
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(out_dir() / 'rules_E4.csv', index=False)


if __name__ == '__main__':
    main()
