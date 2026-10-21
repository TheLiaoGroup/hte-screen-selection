"""Route a revised data domain through unchanged frozen endpoint functions."""
from importlib import import_module
import os

import numpy as np
import pandas as pd

from m2common import active_variant, data_dir, out_dir
from retention import Corpus

e2 = import_module('62_rules_E2')
e4 = import_module('65_rules_E4')


def activate(variant):
    variant = active_variant(variant)
    os.environ['M2_VARIANT'] = variant
    return variant


def init_e2(variant):
    """Replace historical count checks with checks against this domain's targets."""
    activate(variant)
    c = Corpus(variant)
    g = e2.cs.load_graph(variant)
    full = pd.read_csv(out_dir(variant) / 'constraint_strength_edges.csv')
    full = full.loc[full.universe.eq('extended') & full.recordset_id.eq('full') & full.rho.gt(0)]
    assert len(full) > 0 and full.edge_id.is_unique
    e2._G.clear()
    e2._G.update(c=c, g=g, full=full,
                 pos=[g['groups'][int(e)] for e in full.edge_position], rho=full.rho.to_numpy())
    assert e2.evaluate(c.valid)[0].all()


class Judge(e4.Judge):
    """Use the frozen judgment method on the chosen structural catalogue."""

    def __init__(self, variant):
        self.variant = activate(variant)
        refs = pd.read_parquet(out_dir(variant) / 'structural_refs.parquet')
        self.targets, self.endpoint, self.adj = e4.ho.leaveout_graph(refs)
        t = self.targets
        self.truth = e4.ho.events(t.d_variable_low.to_numpy(), t.d_variable_high.to_numpy(), 5)
        self.strong = self.truth != 0
        assert self.strong.any()

    def molecules(self):
        u = pd.read_parquet(data_dir(self.variant) / 'observations.parquet',
                            columns=['rxn_class', 'sub_1_smiles', 'sub_2_smiles'])
        cls, s1, s2 = (u[c].to_numpy(object) for c in ('rxn_class', 'sub_1_smiles', 'sub_2_smiles'))
        return [sorted({(cls[p], 1, s1[p]) for p in row} | {(cls[p], 2, s2[p]) for p in row})
                for row in self.endpoint[self.strong]]
