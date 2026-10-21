"""Post hoc cross-goal recovery by the existing yield-blind retention rules.

The 5 pp structural-reference targets and rule masks are read unchanged. These
counts extend the frozen E1-E4 evaluation; they are not new frozen endpoints.
"""
import argparse
from importlib import import_module
import json

import numpy as np
import pandas as pd

from m2common import data_dir, out_dir


replay = import_module('43_heldout_crossgoal_replay')
goal_codes = import_module('41_crossgoal_targets').goal_codes
RULES = ['R1-6', 'R1-12', 'R1-24', 'R2', 'R3']


def unpack(value, n):
    return np.unpackbits(value, bitorder='big')[:n].astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='clfzero')
    variant = parser.parse_args().variant
    out = out_dir(variant)
    observations = pd.read_parquet(data_dir(variant) / 'observations.parquet',
                                   columns=['analysis_valid', 'qualified_row'])
    valid = observations.analysis_valid.to_numpy(bool)
    q = observations.qualified_row.to_numpy(bool) & valid
    with np.load(out / 'rule_masks.npz') as packed:
        union = unpack(packed['rule_union'], len(valid)) & valid
    sets = {'Q': q}
    with np.load(out / 'retention_rules.npz') as packed:
        for name in RULES:
            added = packed[name]
            assert valid[added].all() and not q[added].any()
            mask = q.copy()
            mask[added] = True
            sets[name] = mask
    sets['screen_rule_union'] = union
    with np.load(out / 'restoration_joint_returned_mask.npz') as packed:
        sets['joint_full_strength_M2'] = unpack(packed['mask'], len(valid)) & valid
    sets['full'] = valid
    names = list(sets)
    masks = np.vstack([sets[name] for name in names])
    assert (masks[:, q]).all() and not (masks & ~valid).any()
    assert np.array_equal(sets['R3'], union)

    refs = pd.read_parquet(out / 'structural_refs.parquet')
    residual = pd.read_csv(out / 'crossgoal_targets.csv.gz', keep_default_na=False)
    residual = residual.loc[residual.margin_pp.eq(5)].copy()
    assert len(residual) > 0
    counts = residual.goal.value_counts().to_dict()
    assert set(counts) == set(replay.GOALS)
    if variant == 'raw':
        assert counts == {'partner': 227, 'structural': 50, 'held_recipe': 48}
    recovered = replay.crossgoal(masks, refs, goal_codes(refs), residual, pp=5)
    rows = []
    for index, name in enumerate(names):
        mask = sets[name]
        rows.append({
            'recordset': name, 'analysis_stage': 'post_hoc', 'margin_pp': 5,
            'n_core_records': int(q.sum()), 'n_retained_records': int(mask.sum()),
            'added_to_Q': int((mask & ~q).sum()),
            'added_to_union': int((mask & ~union).sum()) if union[~mask].sum() == 0 else pd.NA,
            **{f'target_{goal}': counts[goal] for goal in replay.GOALS},
            **{f'recovered_{goal}': int(recovered[goal][index]) for goal in replay.GOALS},
        })
    result = pd.DataFrame(rows)
    result['added_to_union'] = result.added_to_union.astype('Int64')
    indexed = result.set_index('recordset')
    columns = [f'recovered_{goal}' for goal in replay.GOALS]
    assert indexed.loc['full', columns].tolist() == [counts[goal] for goal in replay.GOALS]
    for name in ('Q', 'R3', 'screen_rule_union'):
        assert indexed.loc[name, columns].tolist() == [0, 0, 0], name
    solutions = pd.read_csv(out / 'restoration_class_solutions.csv')
    joint_minimum = solutions.loc[solutions.universe.eq('extended') &
                                  solutions.mode_id.eq('joint_full_strength'), 'minimum_additions'].sum()
    assert indexed.loc['joint_full_strength_M2', 'added_to_union'] == joint_minimum
    assert indexed.loc['joint_full_strength_M2', 'added_to_Q'] == (union & ~q).sum() + joint_minimum
    if variant == 'raw':
        for name in ('R2', 'R1-12'):
            assert indexed.loc[name, columns].tolist() == [50, 227, 48], name
        assert indexed.loc['joint_full_strength_M2', columns].tolist() == [3, 29, 2]
        assert indexed.loc['joint_full_strength_M2', 'added_to_Q'] == 27442
        assert indexed.loc['joint_full_strength_M2', 'added_to_union'] == 2224
    original = pd.read_csv(out / 'heldout_replay.csv').set_index('recordset')
    for name, reference in [('screen_rule_union', 'rule_union'),
                            ('joint_full_strength_M2', 'joint_full_strength_M2'), ('full', 'full')]:
        assert indexed.loc[name, columns].tolist() == original.loc[reference, columns].tolist()
    target = out / 'rules_crossgoal.csv'
    result.to_csv(target, index=False)
    print(result.to_string(index=False))
    print(json.dumps({'output': str(target), 'post_hoc': True, 'expected_counts_match': True}))


if __name__ == '__main__':
    main()
