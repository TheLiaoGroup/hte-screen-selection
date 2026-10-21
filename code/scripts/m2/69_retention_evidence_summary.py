"""Post hoc comparison of existing retained sets across fixed evidence catalogues.

Frozen rule results are read unchanged. The joint strength-restoration set is
evaluated with the existing E1/E2/E3 helpers, without anchor draws or retuning.
"""
import argparse
from importlib import import_module

import numpy as np
import pandas as pd

from m2common import active_variant, out_dir
from retention import Corpus
from variant_support import activate, init_e2, Judge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant')
    variant = activate(active_variant(parser.parse_args().variant))
    out = out_dir(variant)
    c = Corpus(variant)
    with np.load(out / 'retention_rules.npz') as packed:
        four_cell = c.mask(packed['R2'])
    with np.load(out / 'restoration_joint_returned_mask.npz') as packed:
        joint = np.unpackbits(packed['mask'], bitorder='big')[:c.n].astype(bool) & c.valid
    sets = {'screen_rule_union': c.rule_union & c.valid,
            'joint_full_strength_M2': joint, 'R2': four_cell, 'full': c.valid}
    crossgoal = pd.read_csv(out / 'rules_crossgoal.csv').set_index('recordset')
    frozen = {number: pd.read_csv(out / f'rules_E{number}.csv').set_index('rule')
              for number in (1, 2, 3, 4)}
    heldout = pd.read_csv(out / 'heldout_replay.csv').set_index('recordset')

    e1 = import_module('61_rules_E1')
    targets = pd.read_csv(out / 'partner_score_edges.csv')
    targets = targets.loc[targets.new_contradiction].reset_index(drop=True)
    assert len(targets) > 0 and targets.edge_id.is_unique
    edge_ids = targets.edge_id.tolist()
    duals = e1.strong_duals(edge_ids)
    groups = duals.groupby('edge_id', sort=False).indices
    e1_counts = {name: int(e1.evaluate(mask, duals, groups, edge_ids).sum())
                 for name, mask in sets.items()}
    assert e1.evaluate(c.valid, duals, groups, edge_ids).all()

    print('Evaluating post hoc strength comparison.', flush=True)
    e2 = import_module('62_rules_E2')
    init_e2(variant)
    n_strength_targets = len(e2._G['rho'])
    full_strength_sum = float(e2._G['rho'].sum())
    strengths = {}
    for name, mask in sets.items():
        exact, value = e2.evaluate(mask)
        strengths[name] = (int(exact.sum()), float(value.sum()))
    for name, rule in [('screen_rule_union', 'R3'), ('R2', 'R2')]:
        assert strengths[name][0] == int(frozen[2].loc[rule, 'retained'])
        assert abs(strengths[name][1] / full_strength_sum - frozen[2].loc[rule, 'rho_ratio']) <= 1e-12
    assert strengths['joint_full_strength_M2'][0] == n_strength_targets
    assert abs(strengths['joint_full_strength_M2'][1] - full_strength_sum) <= 1e-10

    print('Evaluating post hoc opposed-background comparison.', flush=True)
    e3 = import_module('67_rules_E3')
    catalogue = e3.catalogue()
    background = pd.read_csv(out / 'fixed_change_background_summary.csv')
    full_background = background.loc[background.axis.eq('partner') & background.rxn_class.eq('ALL') &
                                     background.scope.eq('full') & background.margin_pp.eq(5)]
    assert len(full_background) == 1
    assert catalogue['n'] == int(full_background.n_full_opposed_groups.iloc[0])
    e3_counts = {name: int(e3.evaluate(catalogue, mask).sum()) for name, mask in sets.items()}
    assert e3.evaluate(catalogue, c.valid).all()

    judge = Judge(variant)
    answered, correct = judge.per_target(np.vstack(list(sets.values())))
    judgments = {name: (int(a.sum()), int(k.sum()))
                 for name, a, k in zip(sets, answered, correct)}
    n_judgment_targets = int(judge.strong.sum())
    for name, reference in [('screen_rule_union', 'rule_union'),
                            ('joint_full_strength_M2', 'joint_full_strength_M2'), ('full', 'full')]:
        assert judgments[name] == tuple(int(heldout.loc[reference, field]) for field in ('answered', 'correct'))
    if variant == 'raw':
        assert len(targets) == 179 and n_strength_targets == 1446
        assert catalogue['n'] == 35119 and n_judgment_targets == 1109

    rows = []
    for name, mask in sets.items():
        assert mask[c.Q].all() and not (mask & ~c.valid).any()
        e1_count, e3_count = e1_counts[name], e3_counts[name]
        answered, correct = judgments[name]
        if name in ('screen_rule_union', 'R2'):
            rule = 'R3' if name == 'screen_rule_union' else 'R2'
            assert e1_count == int(frozen[1].loc[rule, 'retained'])
            assert e3_count == int(frozen[3].loc[rule, 'retained'])
            assert (answered, correct) == tuple(int(frozen[4].loc[rule, field]) for field in ('answered', 'correct'))
        exact, retained_strength_sum = strengths[name]
        cross = crossgoal.loc[name]
        rows.append({
            'recordset': name, 'comparison_stage': 'post_hoc',
            'n_core_records': int(c.Q.sum()), 'n_retained_records': int(mask.sum()),
            'added_to_Q': int((mask & ~c.Q).sum()),
            'E1_retained': e1_count, 'E1_total': len(targets),
            'E2_full_strength': exact, 'E2_total': n_strength_targets,
            'E2_rho_sum': retained_strength_sum, 'E2_full_rho_sum': full_strength_sum,
            'E2_rho_ratio': retained_strength_sum / full_strength_sum,
            'E3_retained': e3_count, 'E3_total': catalogue['n'],
            'E4_answered': answered, 'E4_correct': correct, 'E4_total': n_judgment_targets,
            **{field: int(cross[field]) for field in (
                'target_structural', 'target_partner', 'target_held_recipe',
                'recovered_structural', 'recovered_partner', 'recovered_held_recipe')},
        })
    result = pd.DataFrame(rows)
    if variant == 'raw':
        assert result.added_to_Q.tolist() == [25218, 27442, 44600, 105426]
    target = out / 'retention_evidence_summary.csv'
    result.to_csv(target, index=False)
    print(result.to_string(index=False))
    print(f'Wrote {target}', flush=True)


if __name__ == '__main__':
    main()
