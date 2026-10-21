"""Necessity tolerance (frozen protocol 4.3, first bullet): joint full-strength restoration in the
extended domain from the rule union, with every target relaxed to rho_full - tau and rho4_full - tau
(tau = 0, 0.5, 1, 2 pp; targets whose relaxed value is <= 0 are dropped). A witness is eligible when every
edge has |d| >= the relaxed bottleneck.

Per tau: targets kept, minimum joint budget (class MILPs from script 21), |F| (union of per-target
intersections of eligible candidate sets), |I*| (records whose exclusion raises the class optimum) and,
when complete witness enumeration stays under LIMIT per target, the number of minimum-size sets
(complete family from script 22). tau = 0 reproduces the main joint result.
"""
import argparse
from importlib import import_module
import json
import time

import numpy as np
import pandas as pd
from m2common import out_dir

r21 = import_module('21_joint_restoration')
r22 = import_module('22_restoration_family')
TAUS_PP = [0, 0.5, 1, 2]
LIMIT = 200000
TOL = 1e-12


def joint_graphs(targets, cls):
    goals = r21.requested_goals(targets, 'extended', 'joint_full_strength', cls)
    graphs = [goal for goal in goals if not goal['actual_satisfied']
              and not (goal['kind'] == 'any' and goal['same_strength'])]
    return goals, graphs


def excluding(graphs, record, missing):
    return [dict(goal, positions=np.array([p for p in goal['positions'] if record not in missing[int(p)]], int))
            for goal in graphs]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    v = ap.parse_args().variant
    g = r21.load_graph(v)
    strength = pd.read_csv(out_dir(v) / 'constraint_strength_edges.csv', float_precision='round_trip',
                           keep_default_na=False, na_values=['']).set_index(['universe', 'recordset_id', 'edge_id'])
    full = strength.loc['extended'].loc['full']
    rows = []
    for tau_pp in TAUS_PP:
        start = time.time()
        tol = tau_pp / 100
        targets, missing = r21.build_targets(g, strength, tol=tol, universes=['extended'])
        position = dict(zip(g['edges'].edge_id, g['edges'].edge_position))
        # duals within 1e-12 of a relaxed bottleneck but not exactly at it (float-rounding sensitivity)
        boundary = sum(int(np.count_nonzero((np.abs(g['weight'][p] - goal['threshold']) <= TOL)
                                            & (g['weight'][p] != goal['threshold'])))
                       for goal in targets.values() for p in [g['groups'][position[goal['edge_id']]]])
        cost = mandatory = intersection = 0
        for cls in g['classes']:
            goals, graphs = joint_graphs(targets, cls)
            answer = r21.optimize(graphs, missing, g['source'], g['target'])
            assert answer['feasible'] and answer['status'] == 0
            required = set().union(*(goal['mandatory'] for goal in goals))
            assert required <= set(answer['records'])
            essential = set(required)
            for record in sorted(set(answer['records']) - required):
                other = r21.optimize(excluding(graphs, record, missing), missing, g['source'], g['target'])
                if not other['feasible'] or other['cost'] > answer['cost']:
                    essential.add(record)
            cost += answer['cost']; mandatory += len(required); intersection += len(essential)
        count, family_cost = np.nan, np.nan
        try:
            catalogue = r22.build_catalogue(g, strength, tol=tol, universes=['extended'], limit=LIMIT)
            fam = r22.families(g, catalogue, universes=['extended'], modes=['joint_full_strength'])
            joint = r22.total(fam, 'extended', 'joint_full_strength', g['classes'])
            assert len(joint['mandatory']) == mandatory and len(joint['intersection']) == intersection
            count, family_cost = joint['count'], joint['cost']
            assert family_cost == cost
        except r22.TooManyCycles as error:
            print(f'tau {tau_pp}: witness enumeration exceeded {LIMIT} for one target ({error})', flush=True)
        n_any = sum(k[2] == 'any' for k in targets)
        rows.append(dict(tau_pp=tau_pp, n_targets=len(targets), n_any_targets=n_any, n_four_targets=len(targets) - n_any,
                         n_targets_retained_by_rule_union=sum(t['actual_satisfied'] for t in targets.values()),
                         n_boundary_near_ties=boundary, minimum_joint_budget=cost, n_F=mandatory, n_I_star=intersection,
                         n_optimal_sets=count))
        print(rows[-1], flush=True)
    result = pd.DataFrame(rows)
    result['n_optimal_sets'] = result.n_optimal_sets.astype('Int64')
    base = result.loc[result.tau_pp.eq(0)].iloc[0]
    if v == 'raw':
        assert (base.minimum_joint_budget, base.n_F, base.n_I_star, base.n_optimal_sets) == (2224, 2185, 2202, 2048)
    family = json.loads((out_dir(v) / 'check_strength_restoration.json').read_text(encoding='utf-8'))
    assert family['variant'] == v and family['all_match']
    for name in ('minimum_joint_budget', 'n_F', 'n_I_star', 'n_optimal_sets'):
        assert base[name] == family['family_summary'][name], (name, base[name], family['family_summary'][name])
    result.to_csv(out_dir(v) / 'restoration_tolerance.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
