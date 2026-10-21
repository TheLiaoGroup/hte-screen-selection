"""Minimum restoration of original records: starting from the rule union, the fewest added measured
records (shared across all targets of a reaction class) that recover every contrast's full rho, full
rho_4, or both, solved as binary MILPs (scipy.optimize.milp / HiGHS, 180 s, zero gap).

Strengths come from out/constraint_strength_edges.csv (script 20).
"""
import argparse
from importlib import import_module
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from m2common import out_dir

load_graph = import_module('20_constraint_strength').load_graph
MODES = ['global_full_rho', 'local_full_rho4', 'joint_full_strength']
UNIVERSES = ['extended', 'frozen']


def encode(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True)


def optimize(graphs, missing, source, target):
    records = sorted({r for graph in graphs for p in graph['positions'] for r in missing[p]})
    if not graphs:
        return dict(feasible=True, cost=0, records=[], status=0, objective=0., dual_bound=0., gap=0.,
                    n_variables=0, n_constraints=0, n_record_variables=0, n_arc_variables=0, n_binary_arc_variables=0)
    record_index = {r: i for i, r in enumerate(records)}
    arcs = [(g, int(p)) for g, graph in enumerate(graphs) for p in graph['positions']]
    nr, nv = len(records), len(records) + len(arcs)
    lower, upper, rr, cc, vv, balance, by_goal = [], [], [], [], [], {}, [[] for _ in graphs]
    bounds_lower, bounds_upper = np.zeros(nv), np.ones(nv)
    integral = np.zeros(nv, np.int8)
    integral[:nr] = 1

    def add(coefficients, lo, hi):
        row = len(lower)
        for variable, coefficient in coefficients.items():
            rr.append(row); cc.append(variable); vv.append(coefficient)
        lower.append(lo); upper.append(hi)

    for j, (g, p) in enumerate(arcs):
        column = nr + j
        by_goal[g].append(column)
        for record in missing[p]:
            add({column: 1, record_index[record]: -1}, -np.inf, 0)
        balance.setdefault((g, int(source[p])), {})[column] = 1
        balance.setdefault((g, int(target[p])), {})[column] = -1
        if graphs[g]['kind'] == 'four':
            integral[column] = 1
        elif graphs[g].get('critical_arc') == p:
            bounds_lower[column] = 1
    for coefficients in balance.values():
        add(coefficients, 0, 0)
    for g, columns in enumerate(by_goal):
        add({column: 1 for column in columns}, 4 if graphs[g]['kind'] == 'four' else 1,
            4 if graphs[g]['kind'] == 'four' else np.inf)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), nv)).tocsr()
    problem = dict(c=np.r_[np.ones(nr), np.zeros(nv - nr)], integrality=integral,
                   bounds=Bounds(bounds_lower, bounds_upper), constraints=LinearConstraint(matrix, lower, upper))
    answer = milp(**problem, options={'time_limit': 180, 'mip_rel_gap': 0})
    if answer.status == 2:
        # HiGHS presolve can misreport infeasibility on these models; infeasibility must survive a presolve-free solve.
        answer = milp(**problem, options={'time_limit': 180, 'mip_rel_gap': 0, 'presolve': False})
    if answer.status == 2:
        return dict(feasible=False, cost=None, records=[], status=2)
    assert answer.status == 0, (answer.status, answer.message, getattr(answer, 'mip_dual_bound', None))
    chosen = [record for record, value in zip(records, answer.x[:nr]) if value > .5]
    assert abs(answer.fun - len(chosen)) < 1e-7 and abs(answer.mip_dual_bound - len(chosen)) < 1e-7
    assert answer.mip_gap < 1e-10
    assert brute_success(graphs, set(chosen), missing, source, target)
    return dict(feasible=True, cost=len(chosen), records=chosen, status=int(answer.status),
                objective=float(answer.fun), dual_bound=float(answer.mip_dual_bound), gap=float(answer.mip_gap),
                n_variables=nv, n_constraints=len(lower), n_record_variables=nr, n_arc_variables=len(arcs),
                n_binary_arc_variables=int(integral[nr:].sum()))


def brute_success(graphs, chosen, missing, source, target):
    for goal in graphs:
        graph = nx.DiGraph((int(source[p]), int(target[p])) for p in goal['positions']
                           if set(missing[p]) <= chosen)
        cycles = nx.simple_cycles(graph)
        if not any(goal['kind'] == 'any' or len(cycle) == 4 for cycle in cycles):
            return False
    return True


def missing_records(positions, g):
    endpoints = np.r_[g['a'][positions], g['b'][positions]]
    return sorted(set(int(r) for r in endpoints if not g['actual'][r]))


def four_scan(positions, g, collect=False):
    aa, ia = np.unique(g['node1'][positions], return_inverse=True)
    bb, ib = np.unique(g['node2'][positions], return_inverse=True)
    sign = np.zeros((len(aa), len(bb)), np.int8)
    lookup = np.full(sign.shape, -1, int)
    sign[ia, ib], lookup[ia, ib] = np.sign(g['d'][positions]), positions
    used, choices, n_four = set(), set(), 0
    for i in range(len(aa) - 1):
        positive = (sign[i] > 0) & (sign[i + 1:] < 0)
        negative = (sign[i] < 0) & (sign[i + 1:] > 0)
        both = positive.any(axis=1) & negative.any(axis=1)
        n_four += int((positive.sum(axis=1) * negative.sum(axis=1)).sum())
        js, bs = np.where((positive | negative) & both[:, None])
        used.update(lookup[i, bs]); used.update(lookup[i + 1 + js, bs])
        if collect:
            for k in np.flatnonzero(both):
                j = i + 1 + int(k)
                for u in np.flatnonzero(positive[k]):
                    for v in np.flatnonzero(negative[k]):
                        chosen = lookup[[j, i, i, j], [u, u, v, v]]
                        choices.add(frozenset(missing_records(chosen, g)))
    return np.array(sorted(used), int), choices, n_four


def build_targets(g, strength, tol=0.0, universes=UNIVERSES):
    """tol relaxes each target to full value - tol (0-1 scale); targets with a relaxed value <= 0 are dropped."""
    source, target, weight = g['source'], g['target'], g['weight']
    targets, missing = {}, {}
    for universe in universes:
        for edge in g['edges'].itertuples(index=False):
            full = strength.loc[(universe, 'full', edge.edge_id)]
            actual = strength.loc[(universe, 'actual', edge.edge_id)]
            positions = g['groups'].get(int(edge.edge_position), np.empty(0, int))
            if universe == 'frozen':
                positions = positions[g['frozen'][positions]]
            assert len(positions) == full.n_duals
            same = bool(full.rho > 0 and full.rho == full.rho4)
            for kind, tau, retained in [('any', full.rho - tol, actual.rho), ('four', full.rho4 - tol, actual.rho4)]:
                if not tau > 0:
                    continue
                p = positions[weight[positions] >= tau]
                graph = nx.DiGraph(zip(source[p], target[p]))
                components = {node: i for i, part in enumerate(nx.strongly_connected_components(graph))
                              if len(part) > 1 for node in part}
                p = p[np.array([u in components and components[u] == components.get(v, -1)
                                for u, v in zip(source[p], target[p])])]
                satisfied = retained >= tau
                critical, mandatory = None, set()
                if kind == 'four':
                    p, choices, n_four = four_scan(p, g, collect=not satisfied)
                    assert n_four > 0
                    if not satisfied:
                        assert choices and all(choices)
                        mandatory = set.intersection(*(set(choice) for choice in choices))
                else:
                    criticals = p[weight[p] == tau]
                    if tol == 0:
                        stronger = nx.DiGraph(zip(source[p[weight[p] > tau]], target[p[weight[p] > tau]]))
                        assert nx.is_directed_acyclic_graph(stronger)
                    if tol == 0 and len(criticals) == 1:
                        critical = int(criticals[0])
                    if not satisfied:
                        graph = nx.DiGraph(zip(source[p], target[p]))
                        for arc in p:
                            endpoints = missing_records(np.array([arc]), g)
                            if endpoints:
                                graph.remove_edge(int(source[arc]), int(target[arc]))
                                if nx.is_directed_acyclic_graph(graph):
                                    mandatory.update(endpoints)
                                graph.add_edge(int(source[arc]), int(target[arc]))
                assert len(p) >= 4
                for arc in p:
                    missing[int(arc)] = tuple(missing_records(np.array([arc]), g))
                targets[(universe, edge.edge_id, kind)] = dict(
                    kind=kind, positions=p, critical_arc=critical, mandatory=mandatory, universe=universe,
                    edge_id=edge.edge_id, rxn_class=edge.rxn_class, threshold=float(tau),
                    actual_satisfied=bool(satisfied), same_strength=same)
        print(encode(dict(stage='catalogue', universe=universe, cumulative_positive_targets=len(targets))), flush=True)
    return targets, missing


def requested_goals(targets, universe, mode, cls):
    goals = []
    for (domain, _, kind), goal in targets.items():
        if domain != universe or goal['rxn_class'] != cls:
            continue
        if (mode == 'global_full_rho' and kind != 'any') or (mode == 'local_full_rho4' and kind != 'four'):
            continue
        goals.append(goal)
    return goals


def solve_modes(g, targets, missing):
    rows = []
    for universe in UNIVERSES:
        for mode in MODES:
            for cls in g['classes']:
                start = time.time()
                goals = requested_goals(targets, universe, mode, cls)
                unresolved = [goal for goal in goals if not goal['actual_satisfied']]
                graphs = [goal for goal in unresolved if not (mode == 'joint_full_strength'
                          and goal['kind'] == 'any' and goal['same_strength'])]
                answer = optimize(graphs, missing, g['source'], g['target'])
                assert answer['feasible']
                mandatory = set().union(*(goal['mandatory'] for goal in goals))
                assert mandatory <= set(answer['records'])
                assert brute_success(goals, set(answer['records']), missing, g['source'], g['target'])
                assert g['valid'][answer['records']].all() and not g['actual'][answer['records']].any()
                assert g['raw'].rxn_class.iloc[answer['records']].eq(cls).all()
                rows.append(dict(universe=universe, mode_id=mode, rxn_class=cls,
                                 status='no_target' if not goals else 'already_retained' if not graphs else 'optimal',
                                 n_required_any=sum(goal['kind'] == 'any' for goal in goals),
                                 n_required_four=sum(goal['kind'] == 'four' for goal in goals),
                                 n_optimization_targets=len(graphs), n_actual_satisfied=len(goals) - len(unresolved),
                                 minimum_additions=answer['cost'], n_mandatory_records=len(mandatory),
                                 solver_status=answer['status'], solver_objective=answer['objective'],
                                 solver_dual_bound=answer['dual_bound'], solver_gap=answer['gap'],
                                 n_variables=answer['n_variables'], n_constraints=answer['n_constraints'],
                                 selected_record_positions_json=encode(answer['records']),
                                 mandatory_record_positions_json=encode(sorted(mandatory))))
            total = sum(row['minimum_additions'] for row in rows[-len(g['classes']):])
            print(encode(dict(stage='optimum', universe=universe, mode=mode, cost=total)), flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    v = ap.parse_args().variant
    start = time.time()
    g = load_graph(v)
    strength = pd.read_csv(out_dir(v) / 'constraint_strength_edges.csv', float_precision='round_trip',
                           keep_default_na=False, na_values=['']).set_index(['universe', 'recordset_id', 'edge_id'])
    targets, missing = build_targets(g, strength)
    print(f'targets built {time.time() - start:.0f}s', flush=True)
    solutions = solve_modes(g, targets, missing)
    solutions.to_csv(out_dir(v) / 'restoration_class_solutions.csv', index=False)
    joint = solutions.loc[solutions.universe.eq('extended') & solutions.mode_id.eq('joint_full_strength')]
    added = sorted(r for text in joint.selected_record_positions_json for r in json.loads(text))
    mask = g['actual'].copy()
    assert not mask[added].any()
    mask[added] = True
    np.savez_compressed(out_dir(v) / 'restoration_joint_returned_mask.npz',
                        mask=np.packbits(mask, bitorder='big'), n_records=np.array(len(mask)),
                        added_record_position=np.array(added, int),
                        added_record_id=g['raw'].record_id.to_numpy(dtype=str)[added])
    print(f'joint extended additions {len(added)}; total {time.time() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
