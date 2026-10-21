"""Complete family of minimum record sets for the restoration goals, and the block check.

For every positive full target, all eligible witnesses (simple cycles at full rho; four-cycles at full
rho_4) give candidate missing-record sets. F = union over targets of the intersection of their candidate
sets (required in every successful restoration); exact positive-CNF components then give the
intersection I*, union U* and count of all minimum-size sets.
Writes out/restoration_joint_family.csv and out/check_strength_restoration.json.
"""
import argparse
from importlib import import_module
from itertools import combinations, islice, product
import json
from math import prod
import time

import networkx as nx
import numpy as np
import pandas as pd
from m2common import out_dir

load_graph = import_module('20_constraint_strength').load_graph
MODES = ['global_full_rho', 'local_full_rho4', 'joint_full_strength']
UNIVERSES = ['extended', 'frozen']


def minimal(sets):
    kept = []
    for value in sorted(set(map(frozenset, sets)), key=lambda s: (len(s), tuple(sorted(s)))):
        if not any(old <= value for old in kept):
            kept.append(value)
    return kept


def cnf_for_candidates(candidates):
    if any(not value for value in candidates):
        return []
    clauses = [frozenset()]
    for candidate in candidates:
        clauses = minimal(clause | {record} for clause in clauses for record in candidate)
    return clauses


def components(clauses):
    graph = nx.Graph()
    for clause in clauses:
        vertices = sorted(clause)
        graph.add_nodes_from(vertices)
        graph.add_edges_from(zip(vertices, vertices[1:]))
    result = []
    for vertices in sorted(nx.connected_components(graph), key=lambda c: min(c)):
        selected = [clause for clause in clauses if clause & vertices]
        assert all(clause <= vertices for clause in selected)
        result.append((sorted(vertices), selected))
    return result


def exhaustive_minimum_covers(vertices, clauses):
    index = {vertex: i for i, vertex in enumerate(vertices)}
    masks = [sum(1 << index[v] for v in clause) for clause in clauses]
    for cardinality in range(len(vertices) + 1):
        solutions = []
        for subset in combinations(range(len(vertices)), cardinality):
            value = sum(1 << i for i in subset)
            if all(value & clause for clause in masks):
                solutions.append(frozenset(vertices[i] for i in subset))
        if solutions:
            return cardinality, solutions
    raise AssertionError('Positive nonempty clauses must be feasible')


def solve_families(targets):
    targets = [minimal(values) for values in targets]
    assert all(targets)
    mandatory = set().union(*(set.intersection(*map(set, values)) for values in targets)) if targets else set()
    residual = [minimal(value - mandatory for value in values) for values in targets]
    clauses = minimal(clause for values in residual for clause in cnf_for_candidates(values))
    factors = []
    for vertices, selected in components(clauses):
        cost, solutions = exhaustive_minimum_covers(vertices, selected)
        factors.append(dict(vertices=vertices, cost=cost, solutions=sorted(solutions, key=lambda s: sorted(s)),
                            intersection=set.intersection(*map(set, solutions)), union=set.union(*map(set, solutions))))
    cost = len(mandatory) + sum(factor['cost'] for factor in factors)
    intersection = mandatory | set().union(*(factor['intersection'] for factor in factors))
    union = mandatory | set().union(*(factor['union'] for factor in factors))
    relevant = set().union(*(set().union(*map(set, values)) for values in targets)) if targets else set()
    return dict(mandatory=mandatory, factors=factors, cost=cost, count=prod(len(f['solutions']) for f in factors),
                intersection=intersection, union=union, relevant=relevant)


def canonical(nodes, lookup):
    first = nodes.index(min(nodes))
    nodes = nodes[first:] + nodes[:first]
    return nodes, [lookup[a, b] for a, b in zip(nodes, nodes[1:] + nodes[:1])]


class TooManyCycles(Exception):
    pass


def enumerate_cycles(positions, kind, g, limit=None):
    lookup = {(int(g['source'][p]), int(g['target'][p])): int(p) for p in positions}
    assert len(lookup) == len(positions)
    if kind == 'any':
        graph = nx.DiGraph()
        graph.add_edges_from(lookup)
        cycles = list(islice(nx.simple_cycles(graph), limit + 1 if limit else None))
        if limit and len(cycles) > limit:
            raise TooManyCycles(len(cycles))
        result = [canonical(list(cycle), lookup) for cycle in cycles]
    else:
        pairs = {(int(g['node1'][p]), int(g['node2'][p])): int(p) for p in positions}
        lefts, rights = sorted(set(g['node1'][positions])), sorted(set(g['node2'][positions]))
        result = []
        for u, v in combinations(rights, 2):
            forward, backward = [], []
            for a in lefts:
                p, q = pairs.get((a, u)), pairs.get((a, v))
                if p is None or q is None:
                    continue
                if g['d'][p] > 0 and g['d'][q] < 0:
                    forward.append(a)
                elif g['d'][p] < 0 and g['d'][q] > 0:
                    backward.append(a)
            for a, b in product(forward, backward):
                result.append(canonical([int(u), int(a), int(v), int(b)], lookup))
            if limit and len(result) > limit:
                raise TooManyCycles(len(result))
    result.sort(key=lambda value: tuple(sorted(value[1])))
    return result


def build_catalogue(g, strength, tol=0.0, universes=UNIVERSES, limit=None):
    """tol relaxes each target to full value - tol (0-1 scale); limit caps witnesses per target."""
    targets = {}
    for universe in universes:
        for edge in g['edges'].itertuples(index=False):
            full = strength.loc[(universe, 'full', edge.edge_id)]
            positions = g['groups'].get(int(edge.edge_position), np.empty(0, int))
            if universe == 'frozen':
                positions = positions[g['frozen'][positions]]
            for kind, threshold in [('any', full.rho - tol), ('four', full.rho4 - tol)]:
                if not threshold > 0:
                    continue
                eligible = positions[g['weight'][positions] >= threshold]
                cycles = enumerate_cycles(eligible, kind, g, limit)
                assert cycles
                sets = []
                for nodes, p in cycles:
                    bottleneck = g['weight'][p].min()
                    assert len(nodes) == len(p) >= 4 and (bottleneck == threshold if tol == 0 else bottleneck >= threshold)
                    records = np.unique(np.r_[g['a'][p], g['b'][p]])
                    assert len(records) == 2 * len(p)
                    sets.append(frozenset(int(r) for r in records if not g['actual'][r]))
                targets[universe, edge.edge_id, kind] = dict(candidates=minimal(sets), rxn_class=edge.rxn_class,
                                                             n_witnesses=len(cycles))
    return targets


def families(g, targets, universes=UNIVERSES, modes=MODES):
    result = {}
    for universe, mode in product(universes, modes):
        kinds = ['any'] if mode == 'global_full_rho' else ['four'] if mode == 'local_full_rho4' else ['any', 'four']
        for cls in g['classes']:
            keys = [k for k, t in targets.items() if k[0] == universe and k[2] in kinds and t['rxn_class'] == cls]
            result[universe, mode, cls] = solve_families([targets[k]['candidates'] for k in keys])
    return result


def total(fam, universe, mode, classes):
    parts = [fam[universe, mode, cls] for cls in classes]
    return dict(cost=sum(p['cost'] for p in parts), count=prod(p['count'] for p in parts),
                mandatory=set().union(*(p['mandatory'] for p in parts)),
                intersection=set().union(*(p['intersection'] for p in parts)),
                union=set().union(*(p['union'] for p in parts)), relevant=set().union(*(p['relevant'] for p in parts)),
                factors=[f for p in parts for f in p['factors']])


def entry(name, manuscript, recomputed, digits=None):
    if recomputed is None:
        match = False
    elif digits is None:
        match = manuscript == recomputed
    else:
        match = round(float(recomputed), digits) == manuscript
    return dict(name=name, manuscript=manuscript, recomputed=recomputed, match=bool(match))


def strength_checks(strength):
    s = strength.reset_index()
    ext = s.loc[s.universe.eq('extended')]
    full = ext.loc[ext.recordset_id.eq('full')].set_index('edge_id')
    checks = []
    cyclic = full.rho.gt(0)
    checks.append(entry('contrasts with rho > 0, full measured data (extended, 845,328 duals)', 1446, int(cyclic.sum())))
    act = ext.loc[ext.recordset_id.eq('actual')].set_index('edge_id').loc[full.index]
    keep = cyclic & act.rho.eq(full.rho)
    weaker = cyclic & act.rho.gt(0) & act.rho.lt(full.rho)
    acyclic = cyclic & act.rho.eq(0)
    nodual = cyclic & act.rho.isna()
    assert int(cyclic.sum()) == int(keep.sum() + weaker.sum() + acyclic.sum() + nodual.sum()) > 0
    checks += [entry('rule union keeps full rho', 673, int(keep.sum())),
               entry('rule union weaker rho > 0', 617, int(weaker.sum())),
               entry('rule union measured but acyclic (rho = 0)', 119, int(acyclic.sum())),
               entry('rule union without any retained dual', 37, int(nodual.sum()))]
    loss = 50 * (full.rho - act.rho)[weaker]
    assert len(loss) > 0
    checks += [entry('median loss of epsilon = rho/2 among weakened (pp)', 1.62, float(loss.median()), 2),
               entry('weakened losing >= 5 pp of epsilon = rho/2', 81, int(loss.ge(5).sum()))]
    gap = cyclic & full.rho4.lt(full.rho)
    checks += [entry('contrasts with rho_4 < rho (incl. no 4-cycle), full data', 263, int(gap.sum())),
               entry('classes containing them', 10, int(full.loc[gap, 'rxn_class'].nunique())),
               entry('of them, only longer cycles (rho_4 = 0)', 20, int((gap & full.rho4.eq(0)).sum()))]
    attains = full.strength_state.eq('four_cycle_attains_global')
    for name in ['Q', 'actual']:
        sel = ext.loc[ext.recordset_id.eq(name)].set_index('edge_id').loc[full.index]
        new = attains & sel.strength_state.isin(['four_cycle_below_global', 'long_cycle_only'])
        checks.append(entry(f'full 4-cycle attained rho, {name} makes rho_4 < rho', 120, int(new.sum())))
    return checks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    args = ap.parse_args()
    v = args.variant
    out = out_dir(v)
    start = time.time()
    g = load_graph(v)
    strength = pd.read_csv(out / 'constraint_strength_edges.csv', float_precision='round_trip',
                           keep_default_na=False, na_values=['']).set_index(['universe', 'recordset_id', 'edge_id'])
    targets = build_catalogue(g, strength)
    print(f'catalogue {len(targets)} targets, {sum(t["n_witnesses"] for t in targets.values())} witnesses, '
          f'{time.time() - start:.0f}s', flush=True)
    fam = families(g, targets)
    print(f'families {time.time() - start:.0f}s', flush=True)
    sol = pd.read_csv(out / 'restoration_class_solutions.csv')
    joint = total(fam, 'extended', 'joint_full_strength', g['classes'])
    chosen = {}
    for (universe, mode), part in sol.groupby(['universe', 'mode_id']):
        chosen[universe, mode] = {r for text in part.selected_record_positions_json for r in json.loads(text)}
        t = total(fam, universe, mode, g['classes'])
        sel = chosen[universe, mode]
        assert len(sel) == t['cost'] and t['mandatory'] <= t['intersection'] <= sel <= t['union']
        for factor in t['factors']:
            assert frozenset(sel & set(factor['vertices'])) in factor['solutions']
    returned = chosen['extended', 'joint_full_strength']
    with np.load(out / 'restoration_joint_returned_mask.npz') as packed:
        mask = np.unpackbits(packed['mask'], bitorder='big')[:len(g['raw'])].astype(bool)
    assert np.array_equal(np.flatnonzero(mask & ~g['actual']), sorted(returned))

    comparison_returned = None

    variable = [f for f in joint['factors'] if len(f['solutions']) > 1]
    record_id = g['raw'].record_id.to_numpy(dtype=str)
    rxn = g['raw'].rxn_class.to_numpy(dtype=str)
    rows = [dict(row_type='record', set_id=np.nan, record_position=r, record_id=record_id[r], rxn_class=rxn[r],
                 in_F=r in joint['mandatory'], in_I_star=r in joint['intersection'], in_U_star=r in joint['union'],
                 in_returned_optimum=r in returned,
                 in_comparison_returned_optimum=(r in comparison_returned if comparison_returned is not None else None))
            for r in sorted(joint['relevant'])]
    optional = joint['union'] - joint['intersection']
    returned_id = comparison_id = None
    for set_id, choice in enumerate(product(*(f['solutions'] for f in variable))):
        records = sorted(set().union(*choice) - joint['intersection'])
        full_set = joint['intersection'] | set(records)
        assert len(full_set) == joint['cost'] and set(records) <= optional
        returned_id = set_id if full_set == returned else returned_id
        comparison_id = set_id if full_set == comparison_returned else comparison_id
        rows += [dict(row_type='optimum', set_id=set_id, record_position=r, record_id=record_id[r], rxn_class=rxn[r])
                 for r in records]
    assert set_id + 1 == joint['count']
    family = pd.DataFrame(rows)
    family['set_id'] = family.set_id.astype('Int64')
    family.to_csv(out / 'restoration_joint_family.csv', index=False)

    checks = strength_checks(strength)
    cost = {(u, m): int(p.minimum_additions.sum()) for (u, m), p in sol.groupby(['universe', 'mode_id'])}
    gaps = sol.loc[sol.solver_status.ne(0) | sol.solver_gap.abs().gt(1e-9)]
    assert gaps.empty
    checks += [entry('joint-restoration minimum, full rho (records)', 1992, cost['extended', 'global_full_rho']),
               entry('joint-restoration minimum, full rho_4 (records)', 1861, cost['extended', 'local_full_rho4']),
               entry('joint-restoration minimum, both (records)', 2224, cost['extended', 'joint_full_strength'])]
    by_class = sol.loc[sol.mode_id.eq('joint_full_strength')].pivot(index='rxn_class', columns='universe',
                                                                     values='minimum_additions')
    for cls, value in [('Acylation', 470), ('Buchwald-Hartwig', 322), ('Suzuki', 280), ('Kumada', 269), ('Ullmann', 253)]:
        checks.append(entry(f'joint minimum, {cls}', value, int(by_class.loc[cls, 'extended'])))
    diff = (by_class.extended - by_class.frozen)
    checks += [entry('joint minimum without mixed-pair noncore duals (frozen domain)', 2195,
                     cost['frozen', 'joint_full_strength']),
               entry('classes carrying the extended - frozen difference', ['Kumada'],
                     sorted(diff.index[diff.ne(0)].tolist()))]
    checks += [entry('F: records in every successful joint restoration', 2185, len(joint['mandatory'])),
               entry('I*: records in every minimum-size joint set', 2202, len(joint['intersection'])),
               entry('I* - F', 17, len(joint['intersection']) - len(joint['mandatory'])),
               entry('records each optimum picks beyond I*', 22, joint['cost'] - len(joint['intersection'])),
               entry('U*: union of all minimum-size joint sets', 2246, len(joint['union'])),
               entry('distinct minimum-size joint sets', 2048, joint['count']),
               entry('family minimum equals MILP minimum (joint)', 2224, joint['cost'])]
    family_summary = dict(minimum_joint_budget=joint['cost'], n_F=len(joint['mandatory']),
                          n_I_star=len(joint['intersection']), n_U_star=len(joint['union']),
                          n_optimal_sets=joint['count'])
    if v != 'raw':
        checks = [entry(f'{u}/{m}: family minimum equals same-variant MILP minimum', cost[u, m],
                        total(fam, u, m, g['classes'])['cost']) for u, m in sorted(cost)]
        checks.append(entry('returned mask contains the same-variant optimum records', True,
                            bool(np.array_equal(np.flatnonzero(mask & ~g['actual']), sorted(returned)))))
    result = dict(block='strength_restoration', variant=v, checks=checks,
                  all_match=all(c['match'] for c in checks),
                  family_summary=family_summary,
                  returned_optimum_set_id=returned_id, comparison_returned_optimum_set_id=comparison_id,
                  family_targets=len([k for k in targets if k[0] == 'extended']),
                  family_components_with_alternatives=len(variable))
    (out / 'check_strength_restoration.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    for c in checks:
        print(c)
    print('all_match', result['all_match'], 'returned set', returned_id, 'comparison set', comparison_id, f'{time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
