"""Continuous constraint strength of every recipe contrast: rho (largest bottleneck |d| over directed
cycles of the partner graph) and rho_4 (the same over four-cycles), for the full measured duals, the
rule union and Q, in the extended (all 845,328 duals) and frozen (Q/O only, 843,057) domains.

Duals are ordered by domain membership, reaction class, substrate pair, cohort
and contrast to give a deterministic variable order for downstream optimization.
"""
import argparse
from heapq import heapify, heappop, heappush
import time

import numpy as np
import pandas as pd
from m2common import data_dir, out_dir

RECORDSETS = ['Q', 'actual', 'full']
UNIVERSES = ['extended', 'frozen']


def load_graph(v=None):
    out = out_dir(v)
    raw = pd.read_parquet(data_dir(v) / 'observations.parquet',
                          columns=['record_id', 'rxn_class', 'analysis_valid', 'qualified_row'])
    edges = pd.read_csv(out / 'recipe_edges.csv', usecols=['rxn_class', 'edge_id', 'condition_A', 'condition_B'],
                        keep_default_na=False)
    edges.insert(0, 'edge_position', np.arange(len(edges)))
    dual = pd.read_parquet(out / 'recipe_duals.parquet', columns=[
        'rxn_class', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'edge_id', 'record_position_A', 'record_position_B',
        'record_id_A', 'record_id_B', 'yield_A', 'yield_B', 'identity_A', 'identity_B', 'd', 'core_outside_domain'])
    dual['cohort'] = np.where(dual.identity_A.eq('Q') & dual.identity_B.eq('Q'), 'Q',
                              np.where(dual.identity_A.eq('O') & dual.identity_B.eq('O'), 'O', 'mixed_extension'))
    dual['is_frozen'] = dual.cohort.ne('mixed_extension')
    assert np.array_equal(dual.is_frozen, dual.core_outside_domain)
    dual = dual.sort_values(['is_frozen', 'rxn_class', 'subpair', 'cohort', 'edge_id'],
                            ascending=[False, True, True, True, True], kind='stable').reset_index(drop=True)
    a, b = dual.record_position_A.to_numpy(int), dual.record_position_B.to_numpy(int)
    edge_code = pd.Index(edges.edge_id).get_indexer(dual.edge_id)
    assert (edge_code >= 0).all()
    endpoints = np.r_[edge_code * len(raw) + a, edge_code * len(raw) + b]
    assert len(np.unique(endpoints)) == len(endpoints)
    d = dual.d.to_numpy(float)
    assert np.array_equal(d, dual.yield_A.to_numpy() - dual.yield_B.to_numpy())
    labels = pd.MultiIndex.from_arrays([np.r_[dual.rxn_class.to_numpy(), dual.rxn_class.to_numpy()],
                                        np.r_[np.ones(len(dual), int), np.full(len(dual), 2, int)],
                                        np.r_[dual.sub_1_smiles.to_numpy(object), dual.sub_2_smiles.to_numpy(object)]])
    codes, _ = pd.factorize(labels, sort=True)
    node1, node2 = codes[:len(dual)], codes[len(dual):]
    with np.load(out / 'rule_masks.npz') as packed:
        actual = np.unpackbits(packed['rule_union'], bitorder='big')[:len(raw)].astype(bool)
    assert v != 'raw' or actual.sum() == 277192
    valid, q = raw.analysis_valid.to_numpy(bool), raw.qualified_row.to_numpy(bool)
    assert (valid[a] & valid[b]).all()
    groups = dual.groupby(edge_code, sort=False).indices
    return dict(raw=raw, edges=edges, dual=dual, a=a, b=b, d=d, weight=np.abs(d), node1=node1, node2=node2,
                source=np.where(d > 0, node2, node1), target=np.where(d > 0, node1, node2),
                actual=actual, valid=valid, frozen=dual.is_frozen.to_numpy(bool),
                masks=dict(Q=q, actual=actual, full=valid),
                groups={int(k): np.asarray(v, int) for k, v in groups.items()},
                classes=sorted(set(raw.rxn_class) - {'Finkelstein'}))


def topology(positions, source, target):
    adjacency, degree = {}, {}
    for p in positions:
        a, b = int(source[p]), int(target[p])
        adjacency.setdefault(a, []).append(b)
        degree.setdefault(a, 0)
        degree[b] = degree.get(b, 0) + 1
    queue = [v for v in degree if degree[v] == 0]
    heapify(queue)
    order = []
    while queue:
        a = heappop(queue)
        order.append(a)
        for b in adjacency.get(a, []):
            degree[b] -= 1
            if degree[b] == 0:
                heappush(queue, b)
    return order if len(order) == len(degree) else None


def whole_rho(positions, source, target, weight):
    positive = positions[weight[positions] > 0]
    if topology(positive, source, target) is not None:
        return 0.0
    values = np.unique(weight[positive])
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if topology(positive[weight[positive] >= values[middle]], source, target) is None:
            low = middle + 1
        else:
            high = middle
    assert low > 0
    rho = float(values[low - 1])
    assert topology(positive[weight[positive] > rho], source, target) is not None
    return rho


def four_certificate(positions, node1, node2, d):
    rho4, rectangles = 0.0, 0
    if len(positions) == 0:
        return rho4, rectangles
    aa, ia = np.unique(node1[positions], return_inverse=True)
    bb, ib = np.unique(node2[positions], return_inverse=True)
    assert len(np.unique(ia * len(bb) + ib)) == len(positions)
    matrix = np.full((len(aa), len(bb)), np.nan)
    matrix[ia, ib] = d[positions]
    for i in range(len(aa) - 1):
        first, others = matrix[i], matrix[i + 1:]
        common = np.isfinite(first) & np.isfinite(others)
        counts = common.sum(axis=1)
        rectangles += int(np.sum(counts * (counts - 1) // 2))
        strength = np.where(common, np.minimum(np.abs(first), np.abs(others)), 0)
        positive = np.where((first > 0) & (others < 0), strength, 0)
        negative = np.where((first < 0) & (others > 0), strength, 0)
        best = float(np.minimum(positive.max(axis=1), negative.max(axis=1)).max())
        rho4 = max(rho4, best)
    return rho4, rectangles


def strength_state(n_duals, rectangles, rho4, rho):
    if n_duals == 0:
        return 'no_measured_support'
    if rho == 0:
        return 'no_complete_rectangle' if rectangles == 0 else 'rectangle_supported_acyclic'
    if rho4 == 0:
        return 'long_cycle_only'
    return 'four_cycle_attains_global' if rho4 == rho else 'four_cycle_below_global'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='clfzero')
    v = ap.parse_args().variant
    start = time.time()
    g = load_graph(v)
    a, b, d, weight = g['a'], g['b'], g['d'], g['weight']
    rows, cache = [], {}
    for universe in UNIVERSES:
        eligible = np.ones(len(d), bool) if universe == 'extended' else g['frozen']
        for recordset in RECORDSETS:
            mask = g['masks'][recordset]
            active = eligible & mask[a] & mask[b]
            for edge in g['edges'].itertuples(index=False):
                positions = g['groups'].get(edge.edge_position, np.empty(0, int))
                positions = positions[active[positions]]
                key = positions.tobytes()
                if key not in cache:
                    rho = whole_rho(positions, g['source'], g['target'], weight)
                    rho4, rectangles = four_certificate(positions, g['node1'], g['node2'], d)
                    assert 0 <= rho4 <= rho
                    cache[key] = (rho, rho4, rectangles)
                rho, rho4, rectangles = cache[key]
                supported = len(positions) > 0
                rows.append(dict(universe=universe, recordset_id=recordset, edge_position=edge.edge_position,
                                 edge_id=edge.edge_id, rxn_class=edge.rxn_class, n_duals=len(positions),
                                 n_exact_zero=int(np.count_nonzero(d[positions] == 0)),
                                 n_complete_rectangles=rectangles,
                                 rho=rho if supported else np.nan, rho4=rho4 if supported else np.nan,
                                 strength_state=strength_state(len(positions), rectangles, rho4, rho)))
            print(universe, recordset, 'graphs', len(cache), f'{time.time() - start:.0f}s', flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 2 * 3 * len(g['edges'])
    result.to_csv(out_dir(v) / 'constraint_strength_edges.csv', index=False)
    full = result.loc[result.universe.eq('extended') & result.recordset_id.eq('full')]
    print('extended full cyclic', int(full.rho.gt(0).sum()), f'{time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
