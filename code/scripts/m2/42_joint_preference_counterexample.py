"""Joint minimum record additions for the 179 preference graphs and the three counterexample goals.

Optimization uses nonzero circulation per unresolved preference graph, AND/OR
counterexample witnesses and per-class HiGHS MILP (180 s, zero gap).
The 179 graphs are the recipe contrasts whose
qualified-only 5 pp strong graph is a nonempty DAG while the full frozen graph is cyclic
Specs: preference only, and
preference + all 5 pp counterexample targets of all three goals.
"""
import argparse
from fractions import Fraction
import json
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from m2common import out_dir

GOALS = ["structural", "partner", "held_recipe"]
TOL = 1e-12


def encode(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def optimize(graphs, targets, quotas, missing, sources, destinations):
    candidate_sets = [tuple(tuple(c) for c in json.loads(s)) for s in targets.candidate_sets_json]
    positions = sorted({r for graph in graphs for p in graph for r in missing[p]}
                       | {r for choices in candidate_sets for choice in choices for r in choice})
    if not graphs and not len(targets):
        return dict(cost=0, records=[], status=0, gap=0., objective=0., dual_bound=0.)
    ri = {r: i for i, r in enumerate(positions)}
    flows = [(g, int(p)) for g, graph in enumerate(graphs) for p in graph]
    witnesses = [(t, c) for t, choices in enumerate(candidate_sets) for c in choices]
    nr, nf, nw, nt = len(positions), len(flows), len(witnesses), len(targets)
    wo, to = nr + nf, nr + nf + nw
    n = to + nt
    rr, cc, vv, lo, hi = [], [], [], [], []

    def add(coefficients, lower, upper):
        row = len(lo)
        for col, value in coefficients.items():
            rr.append(row); cc.append(col); vv.append(value)
        lo.append(lower); hi.append(upper)

    by_graph = [[] for _ in graphs]
    balance = {}
    for i, (g, p) in enumerate(flows):
        variable = nr + i
        by_graph[g].append(variable)
        for record in missing[p]:
            add({variable: 1, ri[record]: -1}, -np.inf, 0)
        balance.setdefault((g, sources[p]), {})[variable] = 1
        balance.setdefault((g, destinations[p]), {})[variable] = -1
    for coefficients in balance.values():
        add(coefficients, 0, 0)
    for variables in by_graph:
        add({i: 1 for i in variables}, 1, np.inf)
    by_target = [[] for _ in candidate_sets]
    for i, (t, choice) in enumerate(witnesses):
        variable = wo + i
        by_target[t].append(variable)
        for record in choice:
            add({variable: 1, ri[record]: -1}, -np.inf, 0)
        add({variable: 1, **{ri[r]: -1 for r in choice}}, 1 - len(choice), np.inf)
        add({to + t: 1, variable: -1}, 0, np.inf)
    for t, variables in enumerate(by_target):
        add({to + t: 1, **{v: -1 for v in variables}}, -np.inf, 0)
    for goal, quota in quotas.items():
        add({to + int(i): 1 for i in targets.index[targets.goal.eq(goal)]}, quota, np.inf)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lo), n)).tocsr()
    objective = np.r_[np.ones(nr), np.zeros(n - nr)]
    integral = np.ones(n, dtype=np.int8)
    integral[nr:nr + nf] = 0
    answer = milp(objective, integrality=integral, bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lo, hi),
                  options={"time_limit": 180, "mip_rel_gap": 0})
    assert answer.status == 0, (answer.message, getattr(answer, "mip_dual_bound", None))
    chosen = [r for r, flag in zip(positions, answer.x[:nr]) if flag > .5]
    assert abs(answer.fun - len(chosen)) < 1e-7
    assert abs(answer.mip_dual_bound - len(chosen)) < 1e-7 and answer.mip_gap < 1e-10
    return dict(cost=len(chosen), records=chosen, status=int(answer.status), gap=float(answer.mip_gap),
                objective=float(answer.fun), dual_bound=float(answer.mip_dual_bound))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    out = out_dir(ap.parse_args().variant)
    n_records = int(np.load(out / "rule_masks.npz")["n_records"])
    actual = np.unpackbits(np.load(out / "rule_masks.npz")["rule_union"], bitorder="big")[:n_records].astype(bool)
    dual = pd.read_parquet(out / "recipe_duals.parquet", columns=[
        "rxn_class", "edge_id", "sub_1_smiles", "sub_2_smiles", "record_position_A", "record_position_B",
        "identity_A", "d", "core_outside_domain"])
    dual = dual.loc[dual.core_outside_domain].reset_index(drop=True)
    a, b, d = dual.record_position_A.to_numpy(int), dual.record_position_B.to_numpy(int), dual.d.to_numpy(float)
    nodes = sorted({(1, str(s)) for s in dual.sub_1_smiles} | {(2, str(s)) for s in dual.sub_2_smiles})
    node_index = {node: i for i, node in enumerate(nodes)}
    n1 = np.array([node_index[(1, str(s))] for s in dual.sub_1_smiles])
    n2 = np.array([node_index[(2, str(s))] for s in dual.sub_2_smiles])
    sources, destinations = np.where(d > 0, n2, n1), np.where(d > 0, n1, n2)
    missing = [tuple(r for r in [int(x), int(y)] if not actual[r]) for x, y in zip(a, b)]
    strong = np.abs(d) >= .05 - TOL
    q = dual.identity_A.eq("Q").to_numpy()
    rows, graph_positions = [], {}
    for (cls, edge), ix in dual.loc[strong].groupby(["rxn_class", "edge_id"], sort=True).indices.items():
        positions = np.flatnonzero(strong)[ix]
        full = nx.DiGraph([(int(sources[p]), int(destinations[p])) for p in positions])
        qpos = positions[q[positions]]
        qgraph = nx.DiGraph([(int(sources[p]), int(destinations[p])) for p in qpos])
        if not (len(qpos) and nx.is_directed_acyclic_graph(qgraph) and not nx.is_directed_acyclic_graph(full)):
            continue
        component = {node: i for i, c in enumerate(nx.strongly_connected_components(full)) if len(c) >= 2 for node in c}
        cycle = np.array(sorted(p for p in positions if sources[p] in component
                                and component[sources[p]] == component.get(destinations[p], -1)), dtype=int)
        base = nx.DiGraph([(int(sources[p]), int(destinations[p])) for p in cycle if not missing[p]])
        graph_positions[edge] = cycle
        rows.append(dict(edge_id=edge, rxn_class=cls, actual_cyclic=not nx.is_directed_acyclic_graph(base)))
    catalogue = pd.DataFrame(rows)
    targets = pd.read_csv(out / "crossgoal_targets.csv.gz", keep_default_na=False)
    classes = sorted(set(dual.rxn_class))
    solutions = {}
    for pp, fraction in [(0, 0.), (5, 1.)]:
        sid = "preference_only" if not pp else f"preference_m{pp}_f{fraction:g}_three_goals"
        chosen, class_costs = set(), {}
        for cls in classes:
            graphs = [graph_positions[e] for e in catalogue.loc[catalogue.rxn_class.eq(cls) & ~catalogue.actual_cyclic, "edge_id"]]
            part = targets.loc[targets.margin_pp.eq(pp) & targets.rxn_class.eq(cls)].reset_index(drop=True)
            quotas = {g: math.ceil(Fraction(str(fraction)) * int(part.goal.eq(g).sum())) for g in GOALS} if pp else {}
            result = optimize(graphs, part, quotas, missing, sources, destinations)
            chosen.update(result["records"])
            class_costs[cls] = result["cost"]
        assert len(chosen) == sum(class_costs.values())
        solutions[sid] = dict(minimum_additions=len(chosen), class_costs=class_costs,
                              selected_record_positions=sorted(int(r) for r in chosen))
        print(sid, len(chosen), flush=True)
    res = dict(n_preference_graphs=len(catalogue), n_actual_cyclic=int(catalogue.actual_cyclic.sum()), solutions=solutions)
    (out / "joint_preference_counterexample.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "solutions"}))


if __name__ == "__main__":
    main()
