"""Ordinal partner-score counterexamples on the cleaned, classifier-zero corpus.

Directed partner-preference graphs identify contradictions and their smallest
cycles, with same-campaign and retained-record comparisons.
(a dual is retained only if both of its records are in the record mask).
Q = qualified-core duals (identity Q on both records); O = all outside-only duals; domain core_outside_domain.
"""
import argparse
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from heapq import heappop, heappush
from m2common import out_dir

TOL = 1e-12
MARGIN = 5
FLAGSHIP_EDGE = "Acylation|C13|C38"
# (sub_1 = amine, sub_2 = acid as recorded) EDC (A) and PyBrOP (B) yields in percent, manuscript Fig. 2.
FLAGSHIP = [(34.61, 12.21), (0.31, 38.61), (60.55, 47.44), (9.61, 15.82)]


def build_graph(frame):
    """Nodes (role, SMILES); d > 0: role2 -> role1 (v_j < u_i); d < 0: role1 -> role2."""
    graph = nx.DiGraph()
    for x, z, d, cohort, did in zip(frame.sub_1_smiles, frame.sub_2_smiles, frame.d, frame.identity_A, frame.dual_id):
        row, col = (1, str(x)), (2, str(z))
        source, target = (col, row) if d > 0 else (row, col)
        graph.add_edge(source, target, cohort=cohort, dual_id=did)
    assert graph.number_of_edges() == len(frame)
    return graph


def canonical_cycle(graph, nodes):
    nodes = tuple(nodes)
    variants = [nodes[i:] + nodes[:i] for i in range(len(nodes))]
    return min(variants)


def cycle_key(graph, nodes):
    nodes = canonical_cycle(graph, nodes)
    arcs = [graph.edges[a, b] for a, b in zip(nodes, nodes[1:] + nodes[:1])]
    return (sum(arc["cohort"] == "O" for arc in arcs), len(nodes), nodes,
            tuple(arc["dual_id"] for arc in arcs))


def select_minimum_cycle(graph):
    best = None
    ordered = sorted(graph)
    adjacency = {node: sorted(graph.successors(node)) for node in ordered}
    for start in ordered:
        initial = (0, 0, (start,))
        distances = {start: initial}
        queue = [(*initial, start)]
        while queue:
            cost, length, path, node = heappop(queue)
            if distances[node] != (cost, length, path):
                continue
            if best is not None and (cost, length) > best[:2]:
                continue
            if node != start and graph.has_edge(node, start):
                candidate = cycle_key(graph, path)
                if best is None or candidate < best:
                    best = candidate
            for nxt in adjacency[node]:
                if nxt <= start:
                    continue
                arc = graph.edges[node, nxt]
                value = (cost + (arc["cohort"] == "O"), length + 1, path + (nxt,))
                if nxt not in distances or value < distances[nxt]:
                    distances[nxt] = value
                    heappush(queue, (*value, nxt))
    return best


def self_test():
    graph = nx.DiGraph()
    for i, (a, b, cohort) in enumerate([((1, "a"), (2, "x"), "Q"), ((2, "x"), (1, "b"), "O"),
                                       ((1, "b"), (2, "y"), "Q"), ((2, "y"), (1, "a"), "O")]):
        graph.add_edge(a, b, cohort=cohort, dual_id=str(i))
    assert select_minimum_cycle(graph)[:2] == (2, 4)
    rng = np.random.default_rng(472026)
    from itertools import product
    for _ in range(300):
        graph = nx.DiGraph()
        for i, (x, z) in enumerate(product(range(3), range(3))):
            state = int(rng.integers(5))
            if state:
                u, v = (1, str(x)), (2, str(z))
                if state in [1, 3]:
                    u, v = v, u
                graph.add_edge(u, v, cohort="Q" if state < 3 else "O", dual_id=str(i))
        truth = [cycle_key(graph, nodes) for nodes in nx.simple_cycles(graph)]
        assert select_minimum_cycle(graph) == (min(truth) if truth else None)


def cyclic(frame):
    return len(frame) > 0 and not nx.is_directed_acyclic_graph(build_graph(frame))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    out = out_dir(v)
    t0 = time.time()
    self_test()
    edges = pd.read_csv(out / "recipe_edges.csv", usecols=["rxn_class", "edge_id", "recipe_A", "recipe_B"])
    edges = edges.sort_values(["rxn_class", "edge_id"]).reset_index(drop=True)
    dual = pd.read_parquet(out / "recipe_duals.parquet")
    dual = dual.loc[dual.core_outside_domain].copy()
    assert dual.identity_A.eq(dual.identity_B).all() and set(dual.identity_A) == {"Q", "O"}
    assert not dual.duplicated(["edge_id", "sub_1_smiles", "sub_2_smiles"]).any()
    dual["dual_id"] = dual.record_id_A.astype(str) + "|" + dual.record_id_B.astype(str)
    for side in "AB":
        dual[f"source_campaign_{side}"] = dual[f"source_campaign_{side}"].astype("string").fillna("")
    masks = np.load(out / "rule_masks.npz")
    n_records = int(masks["n_records"])
    union = np.unpackbits(masks["rule_union"], bitorder="big")[:n_records].astype(bool)
    assert v != 'raw' or int(union.sum()) == 277192
    dual["rule_retained"] = union[dual.record_position_A.to_numpy()] & union[dual.record_position_B.to_numpy()]
    strong = dual.loc[dual.d.abs().ge(MARGIN / 100 - TOL)]
    groups = strong.groupby("edge_id", sort=False).indices
    empty = strong.iloc[:0]
    rows = []
    for cls, edge in edges[["rxn_class", "edge_id"]].itertuples(index=False, name=None):
        z = strong.iloc[groups[edge]] if edge in groups else empty
        q = z.loc[z.identity_A.eq("Q")]
        q_cyclic = cyclic(q)
        full_graph = build_graph(z)
        full_cyclic = len(z) > 0 and not nx.is_directed_acyclic_graph(full_graph)
        new = bool(len(q) and not q_cyclic and full_cyclic)
        row = dict(rxn_class=cls, edge_id=edge, n_Q_strong=len(q), n_O_strong=int(len(z) - len(q)),
                   Q_cyclic=q_cyclic, full_cyclic=full_cyclic, new_contradiction=new,
                   minimum_O=np.nan, certificate_length=np.nan, certificate_dual_ids="",
                   campaign_new=False, n_campaign_new_groups=0, rule_union_cyclic=cyclic(z.loc[z.rule_retained]))
        if new:
            k, length, _, ids = select_minimum_cycle(full_graph)
            assert k >= 1
            row.update(minimum_O=k, certificate_length=length, certificate_dual_ids=json.dumps(list(ids)))
            same = z.loc[z.source_campaign_A.ne("") & z.source_campaign_A.eq(z.source_campaign_B)]
            n_groups = 0
            for _, g in same.groupby("source_campaign_A", sort=True):
                gq = g.loc[g.identity_A.eq("Q")]
                n_groups += bool(len(gq) and not cyclic(gq) and cyclic(g))
            row.update(campaign_new=n_groups > 0, n_campaign_new_groups=n_groups)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "partner_score_edges.csv", index=False)
    new = res.loc[res.new_contradiction]
    collective = new.loc[new.minimum_O.ge(2)]
    campaign = new.loc[new.campaign_new]

    # Flagship acylation example: the four duals of FLAGSHIP_EDGE matching the manuscript yields.
    f = dual.loc[dual.edge_id.eq(FLAGSHIP_EDGE)]
    pct = np.round(100 * f[["yield_A", "yield_B"]].to_numpy(), 2)
    flagship = []
    for a, b in FLAGSHIP:
        hit = f.loc[(pct[:, 0] == a) & (pct[:, 1] == b)]
        assert len(hit) == 1, (a, b, len(hit))
        flagship.append(hit.iloc[0])
    flagship = pd.DataFrame(flagship)[["edge_id", "subpair", "sub_1_smiles", "sub_2_smiles", "record_id_A", "record_id_B",
                                       "yield_A", "yield_B", "d", "identity_A", "source_campaign_A", "source_campaign_B"]]
    flagship.to_csv(out / "partner_score_flagship.csv", index=False)
    fx = flagship.reset_index(drop=True)
    four_cycle = (fx.sub_1_smiles.nunique() == 2 and fx.sub_2_smiles.nunique() == 2
                  and not nx.is_directed_acyclic_graph(build_graph(fx.assign(dual_id=fx.record_id_A)))
                  and nx.is_directed_acyclic_graph(build_graph(fx.iloc[:3].assign(dual_id=fx.record_id_A[:3]))))
    fe = res.set_index("edge_id").loc[FLAGSHIP_EDGE]
    identities = "".join(fx.identity_A)

    def item(name, manuscript, recomputed):
        return dict(name=name, manuscript=manuscript, recomputed=recomputed, match=bool(manuscript == recomputed))

    checks = [item("new_contradiction_contrasts", 177, len(new)),
              item("new_contradiction_classes", 11, int(new.rxn_class.nunique())),
              item("closed_by_one_omitted_combination", 111, int(new.minimum_O.eq(1).sum())),
              item("collective_two_to_four_omitted", 66, int(new.minimum_O.ge(2).sum())),
              item("collective_minimum_O_range", [2, 4], [int(collective.minimum_O.min()), int(collective.minimum_O.max())]),
              item("same_campaign_contrasts_of_177", 27, len(campaign)),
              item("same_campaign_classes", 8, int(campaign.rxn_class.nunique())),
              item("rule_union_retained_of_177", 21, int(new.rule_union_cyclic.sum())),
              item("rule_union_retained_of_66", 4, int(collective.rule_union_cyclic.sum())),
              item("flagship_identities_QQQO", "QQQO", identities),
              item("flagship_three_Q_acyclic_four_cyclic", True, bool(four_cycle)),
              item("flagship_edge_in_177_with_minimum_O_1", True, bool(fe.new_contradiction and fe.minimum_O == 1))]
    for k, r in fx.iterrows():
        checks.append(item(f"flagship_dual_{k + 1}_yield_pair", list(FLAGSHIP[k]),
                           [round(100 * r.yield_A, 2), round(100 * r.yield_B, 2)]))
    assert len(new) > 0 and len(collective) > 0
    report = dict(variant=v, checks=checks, all_match=all(row["match"] for row in checks))
    (out / "check_omitted.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if not report["all_match"]:
        raise ValueError("Cleaned counterexample checkpoints failed: "
                         + ", ".join(row["name"] for row in checks if not row["match"]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
