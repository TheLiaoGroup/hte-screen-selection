"""Approved collective witnesses and identity-only complete-block retention schedules.

This extension consumes frozen analysis outputs without changing their rules.
Witness selection reads measured yield differences; retention scheduling does not.
"""
import argparse
from collections import defaultdict
from importlib import import_module
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "m2"))
ps = import_module("10_partner_score_counterexamples")
chemical = import_module("80_chemical_example")
from retention import anchor_R4, anchor_R4p, rule_R2, SEED, N_DRAWS

CLASSES = ("Acylation", "Heck", "Ullmann")
BUDGETS = tuple(range(0, 55001, 5000)) + (57149,)
MARGIN = ps.MARGIN / 100 - ps.TOL


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def rank_cycles(graph):
    """Minimize omitted arcs, then length; maximize raw minimum strength; break identity ties."""
    minimum = ps.select_minimum_cycle(graph)
    if minimum is None:
        return None
    cost, length = minimum[:2]
    best, count = None, 0
    for component in nx.strongly_connected_components(graph):
        if len(component) < length:
            continue
        adjacency = {node: sorted(n for n in graph.successors(node) if n in component)
                     for node in component}
        for start in sorted(component):
            def visit(nodes, used_outside, identities, strength):
                nonlocal best, count
                node = nodes[-1]
                if len(nodes) == length:
                    if not graph.has_edge(node, start):
                        return
                    closing = graph.edges[node, start]
                    if used_outside + (closing["cohort"] == "O") != cost:
                        return
                    ids = tuple(sorted(identities + (closing["dual_id"],)))
                    weak = min(strength, closing["strength"])
                    count += 1
                    key = (-weak, ids)
                    if best is None or key < best[0]:
                        best = (key, tuple(nodes))
                    return
                for nxt in adjacency[node]:
                    if nxt <= start or nxt in nodes:
                        continue
                    arc = graph.edges[node, nxt]
                    next_cost = used_outside + (arc["cohort"] == "O")
                    if next_cost > cost or next_cost + length - len(nodes) < cost:
                        continue
                    visit(nodes + (nxt,), next_cost, identities + (arc["dual_id"],),
                          min(strength, arc["strength"]))
            visit((start,), 0, (), float("inf"))
    if best is None:
        raise AssertionError("Minimum cycle had no enumerated legal witness")
    key, nodes = best
    cycle_ids = [graph.edges[a, b]["dual_id"] for a, b in zip(nodes, nodes[1:] + nodes[:1])]
    return dict(minimum_O=int(cost), cycle_length=int(length), minimum_strength=float(-key[0]),
                dual_ids=list(key[1]), cycle_order_dual_ids=cycle_ids, candidate_count=count)


def witness_graph(frame):
    graph = ps.build_graph(frame)
    strengths = dict(zip(frame.dual_id, frame.d.abs()))
    for _, _, arc in graph.edges(data=True):
        arc["strength"] = float(strengths[arc["dual_id"]])
    return graph


def collective_details(candidate, duals, observations, recipes, names):
    selected = duals.set_index("dual_id").loc[candidate["dual_ids"]]
    observed = observations.set_index("record_id")
    recipe = recipes.set_index("edge_id").loc[candidate["edge_id"]]
    source = observed.loc[list(selected.record_id_A) + list(selected.record_id_B)]
    assert source.analysis_valid.all() and not source.qc_twin_secondary.any()
    molecules = {str(role): [chemical.molecule_record(smiles, role, source, names)
                             for smiles in sorted(selected[f"sub_{role}_smiles"].unique())]
                 for role in (1, 2)}
    for role, rows in molecules.items():
        for index, item in enumerate(rows, 1):
            item["short_label"] = ("A" if role == "1" else "B") + str(index)
    labels = {role: {m["smiles"]: m for m in molecules[str(role)]} for role in (1, 2)}
    reactions = []
    for index, identity in enumerate(candidate["cycle_order_dual_ids"], 1):
        row = selected.loc[identity]
        record = dict(edge_id=candidate["edge_id"], rxn_class=candidate["rxn_class"],
                      dual_id=identity, cycle_order=index, subpair=row.subpair,
                      sub_1_smiles=row.sub_1_smiles, sub_2_smiles=row.sub_2_smiles,
                      sub_1_short_label=labels[1][row.sub_1_smiles]["short_label"],
                      sub_2_short_label=labels[2][row.sub_2_smiles]["short_label"],
                      sub_1_label=labels[1][row.sub_1_smiles]["display_name"],
                      sub_2_label=labels[2][row.sub_2_smiles]["display_name"],
                      difference_A_minus_B_pct=float(row.d * 100),
                      absolute_yield_difference_pct=float(abs(row.d) * 100),
                      preferred_recipe="A" if row.d > 0 else "B",
                      kept_in_qualified_core=bool(row.identity_A == "Q"),
                      retention_status="kept" if row.identity_A == "Q" else "discarded")
        for side in "AB":
            identity = row[f"record_id_{side}"]
            measurement = observed.loc[identity]
            assert measurement.subpair == row.subpair
            assert measurement.condition_id == recipe[f"condition_{side}"]
            assert bool(measurement.qualified_row) == (row.identity_A == "Q")
            assert np.isclose(measurement["yield"], row[f"yield_{side}"], rtol=0, atol=1e-15)
            record.update({f"record_id_{side}": identity,
                           f"name0_{side}": str(measurement.name0),
                           f"source_file_{side}": str(measurement.source_file),
                           f"source_row_{side}": int(measurement.source_row),
                           f"condition_{side}": str(recipe[f"condition_{side}"]),
                           f"recipe_{side}": str(recipe[f"recipe_{side}"]),
                           f"yield_{side}_fraction": float(row[f"yield_{side}"]),
                           f"yield_{side}_pct": float(row[f"yield_{side}"] * 100),
                           f"outside_reason_{side}": str(measurement.outside_reason)})
        reactions.append(record)
    assert sum(not row["kept_in_qualified_core"] for row in reactions) == candidate["minimum_O"]
    return {**candidate, "condition_A": recipe.condition_A, "condition_B": recipe.condition_B,
            "coordinate": recipe.coordinate, "coordinate_family": recipe.coordinate_family,
            "recipe_A": recipe.recipe_A, "recipe_B": recipe.recipe_B,
            "reagent_A": chemical.changed_label(json.loads(recipe.changed_fields_A_json)),
            "reagent_B": chemical.changed_label(json.loads(recipe.changed_fields_B_json)),
            "raw_condition_fields_A": json.loads(recipe.raw_fields_A_json),
            "raw_condition_fields_B": json.loads(recipe.raw_fields_B_json),
            "molecules_by_role": molecules, "reactions": reactions}


def build_examples(source, destination, catalogue):
    partner = pd.read_csv(source / "partner_score_edges.csv")
    all_targets = partner.loc[partner.new_contradiction]
    classes = sorted(pd.read_csv(source / "recipe_edges.csv").rxn_class.unique())
    distribution = []
    for cls in classes:
        rows = all_targets.loc[all_targets.rxn_class.eq(cls)]
        distribution.append(dict(rxn_class=cls, single_omission=int(rows.minimum_O.eq(1).sum()),
                                 collective=int(rows.minimum_O.ge(2).sum()), total=len(rows)))
    pd.DataFrame(distribution).to_csv(destination / "Figure_3a.csv", index=False)
    assert len(classes) == 19 and sum(row["total"] for row in distribution) == 177
    duals = pd.read_parquet(source / "recipe_duals.parquet")
    duals = duals.loc[duals.core_outside_domain & duals.d.abs().ge(MARGIN)].copy()
    duals["dual_id"] = duals.record_id_A + "|" + duals.record_id_B
    assert duals.identity_A.eq(duals.identity_B).all()
    names = chemical.compound_names(catalogue)
    observations = pd.read_parquet(source / "processed" / "observations.parquet")
    recipes = pd.read_csv(source / "recipe_edges.csv")
    results, audit = [], []
    for cls in CLASSES:
        targets = all_targets.loc[all_targets.rxn_class.eq(cls) & all_targets.minimum_O.ge(2)]
        boundary = min(zip(targets.minimum_O.astype(int), targets.certificate_length.astype(int)))
        candidates = []
        for target in targets.itertuples(index=False):
            rows = duals.loc[duals.edge_id.eq(target.edge_id)]
            graph = witness_graph(rows)
            frozen_cost = (int(target.minimum_O), int(target.certificate_length))
            assert tuple(ps.select_minimum_cycle(graph)[:2]) == frozen_cost
            row = dict(rxn_class=cls, edge_id=target.edge_id, minimum_O=frozen_cost[0],
                       cycle_length=frozen_cost[1], eligible_class_boundary=frozen_cost == boundary)
            if frozen_cost == boundary:
                cycle = rank_cycles(graph)
                candidate = dict(rxn_class=cls, edge_id=target.edge_id, **cycle,
                                 min_abs_yield_difference_pct=cycle["minimum_strength"] * 100)
                candidates.append(candidate)
                row.update(candidate_count=cycle["candidate_count"],
                           best_min_abs_yield_difference_pct=candidate["min_abs_yield_difference_pct"],
                           best_dual_ids=json.dumps(cycle["dual_ids"]))
            audit.append(row)
        winner = min(candidates, key=lambda item: (-item["minimum_strength"], item["edge_id"], tuple(item["dual_ids"])))
        details = collective_details(winner, duals, observations, recipes, names)
        details["eligible_contrasts"] = len(candidates)
        details["eligible_cycles"] = sum(item["candidate_count"] for item in candidates)
        results.append(dict(rxn_class=cls, selected=details))
        print(json.dumps({"selected_class": cls, "edge_id": winner["edge_id"],
                          "minimum_O": winner["minimum_O"], "length": winner["cycle_length"],
                          "minimum_pct": winner["min_abs_yield_difference_pct"]}), flush=True)
    result = dict(selection_rule={
        "scope": "Collective new counterexamples in each requested reaction class (minimum omitted count at least two)",
        "hierarchy": "Within a class: minimum omitted-comparison count; shortest cycle length; maximum minimum unrounded absolute yield difference",
        "ties": "Recipe-comparison edge ID, then lexicographic sorted full source-record dual IDs",
        "threshold_percentage_points": ps.MARGIN,
        "enumeration": "All simple directed cycles on the class minimum-cost/shortest-length boundary; rotations counted once; long cycles allowed"},
        classes=results)
    json_file(destination / "collective_examples.json", result)
    pd.DataFrame([row for item in results for row in item["selected"]["reactions"]]).to_csv(
        destination / "collective_examples.csv", index=False, float_format="%.17g")
    pd.DataFrame(audit).to_csv(destination / "collective_candidate_audit.csv", index=False, float_format="%.17g")
    return result


def load_corpus(source):
    cols = ["rxn_class", "subpair", "analysis_valid", "qualified_row", "record_id"]
    u = pd.read_parquet(source / "processed" / "observations.parquet", columns=cols)
    valid = u.analysis_valid.to_numpy()
    q = (u.qualified_row & u.analysis_valid).to_numpy()
    pool = valid & ~q
    units = u.loc[pool, ["rxn_class", "subpair"]].groupby(["rxn_class", "subpair"], sort=True).ngroup().to_numpy()
    return SimpleNamespace(u=u, n=len(u), Q=q, valid=valid, pool=pool,
                           pool_positions=np.flatnonzero(pool), unit_of=units)


def complete_blocks(duals, corpus):
    """Identity-only distinct Q/non-Q dual pairs sharing exactly one partner."""
    columns = ["edge_id", "sub_1_smiles", "sub_2_smiles", "identity_A", "identity_B",
               "record_position_A", "record_position_B", "record_id_A", "record_id_B"]
    identities = duals[columns].copy()
    identities["dual_id"] = identities.record_id_A + "|" + identities.record_id_B
    identities = identities.sort_values(["edge_id", "dual_id"], kind="stable").reset_index(drop=True)
    identities["dual_index"] = np.arange(len(identities), dtype=np.int32)
    qmask = identities.identity_A.eq("Q") & identities.identity_B.eq("Q")
    q, other = identities.loc[qmask], identities.loc[~qmask]
    blocks = []
    for role in (1, 2):
        shared, changed = f"sub_{role}_smiles", f"sub_{3-role}_smiles"
        left = q[["edge_id", shared, changed, "dual_index"]]
        right = other[["edge_id", shared, changed, "dual_index"]]
        merged = left.merge(right, on=["edge_id", shared], suffixes=("_Q", "_other"), sort=False)
        merged = merged.loc[merged[f"{changed}_Q"].ne(merged[f"{changed}_other"])]
        blocks.append(merged[["dual_index_Q", "dual_index_other"]].to_numpy(dtype=np.int32))
    blocks = np.concatenate(blocks)
    blocks = blocks[np.lexsort((blocks[:, 1], blocks[:, 0]))]
    assert not np.any(np.all(blocks[1:] == blocks[:-1], axis=1))
    a = identities.record_position_A.to_numpy()
    b = identities.record_position_B.to_numpy()
    endpoints = np.column_stack((a[blocks[:, 1]], b[blocks[:, 1]])).astype(np.int32)
    # Qualified endpoints cost no additions; use a sentinel excluded by schedule_prefixes.
    endpoints[~corpus.pool[endpoints]] = -1
    assert (endpoints >= 0).any(axis=1).all()
    frozen = rule_R2(corpus, None, duals)
    assert np.array_equal(np.unique(endpoints[endpoints >= 0]), frozen)
    return identities, blocks, endpoints


def schedule_prefixes(endpoints, order, budgets, n_records):
    """Return the longest complete-block prefix under each budget, deduplicating endpoints."""
    if len(order) != len(endpoints):
        raise ValueError("Block ordering does not contain the complete inventory")
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(len(order), dtype=np.int32)
    first = np.full(n_records, len(order), dtype=np.int32)
    for column in range(endpoints.shape[1]):
        valid = endpoints[:, column] >= 0
        np.minimum.at(first, endpoints[valid, column], ranks[valid])
    positions = np.flatnonzero(first < len(order))
    introduced = np.bincount(first[positions], minlength=len(order))
    cumulative = np.cumsum(introduced)
    result = []
    for budget in budgets:
        prefix = int(np.searchsorted(cumulative, budget, side="right"))
        added = positions[first[positions] < prefix]
        assert len(added) <= budget
        result.append(dict(nominal_budget=int(budget), prefix_blocks=prefix, added=added))
    return result


class PreferenceEvaluator:
    """Topological evaluation of every retained edge in each full target graph."""
    def __init__(self, duals, edge_ids):
        self.a = duals.record_position_A.to_numpy()
        self.b = duals.record_position_B.to_numpy()
        self.graphs = []
        groups = duals.groupby("edge_id", sort=False).indices
        for identity in edge_ids:
            indexes = groups[identity]
            frame = duals.iloc[indexes]
            nodes = sorted({(1, str(x)) for x in frame.sub_1_smiles} |
                           {(2, str(x)) for x in frame.sub_2_smiles})
            node_ids = {value: index for index, value in enumerate(nodes)}
            source, target = [], []
            for row in frame.itertuples(index=False):
                a, b = node_ids[(1, str(row.sub_1_smiles))], node_ids[(2, str(row.sub_2_smiles))]
                start, end = (b, a) if row.d > 0 else (a, b)
                source.append(start)
                target.append(end)
            self.graphs.append((indexes, len(nodes), np.array(source), np.array(target)))

    def evaluate(self, mask):
        keep = mask[self.a] & mask[self.b]
        output = []
        for indexes, n_nodes, starts, ends in self.graphs:
            included = keep[indexes]
            source, target = starts[included], ends[included]
            if len(source) < 4:
                output.append(False)
                continue
            degree = np.bincount(target, minlength=n_nodes)
            adjacency = [[] for _ in range(n_nodes)]
            for a, b in zip(source.tolist(), target.tolist()):
                adjacency[a].append(b)
            pending = list(np.flatnonzero(degree == 0))
            removed = 0
            while pending:
                node = pending.pop()
                removed += 1
                for nxt in adjacency[node]:
                    degree[nxt] -= 1
                    if degree[nxt] == 0:
                        pending.append(nxt)
            output.append(removed < n_nodes)
        return np.array(output, dtype=bool)


def build_budget_curves(source, destination):
    corpus = load_corpus(source)
    # Scheduling receives only these identity columns, never yields or preferences.
    identity_columns = ["edge_id", "sub_1_smiles", "sub_2_smiles", "identity_A", "identity_B",
                        "record_position_A", "record_position_B", "record_id_A", "record_id_B"]
    identities = pd.read_parquet(source / "recipe_duals.parquet", columns=identity_columns)
    inventory, blocks, endpoints = complete_blocks(identities, corpus)
    frozen_r2 = np.load(source / "retention_rules.npz")["R2"]
    assert np.array_equal(np.unique(endpoints[endpoints >= 0]), frozen_r2)
    if len(frozen_r2) != BUDGETS[-1]:
        raise AssertionError("Frozen R2 budget changed; the approved extension must be reviewed")
    inventory[["dual_index", "edge_id", "dual_id", "record_position_A", "record_position_B"]].to_parquet(
        destination / "block_dual_identities.parquet", index=False)
    pd.DataFrame(blocks, columns=["qualified_dual_index", "other_dual_index"]).to_parquet(
        destination / "complete_blocks.parquet", index=False)
    partner = pd.read_csv(source / "partner_score_edges.csv")
    target_ids = list(partner.loc[partner.new_contradiction, "edge_id"])
    duals = pd.read_parquet(source / "recipe_duals.parquet")
    duals = duals.loc[duals.core_outside_domain & duals.edge_id.isin(target_ids) & duals.d.abs().ge(MARGIN)].copy()
    evaluator = PreferenceEvaluator(duals, target_ids)
    assert len(target_ids) == 177
    assert evaluator.evaluate(corpus.valid).all() and not evaluator.evaluate(corpus.Q).any()
    rows, audit = [], []
    cached_r2 = {}
    for draw in range(N_DRAWS):
        order = np.random.default_rng(SEED + draw).permutation(len(blocks))
        checkpoints = schedule_prefixes(endpoints, order, BUDGETS, corpus.n)
        schedule_digest = hashlib.sha256(order.astype("<i4").tobytes()).hexdigest()
        for checkpoint in checkpoints:
            added = checkpoint["added"]
            budget = len(added)
            mask = corpus.Q.copy()
            mask[added] = True
            key = added.astype("<i4").tobytes()
            if key not in cached_r2:
                cached_r2[key] = int(evaluator.evaluate(mask).sum())
            values = {"R2": cached_r2[key]}
            for method, anchor in (("random_pair_blocks", anchor_R4p), ("random_records", anchor_R4)):
                control = anchor(corpus, budget, draw)
                assert len(control) == budget and corpus.pool[control].all()
                mask = corpus.Q.copy()
                mask[control] = True
                values[method] = int(evaluator.evaluate(mask).sum())
            audit.append(dict(draw=draw, seed=SEED + draw, nominal_budget=checkpoint["nominal_budget"],
                              actual_budget=budget, prefix_blocks=checkpoint["prefix_blocks"],
                              order_sha256=schedule_digest,
                              added_record_positions_sha256=hashlib.sha256(added.astype("<i4").tobytes()).hexdigest()))
            for method, retained in values.items():
                rows.append(dict(nominal_budget=checkpoint["nominal_budget"], draw=draw, seed=SEED + draw,
                                 actual_budget=budget, method=method, retained=retained, target_count=len(target_ids)))
        print(json.dumps({"completed_schedule": draw + 1, "schedules": N_DRAWS,
                          "complete_blocks": len(blocks)}), flush=True)
    draws = pd.DataFrame(rows)
    terminal = draws.loc[draws.nominal_budget.eq(BUDGETS[-1])].groupby("method").retained.mean().to_dict()
    expected = {"R2": 176., "random_pair_blocks": 82.85, "random_records": 39.62}
    if terminal != expected:
        json_file(destination / "endpoint_failure.json", dict(expected=expected, observed=terminal))
        raise AssertionError(f"Frozen retention endpoint failed: {terminal!r}")
    assert draws.loc[draws.nominal_budget.eq(0), "retained"].eq(0).all()
    draws.to_csv(destination / "Figure_4_draws.csv", index=False)
    pd.DataFrame(audit).to_csv(destination / "block_schedule_audit.csv", index=False)
    summaries = []
    for (nominal, method), group in draws.groupby(["nominal_budget", "method"], sort=True):
        lo, hi = np.percentile(group.retained, [2.5, 97.5])
        summaries.append(dict(nominal_budget=int(nominal), method=method,
                              actual_budget_mean=float(group.actual_budget.mean()),
                              actual_budget_min=int(group.actual_budget.min()), actual_budget_max=int(group.actual_budget.max()),
                              retained_mean=float(group.retained.mean()), retained_q025=float(lo), retained_q975=float(hi),
                              target_count=len(target_ids), n_draws=len(group)))
    pd.DataFrame(summaries).to_csv(destination / "Figure_4_summary.csv", index=False, float_format="%.17g")
    metadata = dict(
        definition="A distinct identity block is one qualified-core dual and one non-qualified dual for the same recipe comparison, sharing exactly one substrate partner; the four endpoint records complete a two-substrate-by-two-recipe comparison",
        scheduling="Lexicographically order block identities by qualified dual index then other dual index; randomly permute complete identities; take the longest prefix whose deduplicated valid non-core endpoint union does not exceed the nominal budget",
        schedule_reads="Recipe-comparison IDs, exact recorded substrate SMILES, source record IDs/positions, qualified/valid status only; no yields, preference directions or graph outcomes",
        controls="Frozen random whole-substrate-pair blocks with the last block truncated, and frozen random individual records, each matched to the actual added-record count of each R2 draw",
        evaluation="Retain every strong comparison whose two endpoint records are retained; detect cycles in each full target preference graph, not merely the selected certificate",
        interval="2.5th and 97.5th empirical percentiles across schedules; not confidence intervals",
        base_seed=SEED, n_draws=N_DRAWS, nominal_budgets=list(BUDGETS), complete_blocks=len(blocks),
        dual_identities=len(inventory), endpoint_budget=len(frozen_r2), target_count=len(target_ids),
        terminal_means=terminal, frozen_terminal_means=expected, terminal_checks_passed=True)
    json_file(destination / "budget_curve_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "out" / "m2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "science_extensions")
    parser.add_argument("--compound-catalogue", type=Path)
    parser.add_argument("--only", choices=("examples", "curves", "all"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only in ("examples", "all"):
        build_examples(args.input_dir, args.output_dir, args.compound_catalogue or chemical.catalogue_path())
    if args.only in ("curves", "all"):
        build_budget_curves(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
