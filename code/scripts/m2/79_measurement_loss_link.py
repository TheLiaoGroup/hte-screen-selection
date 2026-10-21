"""Relate independently counted measurement losses to observed counterexamples."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from m2common import out_dir

PAIR_KEY = ["rxn_class", "subpair"]


def outside_pair_histories(observations):
    """Keep full histories of outside pairs with at least one valid measurement."""
    u = observations.loc[observations.rxn_class.ne("Finkelstein")].copy()
    valid_outside = u.loc[u.analysis_valid & u.pair_group.eq("outside_only"), PAIR_KEY].drop_duplicates()
    u = u.merge(valid_outside, on=PAIR_KEY, how="inner", validate="many_to_one")
    u["detected_missing"] = u.measurement_status.eq("detected_not_quantified")
    pairs = u.groupby(PAIR_KEY, sort=True).agg(
        wells=("record_id", "size"), detected_not_quantified=("detected_missing", "sum")).reset_index()
    pairs["affected_pair"] = pairs.detected_not_quantified.gt(0)
    return pairs


def any_cycle_membership(frame):
    """An arc lies on a directed cycle exactly when its ends share an SCC."""
    graph = nx.DiGraph()
    arcs = []
    for row in frame.itertuples(index=False):
        left, right = (1, str(row.sub_1_smiles)), (2, str(row.sub_2_smiles))
        source, target = (right, left) if row.d > 0 else (left, right)
        arcs.append((source, target))
        graph.add_edge(source, target)
    assert graph.number_of_edges() == len(frame), "Repeated pair in recipe contrast"
    components = {node: i for i, nodes in enumerate(nx.strongly_connected_components(graph)) for node in nodes}
    return np.array([components[a] == components[b] for a, b in arcs], dtype=bool)


def certificate_pairs(targets, observations):
    index = observations.set_index("record_id")
    keys = set()
    for row in targets.itertuples(index=False):
        for dual in json.loads(row.certificate_dual_ids):
            records = index.loc[dual.split("|")]
            assert len(records) == 2 and records.analysis_valid.all()
            assert records.subpair.nunique() == records.rxn_class.nunique() == 1
            assert records.pair_group.nunique() == 1
            if records.pair_group.iloc[0] == "outside_only":
                keys.add((records.rxn_class.iloc[0], records.subpair.iloc[0]))
    return keys


def run(source, comparison_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(source / "processed" / "observations.parquet")
    partner = pd.read_csv(source / "partner_score_edges.csv")
    targets = partner.loc[partner.new_contradiction].copy()
    assert len(targets) > 0 and targets.edge_id.is_unique
    comparisons = pd.read_parquet(comparison_dir / "measurement_loss_comparisons.parquet")
    target_comparisons = comparisons.loc[comparisons.edge_id.isin(targets.edge_id)]
    target_counts = target_comparisons.groupby(["rxn_class", "edge_id", "comparison_status"]).size().unstack(fill_value=0)
    target_counts = targets[["rxn_class", "edge_id"]].merge(target_counts.reset_index(), how="left", validate="one_to_one").fillna(0)
    for label in ["complete", "quantitation_broken", "other_missing"]:
        if label not in target_counts:
            target_counts[label] = 0
        target_counts[label] = target_counts[label].astype(int)
    pairs = outside_pair_histories(u)
    certificates = certificate_pairs(targets, u)
    pair_keys = list(pairs[PAIR_KEY].itertuples(index=False, name=None))
    pairs["selected_certificate_pair"] = [key in certificates for key in pair_keys]
    assert int(pairs.selected_certificate_pair.sum()) == len(certificates)
    duals = pd.read_parquet(source / "recipe_duals.parquet")
    strong = duals.loc[duals.core_outside_domain & duals.edge_id.isin(targets.edge_id) &
                       duals.d.abs().ge(0.05 - 1e-12)].copy()
    strong["on_any_cycle"] = False
    for _, group in strong.groupby("edge_id", sort=True):
        strong.loc[group.index, "on_any_cycle"] = any_cycle_membership(group)
    outside = strong.loc[strong.identity_A.eq("O")].copy()
    outside = outside.merge(pairs[PAIR_KEY + ["affected_pair"]], on=PAIR_KEY,
                            how="left", validate="many_to_one")
    assert outside.affected_pair.notna().all()
    affected = outside.loc[outside.affected_pair]
    selected = pairs.loc[pairs.selected_certificate_pair]
    affected_pairs = pairs.loc[pairs.affected_pair]
    summary = dict(
        target_contrasts=len(targets), target_complete_comparisons=int(target_counts.complete.sum()),
        target_quantitation_broken_comparisons=int(target_counts.quantitation_broken.sum()),
        outside_pairs=len(pairs), affected_outside_pairs=len(affected_pairs),
        fraction_outside_pairs_affected=len(affected_pairs) / len(pairs),
        outside_pair_history_wells=int(pairs.wells.sum()),
        affected_outside_pair_history_wells=int(affected_pairs.wells.sum()),
        fraction_outside_wells_on_affected_pairs=float(affected_pairs.wells.sum() / pairs.wells.sum()),
        strong_outside_comparisons=len(outside), strong_outside_affected_comparisons=len(affected),
        fraction_strong_outside_comparisons_affected=len(affected) / len(outside),
        strong_outside_affected_comparisons_on_any_cycle=int(affected.on_any_cycle.sum()),
        target_contrasts_with_affected_outside_comparison_on_any_cycle=int(affected.loc[affected.on_any_cycle, "edge_id"].nunique()),
        selected_outside_certificate_pairs=len(selected),
        selected_outside_certificate_pairs_affected=int(selected.affected_pair.sum()),
        fraction_selected_outside_certificate_pairs_affected=float(selected.affected_pair.mean()),
        affected_pair_definition="An analysis-eligible outside-only pair with at least one detected-not-quantified well anywhere in its full cleaned history",
        pair_history_scope="All cleaned records of those pairs, including qc_twin_secondary; recipe comparisons exclude the twins",
        strong_comparison_definition="Absolute measured yield difference at least five percentage points, with 1e-12 numerical tolerance",
        cycle_definition="Any directed cycle in the full strong Q-plus-O graph of a target recipe contrast",
        certificate_definition="Outside pairs used by the deterministic selected minimum certificates; not all possible certificates",
        missing_yields_imputed=0,
        interpretation="Observed strong comparisons and certificates describe measurement depletion; broken tests have unknown refutation outcomes")
    target_counts.to_csv(destination / "measurement_loss_target_comparisons.csv", index=False)
    pairs.to_csv(destination / "measurement_loss_depletion_pairs.csv", index=False)
    cols = ["rxn_class", "edge_id", "subpair", "sub_1_smiles", "sub_2_smiles", "d", "affected_pair", "on_any_cycle"]
    outside[cols].sort_values(["rxn_class", "edge_id", "subpair"], kind="stable").to_csv(
        destination / "measurement_loss_depletion_comparisons.csv", index=False)
    (destination / "measurement_loss_link.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="clfzero")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.input_dir or out_dir(args.variant)
    print(json.dumps(run(source, args.comparison_dir or source, args.output_dir or source), indent=2), flush=True)


if __name__ == "__main__":
    main()
