"""Count paired recipe tests prevented by unquantifiable detected product.

This stage uses only cleaned observations and the recipe catalogue. It does not
infer yields or ask whether a prevented test would have been a counterexample.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m2common import out_dir

CELL_KEY = ["rxn_class", "subpair", "condition_id"]
PAIR_KEY = ["rxn_class", "subpair"]
COMPARISON_KEY = ["rxn_class", "edge_id", "subpair"]
NUMERIC = "valid_numeric"
DETECTED = "detected_not_quantified"


def collapse_cells(observations):
    """Choose one state per cell, preferring usable numeric measurements."""
    u = observations.loc[
        observations.rxn_class.ne("Finkelstein") & ~observations.qc_twin_secondary
    ].copy()
    numeric = u.analysis_valid & np.isfinite(u["yield"])
    u["cell_priority"] = np.select(
        [numeric, u.measurement_status.eq(DETECTED),
         u.measurement_status.eq("no_screen_record")], [3, 2, 1], default=0)
    u["cell_state"] = u.cell_priority.map(
        {3: NUMERIC, 2: DETECTED, 1: "no_screen_record", 0: "other"})
    order = CELL_KEY + ["cell_priority", "record_id"]
    u = u.sort_values(order, ascending=[True] * 3 + [False, True], kind="stable")
    cells = u.drop_duplicates(CELL_KEY, keep="first")
    assert not cells.duplicated(CELL_KEY).any()
    return cells[CELL_KEY + ["sub_1_smiles", "sub_2_smiles", "pair_group", "cell_state"]].copy()


def build_comparisons(cells, edges):
    """Inner joins require an actually recorded cell under both recipes."""
    a = cells.rename(columns={"condition_id": "condition_A", "cell_state": "state_A"})
    b = cells[CELL_KEY + ["cell_state"]].rename(
        columns={"condition_id": "condition_B", "cell_state": "state_B"})
    keep = ["rxn_class", "edge_id", "condition_A", "condition_B"]
    comparisons = a.merge(edges[keep], on=["rxn_class", "condition_A"], validate="many_to_many")
    comparisons = comparisons.merge(b, on=["rxn_class", "subpair", "condition_B"],
                                    validate="many_to_one")
    complete = comparisons.state_A.eq(NUMERIC) & comparisons.state_B.eq(NUMERIC)
    usable = comparisons.state_A.isin([NUMERIC, DETECTED]) & comparisons.state_B.isin([NUMERIC, DETECTED])
    comparisons["comparison_status"] = np.select(
        [complete, usable & ~complete], ["complete", "quantitation_broken"], default="other_missing")
    assert not comparisons.duplicated(COMPARISON_KEY).any()
    assert not (complete & comparisons.comparison_status.eq("quantitation_broken")).any()
    return comparisons.sort_values(COMPARISON_KEY, kind="stable").reset_index(drop=True)


def count_rectangles(frame):
    """Count four-cycle rectangles without enumerating their eight wells."""
    if frame.empty:
        return 0
    left, right = pd.factorize(frame.sub_1_smiles)[0], pd.factorize(frame.sub_2_smiles)[0]
    shape = (int(left.max()) + 1, int(right.max()) + 1)
    adjacency = np.zeros(shape, dtype=np.int64)
    adjacency[left, right] = 1
    assert int(adjacency.sum()) == len(frame), "Duplicate substrate combination in recipe contrast"
    # Multiply along the smaller side; a pair with k common partners gives C(k, 2).
    if shape[0] > shape[1]:
        adjacency = adjacency.T
    shared = adjacency @ adjacency.T
    common = shared[np.triu_indices(len(shared), k=1)]
    return int(np.sum(common * (common - 1) // 2))


def rectangle_catalogue(comparisons):
    rows = []
    usable = comparisons.loc[comparisons.comparison_status.ne("other_missing")]
    for (cls, edge_id), group in usable.groupby(["rxn_class", "edge_id"], sort=True):
        complete = count_rectangles(group.loc[group.comparison_status.eq("complete")])
        possible = count_rectangles(group)
        assert possible >= complete
        rows.append(dict(rxn_class=cls, edge_id=edge_id, complete=complete,
                         quantitation_broken=possible - complete))
    return pd.DataFrame(rows, columns=["rxn_class", "edge_id", "complete", "quantitation_broken"])


def summarize(comparisons, rectangles, full_duals):
    counts = comparisons.comparison_status.value_counts()
    complete, broken = int(counts.get("complete", 0)), int(counts.get("quantitation_broken", 0))
    assert complete == full_duals, ("Complete comparisons disagree with recipe duals", complete, full_duals)
    lost = comparisons.loc[comparisons.comparison_status.eq("quantitation_broken")]
    outside = int(lost.pair_group.eq("outside_only").sum())
    unassigned = int(lost.pair_group.isna().sum())
    represented = int(lost.pair_group.isin(["qualified_only", "mixed"]).sum())
    assert outside + unassigned + represented == broken
    missing_group = lost.loc[lost.pair_group.isna()]
    assert missing_group.state_A.eq(DETECTED).all() and missing_group.state_B.eq(DETECTED).all()
    rect_complete, rect_broken = int(rectangles.complete.sum()), int(rectangles.quantitation_broken.sum())
    by_class = rectangles.groupby("rxn_class")["quantitation_broken"].sum()
    return dict(
        definition="Two recorded cells for one substrate pair and one single-coordinate recipe contrast",
        cell_priority=[NUMERIC, DETECTED, "no_screen_record", "other"],
        comparison_scope="Excludes Finkelstein and qc_twin_secondary; valid numeric means analysis_valid and finite yield",
        complete_comparisons=complete, quantitation_broken_comparisons=broken,
        complete_or_quantitation_broken_comparisons=complete + broken,
        fraction_quantitation_broken=broken / (complete + broken),
        other_recorded_comparisons=int(counts.get("other_missing", 0)),
        quantitation_broken_outside_only=outside,
        quantitation_broken_not_outside_only=broken - outside,
        fraction_quantitation_broken_outside_only=outside / broken,
        quantitation_broken_unassigned_pair_group=unassigned,
        quantitation_broken_in_core_represented_pairs=represented,
        quantitation_broken_outside_core=outside + unassigned,
        fraction_quantitation_broken_outside_core=(outside + unassigned) / broken,
        outside_core_definition="outside_only plus pairs with no valid numeric observations and therefore unassigned pair_group; mixed and qualified_only pairs are represented in the core",
        complete_rectangles=rect_complete, quantitation_broken_rectangles=rect_broken,
        fraction_rectangles_quantitation_broken=rect_broken / (rect_complete + rect_broken),
        acylation_quantitation_broken_rectangles=int(by_class.get("Acylation", 0)),
        fraction_broken_rectangles_acylation=float(by_class.get("Acylation", 0) / rect_broken),
        missing_yields_imputed=0,
        interpretation="A broken comparison is an unperformed test; whether it would refute a claim is unknown")


def run(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(source / "processed" / "observations.parquet")
    cells = collapse_cells(observations)
    comparisons = build_comparisons(cells, pd.read_csv(source / "recipe_edges.csv"))
    duals = pd.read_parquet(source / "recipe_duals.parquet", columns=COMPARISON_KEY)
    actual_keys = pd.MultiIndex.from_frame(comparisons.loc[comparisons.comparison_status.eq("complete"), COMPARISON_KEY])
    expected_keys = pd.MultiIndex.from_frame(duals[COMPARISON_KEY])
    assert len(actual_keys.difference(expected_keys)) == len(expected_keys.difference(actual_keys)) == 0
    rectangles = rectangle_catalogue(comparisons)
    full_duals = json.loads((source / "recipe_dual_counts.json").read_text(encoding="utf-8"))["full_duals"]
    summary = summarize(comparisons, rectangles, full_duals)
    comparisons.to_parquet(destination / "measurement_loss_comparisons.parquet", index=False)
    by_contrast = comparisons.groupby(["rxn_class", "edge_id", "comparison_status"]).size().unstack(fill_value=0).reset_index()
    by_contrast.to_csv(destination / "measurement_loss_by_contrast.csv", index=False)
    rectangles.to_csv(destination / "measurement_loss_rectangles.csv", index=False)
    (destination / "measurement_loss_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="clfzero")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.input_dir or out_dir(args.variant)
    print(json.dumps(run(source, args.output_dir or source), indent=2), flush=True)


if __name__ == "__main__":
    main()
