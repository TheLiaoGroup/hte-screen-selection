"""Select the strongest weakest-link four-combination chemical counterexample.

Candidate cycles have three qualified-core comparisons and exactly one
outside-only comparison, within contrasts whose minimum omitted count is one.
Selection never reads the review's example identities or yield checkpoints.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from m2common import RAW, out_dir


def directed_arc(row):
    left, right = (1, str(row.sub_1_smiles)), (2, str(row.sub_2_smiles))
    return (right, left) if row.d > 0 else (left, right)


def rank_four_cycles(duals, target_ids):
    """Enumerate O arcs closed by three Q arcs, then rank unrounded minima."""
    selected = duals.loc[duals.edge_id.isin(target_ids) & duals.core_outside_domain &
                         duals.d.abs().ge(.05 - 1e-12)].copy()
    assert selected.identity_A.eq(selected.identity_B).all()
    selected["dual_id"] = selected.record_id_A + "|" + selected.record_id_B
    candidates = {}
    for edge_id, group in selected.groupby("edge_id", sort=True):
        assert not group.duplicated(["sub_1_smiles", "sub_2_smiles"]).any()
        adjacency = defaultdict(list)
        by_id = group.set_index("dual_id")
        outside = []
        for row in group.itertuples(index=False):
            start, end = directed_arc(row)
            if row.identity_A == "Q":
                adjacency[start].append((end, row.dual_id))
            elif row.identity_A == "O":
                outside.append((start, end, row.dual_id))
            else:
                raise ValueError("Candidate domain contains an unexpected cohort")
        for arcs in adjacency.values():
            arcs.sort()
        for start, end, omitted_id in sorted(outside):
            for middle1, id1 in adjacency[end]:
                if middle1 == start:
                    continue
                for middle2, id2 in adjacency[middle1]:
                    if middle2 in (start, end):
                        continue
                    for finish, id3 in adjacency[middle2]:
                        if finish != start:
                            continue
                        ids = tuple(sorted([omitted_id, id1, id2, id3]))
                        rows = by_id.loc[list(ids)]
                        assert len(rows) == 4 and rows.identity_A.eq("O").sum() == 1
                        assert rows.sub_1_smiles.nunique() == rows.sub_2_smiles.nunique() == 2
                        key = (edge_id, ids)
                        minimum = float(rows.d.abs().min() * 100)
                        candidates[key] = dict(edge_id=edge_id, rxn_class=rows.rxn_class.iloc[0],
                            dual_ids=ids, omitted_dual_id=omitted_id,
                            min_abs_yield_difference_pct=minimum,
                            cycle_id=hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:16])
    ranked = sorted(candidates.values(), key=lambda row:
                    (-row["min_abs_yield_difference_pct"], row["edge_id"], row["dual_ids"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def catalogue_path():
    explicit = os.environ.get("M2_COMPOUND_CATALOGUE")
    if explicit:
        return Path(explicit)
    condition = os.environ.get("M2_CONDITION_CATALOGUE")
    return ((Path(condition).parent if condition else RAW) / "0_Compound ID_260810.xlsx")


def cas_key(value):
    return str(value).strip().lstrip("'\"\u2018\u2019")


def compound_names(catalogue):
    frame = pd.read_excel(catalogue, dtype=str, keep_default_na=False)
    assert {"cas", "Standardized Name", "Compound ID"} <= set(frame)
    names = defaultdict(set)
    for row in frame.to_dict("records"):
        if row["cas"] and row["Standardized Name"]:
            names[cas_key(row["cas"])].add(row["Standardized Name"])
    return {key: sorted(values) for key, values in names.items()}


def molecule_record(smiles, role, records, names):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Selected substrate SMILES does not parse")
    rows = records.loc[records[f"sub_{role}_smiles"].eq(smiles)]
    cas = sorted({cas_key(value) for value in rows[f"sub_{role}_cas"].dropna() if str(value).strip()})
    labels = sorted({name for value in cas for name in names.get(value, [])})
    functional_class = ("amine" if molecule.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H1,H2]"))
                        else "aryl_triflate" if molecule.HasSubstructMatch(Chem.MolFromSmarts("cOS(=O)(=O)C(F)(F)F"))
                        else "substrate")
    return dict(role=role, smiles=smiles, cas=cas, catalogue_names=labels,
                display_name=labels[0] if labels else ("CAS " + ", ".join(cas) if cas else smiles),
                name_source="compound catalogue CAS match" if labels else "CAS/SMILES only; no catalogue name",
                functional_class=functional_class,
                canonical_isomeric_smiles_for_drawing=Chem.MolToSmiles(molecule, isomericSmiles=True))


def changed_label(fields):
    names = [value for key, value in fields.items() if key.endswith("Standardized Name") and value]
    return "; ".join(names) or json.dumps(fields, sort_keys=True)


def selected_details(candidate, duals, observations, recipe_edges, names):
    duals = duals.assign(dual_id=duals.record_id_A + "|" + duals.record_id_B).set_index("dual_id")
    rows = duals.loc[list(candidate["dual_ids"])].sort_values(["sub_1_smiles", "sub_2_smiles"], kind="stable")
    observed = observations.set_index("record_id")
    record_ids = list(rows.record_id_A) + list(rows.record_id_B)
    source = observed.loc[record_ids]
    assert source.analysis_valid.all() and not source.qc_twin_secondary.any()
    edge = recipe_edges.set_index("edge_id").loc[candidate["edge_id"]]
    molecules = {str(role): [molecule_record(smiles, role, source, names)
                           for smiles in sorted(rows[f"sub_{role}_smiles"].unique())]
                 for role in [1, 2]}
    for role, items in molecules.items():
        classes = {item["functional_class"] for item in items}
        prefix = {"amine": "Amine", "aryl_triflate": "Triflate"}.get(next(iter(classes)), "Substrate") if len(classes) == 1 else "Substrate"
        for number, item in enumerate(items, 1):
            item["short_label"] = f"{prefix}{number}"
    labels = {role: {item["smiles"]: item for item in molecules[str(role)]} for role in [1, 2]}
    reagent_labels = {side: changed_label(json.loads(edge[f"changed_fields_{side}_json"])) for side in "AB"}
    records = []
    for row in rows.itertuples():
        result = dict(edge_id=candidate["edge_id"], cycle_id=candidate["cycle_id"],
            selection_rank=candidate["rank"], rxn_class=candidate["rxn_class"],
            sub_1_smiles=row.sub_1_smiles, sub_2_smiles=row.sub_2_smiles,
            sub_1_label=labels[1][row.sub_1_smiles]["display_name"],
            sub_2_label=labels[2][row.sub_2_smiles]["display_name"],
            sub_1_short_label=labels[1][row.sub_1_smiles]["short_label"],
            sub_2_short_label=labels[2][row.sub_2_smiles]["short_label"],
            sub_1_cas=";".join(labels[1][row.sub_1_smiles]["cas"]),
            sub_2_cas=";".join(labels[2][row.sub_2_smiles]["cas"]),
            difference_A_minus_B_pct=float(row.d * 100),
            absolute_yield_difference_pct=float(abs(row.d) * 100),
            kept_in_qualified_core=row.identity_A == "Q",
            retention_status="kept" if row.identity_A == "Q" else "excluded_from_qualified_core")
        for side in "AB":
            record_id = getattr(row, f"record_id_{side}")
            measurement = observed.loc[record_id]
            assert measurement.subpair == row.subpair
            assert measurement.condition_id == edge[f"condition_{side}"]
            expected = getattr(row, f"yield_{side}")
            assert np.isclose(measurement["yield"], expected, rtol=0, atol=1e-15)
            assert bool(measurement.qualified_row) == (row.identity_A == "Q")
            result.update({f"record_id_{side}": record_id, f"name0_{side}": str(measurement.name0),
                f"source_file_{side}": str(measurement.source_file),
                f"source_row_{side}": int(measurement.source_row),
                f"condition_{side}": str(edge[f"condition_{side}"]),
                f"reagent_{side}": reagent_labels[side], f"recipe_{side}": str(edge[f"recipe_{side}"]),
                f"yield_{side}_fraction": float(expected), f"yield_{side}_pct": float(expected * 100),
                f"outside_reason_{side}": str(measurement.outside_reason)})
        records.append(result)
    assert sum(row["kept_in_qualified_core"] for row in records) == 3
    return {**candidate, "dual_ids": list(candidate["dual_ids"]),
            "condition_A": edge.condition_A, "condition_B": edge.condition_B,
            "recipe_A": edge.recipe_A, "recipe_B": edge.recipe_B,
            "reagent_A": reagent_labels["A"], "reagent_B": reagent_labels["B"],
            "raw_condition_fields_A": json.loads(edge.raw_fields_A_json),
            "raw_condition_fields_B": json.loads(edge.raw_fields_B_json),
            "molecules_by_role": molecules, "reactions": records}, pd.DataFrame(records)


def selected_certificate_ranking(partner, duals):
    """Audit the separate universe of one selected four-cycle per contrast."""
    measured = duals.assign(dual_id=duals.record_id_A + "|" + duals.record_id_B).set_index("dual_id")
    rows = []
    for row in partner.loc[partner.new_contradiction & partner.certificate_length.eq(4)].itertuples(index=False):
        ids = tuple(sorted(json.loads(row.certificate_dual_ids)))
        certificate = measured.loc[list(ids)]
        assert len(certificate) == 4 and certificate.edge_id.eq(row.edge_id).all()
        rows.append(dict(edge_id=row.edge_id, minimum_O=int(row.minimum_O), dual_ids=ids,
                         min_abs_yield_difference_pct=float(certificate.d.abs().min() * 100)))
    rows.sort(key=lambda item: (-item["min_abs_yield_difference_pct"], item["edge_id"], item["dual_ids"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def run(source, destination, catalogue):
    partner = pd.read_csv(source / "partner_score_edges.csv")
    single = partner.loc[partner.new_contradiction & partner.minimum_O.eq(1)]
    duals = pd.read_parquet(source / "recipe_duals.parquet")
    assert np.allclose(duals.d, duals.yield_A - duals.yield_B, rtol=0, atol=1e-15)
    ranked = rank_four_cycles(duals, set(single.edge_id))
    if not ranked:
        raise ValueError("No eligible four-combination counterexample")
    old = pd.read_csv(source / "partner_score_flagship.csv")
    old_ids = tuple(sorted(old.record_id_A + "|" + old.record_id_B))
    previous = [row for row in ranked if row["dual_ids"] == old_ids]
    assert len(previous) == 1, "Previous flagship is not a unique ranked eligible cycle"
    previous = previous[0]
    selected_certificates = selected_certificate_ranking(partner, duals)
    previous_selected = [row for row in selected_certificates if row["dual_ids"] == old_ids]
    context = dict(
        definition="One deterministically selected minimum four-combination certificate per new contrast, including collective certificates; a different universe from all eligible single-omission cycles",
        four_combination_selected_certificates=len(selected_certificates),
        single_omission_selected_certificates=sum(row["minimum_O"] == 1 for row in selected_certificates),
        previous_flagship_exact_match=bool(previous_selected),
        previous_flagship_score_rank=1 + sum(row["min_abs_yield_difference_pct"] > previous["min_abs_yield_difference_pct"] for row in selected_certificates),
        previous_flagship_exact_selected_rank=previous_selected[0]["rank"] if previous_selected else None)
    observations = pd.read_parquet(source / "processed" / "observations.parquet")
    selected, table = selected_details(ranked[0], duals, observations,
                                      pd.read_csv(source / "recipe_edges.csv"), compound_names(catalogue))
    result = dict(selection_rule={
        "scope": "New counterexample contrasts whose minimum number of outside-only comparisons is one",
        "candidate": "Directed four-combination cycle with three Q comparisons and one O comparison at the five-percentage-point strength threshold",
        "objective": "Maximize the minimum unrounded absolute yield difference among the four comparisons",
        "ties": "Lexicographic edge ID, then the sorted full source-record dual IDs",
        "interpretation": "Each combination is measured under both recipes; the selected cycle therefore contains eight recorded wells"},
        eligible_single_omitted_contrasts=len(single), candidate_four_reaction_cycles=len(ranked),
        contrasts_with_four_reaction_cycles=len({row["edge_id"] for row in ranked}),
        selected=selected, previous_flagship={**previous, "dual_ids": list(previous["dual_ids"])},
        selected_certificate_context=context,
        molecule_drawing_note="RDKit-ready exact recorded structures; drawings may be replaced with ChemDraw",
        name_catalogue=catalogue.name)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "chemical_example.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    table.to_csv(destination / "chemical_example.csv", index=False, float_format="%.17g")
    catalogue_rows = [{**row, "dual_ids": json.dumps(row["dual_ids"])} for row in ranked]
    pd.DataFrame(catalogue_rows).to_csv(destination / "chemical_example_candidates.csv", index=False, float_format="%.17g")
    pd.DataFrame([{**row, "dual_ids": json.dumps(row["dual_ids"])} for row in selected_certificates]).to_csv(
        destination / "chemical_example_selected_certificates.csv", index=False, float_format="%.17g")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="clfzero")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compound-catalogue", type=Path)
    args = parser.parse_args()
    source = args.input_dir or out_dir(args.variant)
    result = run(source, args.output_dir or source, args.compound_catalogue or catalogue_path())
    print(json.dumps({"eligible_contrasts": result["eligible_single_omitted_contrasts"],
        "candidate_cycles": result["candidate_four_reaction_cycles"],
        "selected_edge": result["selected"]["edge_id"],
        "selected_minimum_pct": result["selected"]["min_abs_yield_difference_pct"],
        "previous_rank": result["previous_flagship"]["rank"],
        "previous_minimum_pct": result["previous_flagship"]["min_abs_yield_difference_pct"],
        "selected_certificate_context": result["selected_certificate_context"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
