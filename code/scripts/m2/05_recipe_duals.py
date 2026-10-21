"""Single-coordinate recipe contrasts and paired (dual) comparisons.

A dual is one substrate pair measured under both recipes of one contrast.
"""
import argparse
from collections import defaultdict
from io import StringIO
from itertools import combinations
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from m2common import RAW, data_dir, out_dir

TYPE_MAP = {
    "Acylation": "acylation", "Allylic-substitution": "allylic-substitution",
    "Buchwald-Hartwig": "buchwald", "Buchwald-Hartwig(CMC)": "B-H (CMC)",
    "C-O competitive reactions": "C-N or C-O competitive reactions",
    "Chan-Lam": "chan-lam", "Esterification": "esterification", "Grignard": "grignard",
    "Heck": "heck", "Hiyama": "hiyama", "Kumada": "kumada", "Minisci": "minisci",
    "Pd-carbonylation": "pd-carbonylation", "Reductive-amination": "reductive-amination",
    "SNAr": "snar", "Sonogashira": "sonogashira", "Suzuki": "suzuki",
    "Suzuki Bpin": "suzuki(bpin)", "Ullmann": "ullmann"}
COORDINATES = {}
for slot in range(1, 7):
    COORDINATES[f"R{slot}_identity"] = (f"Reagent{slot}_id", f"Reagent{slot}_HTE_CAS", f"Reagent{slot}_Standardized Name")
    COORDINATES[f"R{slot}_equiv"] = (f"Reagent{slot}_HTE_Equiv",)
COORDINATES.update(solvent=("solvent_id", "Standardized Name"), temp=("temp_id",), time=("time_id",))
COORD_NAMES = list(COORDINATES)
RAW_FIELDS = [field for fields in COORDINATES.values() for field in fields]


def packed(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def condition_sort(value):
    match = re.fullmatch(r"C(\d+)", value)
    return (0, int(match.group(1))) if match else (1, value)


def family(coordinate):
    if coordinate.endswith("_identity"):
        return "reagent_identity"
    if coordinate.endswith("_equiv"):
        return "reagent_equiv"
    return coordinate


def signature(row):
    return tuple(tuple(row[field] for field in fields) for fields in COORDINATES.values())


def field_display(row, fields):
    return packed({field: row[field] if row[field] != "" else "<MISSING>" for field in fields})


def recipe(row):
    terms = []
    for slot in range(1, 7):
        name = row.get(f"Reagent{slot}_Standardized Name")
        equiv = row.get(f"Reagent{slot}_HTE_Equiv")
        if pd.notna(name):
            terms.append(f"R{slot}: {name}" + (f" ({equiv:g} eq)" if pd.notna(equiv) else ""))
    solvent = row.get("Standardized Name")
    if pd.notna(solvent):
        terms.append(str(solvent))
    terms.extend([f"{row['temp_id']:g} C", f"{row['time_id']:g} h"])
    return "; ".join(terms)


def master_conditions():
    catalogue = Path(os.environ.get('M2_CONDITION_CATALOGUE',
                                   RAW / '0_total condition_260810.xlsx'))
    master = pd.read_excel(catalogue, sheet_name="Sheet2")
    master = master.loc[master["reaction_type"].notna()].copy()
    assert not master.duplicated(["reaction_type", "condition_id"]).any()
    master["recipe"] = master.apply(recipe, axis=1)
    # Round-trip through CSV to compare catalogue fields as exact strings.
    return pd.read_csv(StringIO(master.to_csv(index=False)), dtype=str, keep_default_na=False)


def make_graph(master, classes):
    lookups, edge_rows, aliases, inventory = {}, [], [], []
    independently_checked_pairs = 0
    for cls in sorted(classes):
        rows = master.loc[master.reaction_type.eq(TYPE_MAP[cls])].to_dict("records")
        lookup = {row["condition_id"]: row for row in rows}
        assert len(lookup) == len(rows) and len(rows)
        lookups[cls] = lookup
        sigs = {cid: signature(row) for cid, row in lookup.items()}
        found = {}
        for position, coordinate in enumerate(COORD_NAMES):
            buckets = defaultdict(list)
            for cid, values in sigs.items():
                buckets[values[:position] + values[position + 1:]].append(cid)
            for members in buckets.values():
                for a, b in combinations(sorted(members, key=condition_sort), 2):
                    if sigs[a][position] != sigs[b][position]:
                        assert (a, b) not in found
                        found[(a, b)] = coordinate
        brute = {}
        n_identical = 0
        for a, b in combinations(sorted(lookup, key=condition_sort), 2):
            changed = [name for name, av, bv in zip(COORD_NAMES, sigs[a], sigs[b]) if av != bv]
            if len(changed) == 1:
                brute[(a, b)] = changed[0]
            elif not changed:
                n_identical += 1
                aliases.append({"rxn_class": cls, "condition_A": a, "condition_B": b,
                                "recipe_A": lookup[a]["recipe"], "recipe_B": lookup[b]["recipe"],
                                "note_A": lookup[a]["note"], "note_B": lookup[b]["note"]})
            independently_checked_pairs += 1
        assert found == brute
        for (a, b), coordinate in sorted(found.items(), key=lambda x: (condition_sort(x[0][0]), condition_sort(x[0][1]))):
            ra, rb = lookup[a], lookup[b]
            fields = COORDINATES[coordinate]
            id_change = cas_change = name_change = False
            solvent_id_change = solvent_name_change = False
            label_only_identity = label_only_solvent = False
            if coordinate.endswith("_identity"):
                id_change, cas_change, name_change = [ra[field] != rb[field] for field in fields]
                label_only_identity = not id_change and not cas_change and name_change
            elif coordinate == "solvent":
                solvent_id_change, solvent_name_change = [ra[field] != rb[field] for field in fields]
                label_only_solvent = not solvent_id_change and solvent_name_change
            held = [name for name in COORD_NAMES if name != coordinate]
            edge_rows.append({"rxn_class": cls, "reaction_type": TYPE_MAP[cls],
                "edge_id": f"{cls}|{a}|{b}", "condition_A": a, "condition_B": b,
                "coordinate": coordinate, "coordinate_family": family(coordinate),
                "changed_fields_A_json": field_display(ra, fields),
                "changed_fields_B_json": field_display(rb, fields),
                "identity_id_changed": id_change, "identity_CAS_changed": cas_change,
                "identity_name_changed": name_change,
                "solvent_id_changed": solvent_id_change, "solvent_name_changed": solvent_name_change,
                "label_only_identity_change": label_only_identity,
                "label_only_solvent_change": label_only_solvent,
                "label_only_change": label_only_identity or label_only_solvent,
                "changed_coordinate_has_missing": any(ra[f] == "" or rb[f] == "" for f in fields),
                "n_held_coordinates_with_missing": sum(any(ra[f] == "" for f in COORDINATES[name]) for name in held),
                "both_notes_empty": ra["note"] == "" and rb["note"] == "",
                "notes_differ": ra["note"] != rb["note"], "note_A": ra["note"], "note_B": rb["note"],
                "recipe_A": ra["recipe"], "recipe_B": rb["recipe"],
                "raw_fields_A_json": packed({f: ra[f] for f in RAW_FIELDS}),
                "raw_fields_B_json": packed({f: rb[f] for f in RAW_FIELDS})})
        inventory.append({"rxn_class": cls, "reaction_type": TYPE_MAP[cls], "n_master_conditions": len(rows),
                          "n_single_coordinate_edges": len(found), "n_zero_coordinate_condition_pairs": n_identical})
    edges = pd.DataFrame(edge_rows)
    assert not edges.duplicated(["rxn_class", "condition_A", "condition_B"]).any()
    for item in inventory:
        g = edges.loc[edges.rxn_class.eq(item["rxn_class"])]
        item.update(n_non_label_only_edges=int((~g.label_only_change).sum()),
                    n_label_only_identity_edges=int(g.label_only_identity_change.sum()),
                    n_label_only_solvent_edges=int(g.label_only_solvent_change.sum()),
                    n_non_label_only_both_notes_empty_edges=int((~g.label_only_change & g.both_notes_empty).sum()))
    return edges, lookups, independently_checked_pairs


def write_duals(dual, path):
    stable = dual.astype({'record_id_A': object, 'record_id_B': object})
    stable.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    out = out_dir(v)
    raw = pd.read_parquet(data_dir(v) / "observations.parquet")
    raw["record_position"] = np.arange(len(raw))
    valid = raw.loc[raw.analysis_valid].copy()
    edges, lookups, _ = make_graph(master_conditions(), valid.rxn_class.unique())
    assert not edges.label_only_change.any()
    edges.to_csv(out / "recipe_edges.csv", index=False)
    valid["identity"] = np.select([valid.qualified_row, valid.pair_group.eq("outside_only")], ["Q", "O"], "mixed_noncore")
    keep = ["rxn_class", "subpair", "sub_1_smiles", "sub_2_smiles"]
    rec = ["record_position", "record_id", "yield", "identity", "source_campaign", "plate_id"]
    a = valid[keep + ["condition_id"] + rec].rename(columns={"condition_id": "condition_A", **{f: f + "_A" for f in rec}})
    b = valid[["rxn_class", "subpair", "condition_id"] + rec].rename(columns={"condition_id": "condition_B", **{f: f + "_B" for f in rec}})
    dual = a.merge(edges[["rxn_class", "edge_id", "condition_A", "condition_B", "coordinate", "coordinate_family"]],
                   on=["rxn_class", "condition_A"]).merge(b, on=["rxn_class", "subpair", "condition_B"])
    assert not dual.duplicated(["rxn_class", "subpair", "edge_id"]).any()
    dual["d"] = dual.yield_A - dual.yield_B
    dual = dual.sort_values(["rxn_class", "edge_id", "subpair"], kind="stable").reset_index(drop=True)
    dual["core_outside_domain"] = ~(dual.identity_A.eq("mixed_noncore") | dual.identity_B.eq("mixed_noncore"))
    write_duals(dual, out / "recipe_duals.parquet")
    res = dict(variant=v, recipe_contrasts=len(edges), classes_with_contrasts=int(edges.rxn_class.nunique()),
               full_duals=len(dual), core_outside_duals=int(dual.core_outside_domain.sum()),
               excluded_mixed_noncore_duals=int((~dual.core_outside_domain).sum()),
               contrasts_with_duals=int(dual.edge_id.nunique()))
    (out / "recipe_dual_counts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
