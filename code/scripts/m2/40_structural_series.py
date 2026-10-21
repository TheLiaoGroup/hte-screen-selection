"""Exact single-cut structural transformations and their four-hole references.

Single-cut transformations use method default_H, local stratum core >= 6,
both core fractions >= 2/3 and variable fragments <= 8. Inventories connect
molecules, identity pairs, recipe replacements and held contexts.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from m2common import data_dir, out_dir

RDLogger.DisableLog("rdApp.*")
DEFAULT_PATTERN = "[#6+0;!$(*=,#[!#6])]!@!=!#[*]"
TOL = 1e-12
FIELDS = [f"Reagent{i}_{field}" for i in range(1, 7)
          for field in ["id", "HTE_CAS", "Standardized Name", "HTE_Equiv"]]
FIELDS += ["solvent_id", "Standardized Name", "temp_id", "time_id"]
KEYS = ["transformation_id", "canonical_replacement_id", "held_context_id", "common_partner_id"]


def compact(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def identifier(prefix, value):
    return prefix + sha256(compact(value).encode()).hexdigest()[:24]


def identifiers(keys, prefix):
    return {key: f"{prefix}{number:06d}" for number, key in enumerate(sorted(set(keys)), 1)}


def normalized_fragment(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(1 if atom.GetAtomicNum() == 0 else 0)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    order = json.loads(mol.GetProp("_smilesAtomOutputOrder"))
    raw_order = [mol.GetAtomWithIdx(i).GetIntProp("_raw_atom_index")
                 if mol.GetAtomWithIdx(i).HasProp("_raw_atom_index") else -1 for i in order]
    dummy = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    assert len(dummy) == 1 and len(dummy[0].GetNeighbors()) == 1
    anchor = dummy[0].GetNeighbors()[0]
    anchor_raw = anchor.GetIntProp("_raw_atom_index") if anchor.HasProp("_raw_atom_index") else -1
    return smiles, raw_order, anchor_raw


def fragment_molecule(raw_smiles, molecule_id):
    mol = Chem.MolFromSmiles(raw_smiles)
    assert mol is not None and len(Chem.GetMolFrags(mol)) == 1
    for atom in mol.GetAtoms():
        atom.SetIntProp("_raw_atom_index", atom.GetIdx())
    target = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    default_bonds = {mol.GetBondBetweenAtoms(*match).GetIdx()
                     for match in mol.GetSubstructMatches(Chem.MolFromSmarts(DEFAULT_PATTERN), uniquify=False)}
    nonring_bonds = {b.GetIdx() for b in mol.GetBonds() if not b.IsInRing()
                     and b.GetBondType() == Chem.BondType.SINGLE
                     and min(b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()) > 1}
    assert default_bonds <= nonring_bonds
    cuts = [(mol, bond, "heavy", bond in default_bonds) for bond in sorted(nonring_bonds)]
    hydrogenated = Chem.AddHs(mol)
    for atom in hydrogenated.GetAtoms():
        if atom.GetAtomicNum() not in {6, 7, 8, 15, 16}:
            continue
        hydrogen = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 1 and a.GetIsotope() == 0]
        if hydrogen:
            bond = hydrogenated.GetBondBetweenAtoms(atom.GetIdx(), hydrogen[0].GetIdx()).GetIdx()
            cuts.append((hydrogenated, bond, "hydrogen", False))
    rows = []
    for source, bond_index, kind, allowed_default in cuts:
        side = Chem.FragmentOnBonds(source, [bond_index], addDummies=True, dummyLabels=[(1, 1)])
        for atom in side.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
                atom.SetAtomMapNum(1)
        reconstructed = Chem.RemoveHs(Chem.molzip(Chem.Mol(side)))
        valid = Chem.MolToSmiles(reconstructed, isomericSmiles=True) == target
        components = [Chem.RemoveHs(x) for x in Chem.GetMolFrags(side, asMols=True, sanitizeFrags=True)]
        assert len(components) == 2
        for core, variable in [components, components[::-1]]:
            nc = sum(a.GetAtomicNum() > 1 for a in core.GetAtoms())
            nv = sum(a.GetAtomicNum() > 1 for a in variable.GetAtoms())
            if nc == 0:
                continue
            core_smiles, core_order, _ = normalized_fragment(core)
            variable_smiles, variable_order, _ = normalized_fragment(variable)
            if kind == "hydrogen" and nv != 0:
                continue
            key = [molecule_id, kind, bond_index, core_order, variable_order]
            rows.append(dict(fragment_id=identifier("F", key), core_smiles=core_smiles,
                             variable_smiles=variable_smiles, n_core_heavy=nc, n_variable_heavy=nv,
                             allowed_default=allowed_default, hydrogen_cut=kind == "hydrogen",
                             reconstruction_valid=valid))
    assert len({row["fragment_id"] for row in rows}) == len(rows)
    return rows


def maximum_shared_cores(left, right):
    """default_H: default MMPA heavy cuts plus hydrogen substitutions."""
    def index(rows):
        result = defaultdict(list)
        for row in rows:
            if row["reconstruction_valid"] and (row["allowed_default"] or row["hydrogen_cut"]):
                result[row["core_smiles"]].append(row)
        return result
    a, b = index(left), index(right)
    found = {}
    for core in sorted(a.keys() & b.keys()):
        for x in a[core]:
            for y in b[core]:
                if x["variable_smiles"] == y["variable_smiles"]:
                    continue
                key = (core, x["variable_smiles"], y["variable_smiles"])
                if key not in found:
                    found[key] = dict(core_smiles=core, variable_left=key[1], variable_right=key[2],
                                      n_core_heavy=x["n_core_heavy"], n_variable_left=x["n_variable_heavy"],
                                      n_variable_right=y["n_variable_heavy"])
    if not found:
        return []
    maximum = max(row["n_core_heavy"] for row in found.values())
    return [row for _, row in sorted(found.items()) if row["n_core_heavy"] == maximum]


def coordinate_fields(coordinate):
    if coordinate.endswith("_identity"):
        p = "Reagent" + coordinate[1:coordinate.index("_")]
        return [p + "_" + field for field in ["id", "HTE_CAS", "Standardized Name"]]
    if coordinate.endswith("_equiv"):
        return ["Reagent" + coordinate[1:coordinate.index("_")] + "_HTE_Equiv"]
    return {"solvent": ["solvent_id", "Standardized Name"], "temp": ["temp_id"], "time": ["time_id"]}[coordinate]


def edge_dimensions(edges):
    replacement_keys, held_keys, signs = [], [], []
    for edge in edges.itertuples(index=False):
        a, b = json.loads(edge.raw_fields_A_json), json.loads(edge.raw_fields_B_json)
        assert set(a) == set(b) == set(FIELDS)
        changed = coordinate_fields(edge.coordinate)
        va, vb = tuple(a[f] for f in changed), tuple(b[f] for f in changed)
        held_a = tuple((f, a[f]) for f in FIELDS if f not in changed)
        assert va != vb and held_a == tuple((f, b[f]) for f in FIELDS if f not in changed)
        low, high = sorted([va, vb])
        replacement_keys.append(compact([edge.rxn_class, edge.coordinate, changed, low, high]))
        held_keys.append(compact([edge.rxn_class, edge.coordinate, held_a]))
        signs.append(1 if va == low else -1)
    rids, hids = identifiers(replacement_keys, "R"), identifiers(held_keys, "H")
    return pd.DataFrame(dict(edge_id=edges.edge_id, canonical_sign=signs,
                             canonical_replacement_id=[rids[k] for k in replacement_keys],
                             held_context_id=[hids[k] for k in held_keys]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    out = out_dir(v)
    raw = pd.read_parquet(data_dir(v) / "observations.parquet",
                          columns=["rxn_class", "subpair", "sub_1_smiles", "sub_2_smiles", "analysis_valid",
                                   "qualified_row", "pair_group"])
    eligible = raw.loc[raw.analysis_valid & (raw.qualified_row | raw.pair_group.eq("outside_only"))].copy()
    eligible["cohort"] = np.where(eligible.qualified_row, "Q", "O")
    pairs = eligible.groupby(["rxn_class", "subpair", "cohort"]).agg(
        sub_1_smiles=("sub_1_smiles", "first"), sub_2_smiles=("sub_2_smiles", "first")).reset_index()
    assert not pairs.duplicated(["rxn_class", "subpair"]).any()
    mol_ids = identifiers([(r.rxn_class, role, str(getattr(r, f"sub_{role}_smiles")))
                           for r in pairs.itertuples(index=False) for role in [1, 2]], "M")
    for role in [1, 2]:
        pairs[f"molecule_id_role{role}"] = [mol_ids[(c, role, str(s))] for c, s in
                                            pairs[["rxn_class", f"sub_{role}_smiles"]].itertuples(index=False, name=None)]
    smiles = {mid: key[2] for key, mid in mol_ids.items()}
    heavy = {mid: Chem.MolFromSmiles(s).GetNumHeavyAtoms() for mid, s in smiles.items()}
    candidate = defaultdict(set)
    for role in [1, 2]:
        for (cls, held), part in pairs.groupby(["rxn_class", f"molecule_id_role{3 - role}"], sort=True):
            for low, high in combinations(sorted(part[f"molecule_id_role{role}"].unique()), 2):
                candidate[(cls, role, low, high)].add(held)
    candidate_ids = identifiers(candidate, "P")
    print("inventory:", len(pairs), "Q/O pairs;", len(mol_ids), "molecules;", len(candidate_ids), "candidate identity pairs", flush=True)

    fragments = {mid: fragment_molecule(s, mid) for mid, s in smiles.items()}
    assert all(f["reconstruction_valid"] for rows in fragments.values() for f in rows)
    mapping_rows, local = [], {}
    for (cls, role, low, high), pid in candidate_ids.items():
        mappings = maximum_shared_cores(fragments[low], fragments[high])
        if not mappings:
            continue
        nc = mappings[0]["n_core_heavy"]
        frac = min(nc / heavy[low], nc / heavy[high])
        varmax = max(mappings[0]["n_variable_left"], mappings[0]["n_variable_right"])
        local[pid] = bool(nc >= 6 and frac >= 2 / 3 - TOL and varmax <= 8)
        for m in mappings:
            vl, vh = sorted([m["variable_left"], m["variable_right"]])
            mapping_rows.append(dict(identity_pair_id=pid, rxn_class=cls, changed_role=role,
                                     molecule_low_id=low, molecule_high_id=high,
                                     transformation_id=identifier("T", [cls, int(role), vl, vh]),
                                     core_id=identifier("C", m["core_smiles"]),
                                     swap_identity_orientation=m["variable_left"] != vl))
    mappings = pd.DataFrame(mapping_rows)
    memberships = []
    for (pid, tid), g in mappings.groupby(["identity_pair_id", "transformation_id"], sort=True):
        assert g.swap_identity_orientation.nunique() == 1
        first = g.iloc[0]
        memberships.append(dict(identity_pair_id=pid, transformation_id=tid, rxn_class=first.rxn_class,
                                changed_role=int(first.changed_role), molecule_low_id=first.molecule_low_id,
                                molecule_high_id=first.molecule_high_id,
                                swap_identity_orientation=bool(first.swap_identity_orientation),
                                core_ids_json=compact(sorted(g.core_id.unique())), local_6_2thirds_8=local[pid]))
    memberships = pd.DataFrame(memberships)
    memberships["n_identity_pairs_method"] = memberships.groupby("transformation_id").identity_pair_id.transform("nunique")
    series = memberships.loc[memberships.local_6_2thirds_8 & memberships.n_identity_pairs_method.ge(2)].reset_index(drop=True)
    assert not series.duplicated(["identity_pair_id", "transformation_id"]).any()
    print("memberships:", len(memberships), "default_H;", len(series), "local repeated", flush=True)

    edges = edge_dimensions(pd.read_csv(out / "recipe_edges.csv", keep_default_na=False))
    duals = pd.read_parquet(out / "recipe_duals.parquet")
    duals = duals.loc[duals.core_outside_domain].reset_index(drop=True)
    assert duals.identity_A.eq(duals.identity_B).all()
    duals["cohort"] = duals.identity_A
    n = len(duals)
    duals = duals.merge(pairs[["rxn_class", "subpair", "molecule_id_role1", "molecule_id_role2"]],
                        on=["rxn_class", "subpair"], validate="many_to_one")
    duals = duals.merge(edges, on="edge_id", validate="many_to_one")
    assert len(duals) == n
    duals["canonical_d"] = duals.d * duals.canonical_sign
    assert not duals.duplicated(["rxn_class", "subpair", "edge_id"]).any()
    print("frozen duals:", len(duals), duals.cohort.value_counts().to_dict(), flush=True)

    parts = []
    for (cls, role), block in series.groupby(["rxn_class", "changed_role"], sort=True):
        focal, common = f"molecule_id_role{role}", f"molecule_id_role{3 - role}"
        dd = duals.loc[duals.rxn_class.eq(cls)].reset_index(drop=True)
        ec = pd.factorize(dd.edge_id + "|" + dd[common])[0]
        by_mol = {}
        for mid, ix in dd.groupby(focal).indices.items():
            order = np.argsort(ec[ix])
            by_mol[mid] = (ec[ix][order], ix[order])
        for row in block.itertuples(index=False):
            if row.molecule_low_id not in by_mol or row.molecule_high_id not in by_mol:
                continue
            (ka, ia), (kb, ib) = by_mol[row.molecule_low_id], by_mol[row.molecule_high_id]
            _, xa, xb = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
            if not len(xa):
                continue
            lo, hi = dd.iloc[ia[xa]], dd.iloc[ib[xb]]
            parts.append(pd.DataFrame(dict(
                rxn_class=cls, changed_role=role, identity_pair_id=row.identity_pair_id,
                transformation_id=row.transformation_id, swap_identity_orientation=row.swap_identity_orientation,
                core_ids_json=row.core_ids_json, edge_id=lo.edge_id.to_numpy(),
                canonical_replacement_id=lo.canonical_replacement_id.to_numpy(),
                held_context_id=lo.held_context_id.to_numpy(), common_partner_id=lo[common].to_numpy(),
                d_low=lo.canonical_d.to_numpy(), d_high=hi.canonical_d.to_numpy(),
                cohort_low=lo.cohort.to_numpy(), cohort_high=hi.cohort.to_numpy(),
                record_position_low_A=lo.record_position_A.to_numpy(), record_position_low_B=lo.record_position_B.to_numpy(),
                record_position_high_A=hi.record_position_A.to_numpy(), record_position_high_B=hi.record_position_B.to_numpy())))
    refs = pd.concat(parts, ignore_index=True)
    assert not refs.duplicated([*KEYS, "identity_pair_id"]).any()
    refs = refs.sort_values([*KEYS, "identity_pair_id"], kind="stable").reset_index(drop=True)
    refs["reference_position"] = np.arange(len(refs))
    refs["group_position"] = refs.groupby(KEYS, sort=True).ngroup()
    refs["d_variable_low"] = np.where(refs.swap_identity_orientation, refs.d_high, refs.d_low)
    refs["d_variable_high"] = np.where(refs.swap_identity_orientation, refs.d_low, refs.d_high)
    refs["QQ"] = refs.cohort_low.eq("Q") & refs.cohort_high.eq("Q")
    union = np.unpackbits(np.load(out / "rule_masks.npz")["rule_union"], bitorder="big")[:len(raw)].astype(bool)
    holes = refs[["record_position_low_A", "record_position_low_B", "record_position_high_A", "record_position_high_B"]].to_numpy()
    refs["actual_union"] = union[holes].all(axis=1)
    refs.to_parquet(out / "structural_refs.parquet", index=False)
    sizes = refs.groupby("group_position").size()
    res = dict(variant=v, qo_pairs=len(pairs), Q_pairs=int(pairs.cohort.eq("Q").sum()), molecules=len(mol_ids),
               candidate_identity_pairs=len(candidate_ids), mapped_identity_pairs=len(local),
               local_identity_pairs=int(sum(local.values())), default_H_memberships=len(memberships),
               local_repeated_memberships=len(series), frozen_duals=len(duals), references=len(refs),
               exact_groups=int(refs.group_position.nunique()), fixed_target_groups=int(sizes.ge(3).sum()),
               fixed_targets=int(sizes[sizes.ge(3)].sum()), QQ_references=int(refs.QQ.sum()),
               actual_union_references=int(refs.actual_union.sum()))
    (out / "structural_series_counts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
