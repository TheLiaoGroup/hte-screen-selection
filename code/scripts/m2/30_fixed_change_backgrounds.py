"""Fixed molecular changes across backgrounds: four-hole comparison census.

Canonical recipe replacements and held contexts define comparisons across
molecular backgrounds. Universe: all valid duals
("extended", including mixed_noncore endpoints). Scopes: full measurements and the
rule union (all four records retained).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from m2common import data_dir, out_dir

TOL = 1e-12
MARGINS = [1, 5, 10]
SCOPES = ["Q", "actual", "full"]
STATES = ["absent", "single_context", "multi_context_exact_zero_only",
          "multi_context_no_strong", "multi_context_single_strong",
          "multi_context_same_strong", "multi_context_opposed_strong"]
AXES = ["partner", "held_recipe"]


def js(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


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
        changed = coordinate_fields(edge.coordinate)
        va, vb = tuple(a[field] for field in changed), tuple(b[field] for field in changed)
        assert va != vb
        held_a = tuple((field, a[field]) for field in a if field not in changed)
        held_b = tuple((field, b[field]) for field in b if field not in changed)
        assert held_a == held_b
        low, high = sorted([va, vb])
        replacement_keys.append(js([edge.rxn_class, edge.coordinate, changed, low, high]))
        held_keys.append(js([edge.rxn_class, edge.coordinate, held_a]))
        signs.append(1 if va == low else -1)
    edges = edges[["edge_id"]].copy()
    edges["replacement"] = pd.factorize(pd.Series(replacement_keys), sort=True)[0]
    edges["held"] = pd.factorize(pd.Series(held_keys), sort=True)[0]
    edges["canonical_sign"] = signs
    return edges


def aggregate_statistics(code, a, b, mask, n):
    def count(hit):
        return np.bincount(code[hit], minlength=n).astype(np.uint32)
    out = {"n_contexts": count(mask), "n_exact_zero": count(mask & (a == 0) & (b == 0))}
    for pp in MARGINS:
        m = pp / 100
        out[f"n_positive_{pp}pp"] = count(mask & (a >= m-TOL) & (b <= -m+TOL))
        out[f"n_negative_{pp}pp"] = count(mask & (a <= -m+TOL) & (b >= m-TOL))
    return out


def states(stats, pp):
    total, zero = stats["n_contexts"], stats["n_exact_zero"]
    pos, neg = stats[f"n_positive_{pp}pp"], stats[f"n_negative_{pp}pp"]
    strong = pos + neg
    out = np.full(len(total), 5, np.uint8)
    out[(pos > 0) & (neg > 0)] = 6
    out[(total >= 2) & (strong == 1)] = 4
    out[(total >= 2) & (strong == 0)] = 3
    out[(total >= 2) & (zero == total)] = 2
    out[total == 1] = 1
    out[total == 0] = 0
    return out


def comparisons(part, role):
    """All pairs of distinct focal molecules sharing one recipe contrast and one common partner."""
    focal, common = f"mol{role}", f"mol{3-role}"
    part = part.sort_values(["edge_code", common, focal], kind="stable")
    index = part.index.to_numpy()
    key = part[["edge_code", common]].to_numpy()
    start = np.flatnonzero(np.r_[True, (key[1:] != key[:-1]).any(axis=1)])
    size = np.diff(np.r_[start, len(part)])
    low, high = [], []
    for s in np.unique(size[size >= 2]):
        a, b = np.triu_indices(s, 1)
        first = start[size == s][:, None]
        low.append((first + a).ravel()); high.append((first + b).ravel())
    low = index[np.concatenate(low)] if low else np.array([], np.int64)
    high = index[np.concatenate(high)] if high else np.array([], np.int64)
    return low, high


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    start_time = time.perf_counter()
    out = out_dir(v)
    edges = edge_dimensions(pd.read_csv(out / "recipe_edges.csv", dtype=str, keep_default_na=False))
    duals = pd.read_parquet(out / "recipe_duals.parquet", columns=[
        "rxn_class", "subpair", "sub_1_smiles", "sub_2_smiles", "edge_id",
        "record_position_A", "record_position_B", "d"])
    duals = duals.merge(edges, on="edge_id", validate="many_to_one")
    upstream = json.loads((out / "recipe_dual_counts.json").read_text(encoding="utf-8"))
    assert len(duals) == upstream["full_duals"]
    if v == "raw":
        assert len(duals) == 845328
    assert not duals.duplicated(["rxn_class", "subpair", "replacement", "held"]).any()
    duals["edge_code"] = pd.factorize(duals.edge_id, sort=True)[0]
    for role in [1, 2]:
        keys = duals.rxn_class.astype(str) + "\t" + duals[f"sub_{role}_smiles"].astype(str)
        duals[f"mol{role}"] = pd.factorize(keys, sort=True)[0]
    canonical_d = (duals.d * duals.canonical_sign).to_numpy()
    obs = pd.read_parquet(data_dir(v) / "observations.parquet", columns=["qualified_row"])
    packed = np.load(out / "rule_masks.npz")
    union = np.unpackbits(packed["rule_union"], bitorder="big")[:int(packed["n_records"])].astype(bool)
    assert len(union) == len(obs)
    pos = duals[["record_position_A", "record_position_B"]].to_numpy()
    dual_scope = {"actual": union[pos].all(axis=1), "Q": obs.qualified_row.to_numpy()[pos].all(axis=1)}
    n_mol = int(max(duals.mol1.max(), duals.mol2.max())) + 1
    rows, n_comparisons = [], 0
    for cls in sorted(duals.rxn_class.unique()):
        part = duals.loc[duals.rxn_class.eq(cls), ["edge_code", "mol1", "mol2"]]
        for role in [1, 2]:
            lo, hi = comparisons(part, role)
            n_comparisons += len(lo)
            focal, common = duals[f"mol{role}"].to_numpy(), duals[f"mol{3-role}"].to_numpy()
            assert np.all(focal[lo] < focal[hi]) and np.array_equal(common[lo], common[hi])
            pair = focal[lo].astype(np.int64) * n_mol + focal[hi]
            replacement = duals.replacement.to_numpy()[lo]
            held, partner = duals.held.to_numpy()[lo], common[lo]
            a, b = canonical_d[lo], canonical_d[hi]
            scope_mask = {s: dual_scope[s][lo] & dual_scope[s][hi] for s in ["Q", "actual"]}
            scope_mask["full"] = np.ones(len(lo), bool)
            for axis in AXES:
                fixed, variable = (held, partner) if axis == "partner" else (partner, held)
                key = np.stack([pair, replacement, fixed], axis=1)
                _, code = np.unique(key, axis=0, return_inverse=True)
                code = code.ravel()
                n = int(code.max()) + 1 if len(code) else 0
                assert len(np.unique(np.c_[key, variable], axis=0)) == len(code)
                stats = {s: aggregate_statistics(code, a, b, scope_mask[s], n) for s in SCOPES}
                for pp in MARGINS:
                    full_state = states(stats["full"], pp)
                    for scope in SCOPES:
                        state = states(stats[scope], pp)
                        same, opposed = state == 5, full_state == 6
                        assert not (same & ~np.isin(full_state, [5, 6])).any()
                        rows.append(dict(axis=axis, rxn_class=cls, changed_role=role, scope=scope, margin_pp=pp,
                            n_comparisons=len(lo), n_fixed_groups=n,
                            n_full_opposed_groups=int(opposed.sum()),
                            n_retained_full_opposed_groups=int((opposed & (state == 6)).sum()),
                            n_selected_same_strong=int(same.sum()),
                            n_apparent_same_strong=int((same & opposed).sum()),
                            **{"n_state_" + label: int((state == i).sum()) for i, label in enumerate(STATES)}))
            print(f"30: {cls} role{role}: {len(lo):,} comparisons", flush=True)
    frame = pd.DataFrame(rows)
    keys = ["axis", "rxn_class", "scope", "margin_pp"]
    values = [c for c in frame if c not in keys + ["changed_role"]]
    summary = frame.groupby(keys, as_index=False)[values].sum()
    total = summary.groupby(["axis", "scope", "margin_pp"], as_index=False)[values].sum()
    total["rxn_class"] = "ALL"
    summary = pd.concat([summary, total[summary.columns]], ignore_index=True).sort_values(keys).reset_index(drop=True)
    summary["apparent_fraction"] = summary.n_apparent_same_strong / summary.n_selected_same_strong.where(summary.n_selected_same_strong > 0)
    summary.to_csv(out / "fixed_change_background_summary.csv", index=False)

    def get(axis, field, cls="ALL", scope="actual", pp=5):
        row = summary.loc[summary.axis.eq(axis) & summary.rxn_class.eq(cls) & summary.scope.eq(scope) & summary.margin_pp.eq(pp)]
        assert len(row) == 1
        return row[field].iloc[0]

    p = summary.loc[summary.axis.eq("partner") & summary.scope.eq("actual") & summary.margin_pp.eq(5) & summary.rxn_class.ne("ALL")]
    by_class = p.set_index("rxn_class").n_apparent_same_strong.sort_values(ascending=False)
    top2 = sorted(by_class.index[:2])
    n_comp = int(get("partner", "n_comparisons", scope="full"))
    assert n_comp == n_comparisons == int(get("held_recipe", "n_comparisons", scope="full"))
    apparent, denominator = int(get("partner", "n_apparent_same_strong")), int(get("partner", "n_selected_same_strong"))
    checks = [
        ("four_hole_comparisons", 11775070, n_comp),
        ("partner_5pp_full_opposed_groups", 35119, int(get("partner", "n_full_opposed_groups"))),
        ("partner_5pp_rule_union_retained_opposed_groups", 21322, int(get("partner", "n_retained_full_opposed_groups"))),
        ("partner_5pp_rule_union_same_strong_groups", 65300, denominator),
        ("partner_5pp_apparent_same_strong_with_full_opposite", 1639, apparent),
        ("partner_5pp_apparent_fraction_pct", 2.51, round(100 * apparent / denominator, 2)),
        ("partner_5pp_apparent_top_two_classes", ["Acylation", "Suzuki"], top2),
        ("held_recipe_5pp_apparent_same_strong_with_full_opposite", 0, int(get("held_recipe", "n_apparent_same_strong"))),
    ]
    result = [dict(name=n, manuscript=m, recomputed=r, match=m == r) for n, m, r in checks]
    result.append(dict(name="partner_5pp_apparent_by_class", manuscript="concentrated in acylation and Suzuki coupling",
                       recomputed={k: int(x) for k, x in by_class.items() if x > 0},
                       match=bool(by_class[["Acylation", "Suzuki"]].sum() / apparent > 0.5 and top2 == ["Acylation", "Suzuki"])))
    (out / "check_background.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"30: completed in {time.perf_counter() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
