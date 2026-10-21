"""Fixed cross-background counterexample targets missing from the rule union.

Target construction minimizes missing record sets. Three goals share the structural reference
table: structural = exact transformation x replacement x held recipe x complete partner;
partner = identity pair x replacement x held recipe (partner varies);
held_recipe = identity pair x replacement x complete partner (remaining recipe varies).
"""
import argparse
from itertools import product
import json

import numpy as np
import pandas as pd
from m2common import out_dir

MARGINS = [1, 5, 10]
HOLES = ["record_position_low_A", "record_position_low_B", "record_position_high_A", "record_position_high_B"]
COMMON = ["rxn_class", "changed_role", "identity_pair_id", "canonical_replacement_id"]
TOL = 1e-12


def encode(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def minimal_sets(choices):
    kept = []
    for choice in sorted(set(map(tuple, choices)), key=lambda x: (len(x), x)):
        s = set(choice)
        if not any(set(old) <= s for old in kept):
            kept.append(choice)
    return tuple(kept)


def goal_codes(refs):
    codes = {"structural": refs.group_position.to_numpy(int)}
    for goal, fixed in [("partner", "held_context_id"), ("held_recipe", "common_partner_id")]:
        codes[goal] = refs.groupby(COMMON + [fixed], sort=True).ngroup().to_numpy(int)
    return codes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    out = out_dir(ap.parse_args().variant)
    refs = pd.read_parquet(out / "structural_refs.parquet")
    n_records = int(np.load(out / "rule_masks.npz")["n_records"])
    actual = np.unpackbits(np.load(out / "rule_masks.npz")["rule_union"], bitorder="big")[:n_records].astype(bool)
    endpoints = refs[HOLES].to_numpy(int)
    selected = actual[endpoints].all(axis=1)
    missing = [tuple(sorted(set(map(int, row[~actual[row]])))) for row in endpoints]
    core = refs.core_ids_json.map(lambda x: json.loads(x)[0]).to_numpy()
    a, b = refs.d_variable_low.to_numpy(), refs.d_variable_high.to_numpy()
    signs = {pp: np.where((a >= pp / 100 - TOL) & (b <= -pp / 100 + TOL), 1,
                          np.where((a <= -pp / 100 + TOL) & (b >= pp / 100 - TOL), -1, 0)) for pp in MARGINS}
    targets, counts = [], {}
    for goal, codes in goal_codes(refs).items():
        order = np.argsort(codes, kind="stable")
        grouped = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        for pp in MARGINS:
            full, retained = 0, 0
            for indices in grouped:
                pos = indices[signs[pp][indices] == 1]
                neg = indices[signs[pp][indices] == -1]
                if not len(pos) or not len(neg):
                    continue
                full += 1
                if goal == "structural":
                    assert refs.identity_pair_id.iloc[indices].is_unique and len(set(core[indices])) == len(indices)
                if selected[pos].any() and selected[neg].any():
                    retained += 1
                    continue
                choices = {tuple(sorted(set(missing[p]) | set(missing[q]))) for p, q in product(pos, neg)}
                assert all(choices)
                minimal = minimal_sets(choices)
                targets.append(dict(target_id=f"{goal}|m{pp}|g{int(codes[indices[0]])}", goal=goal, margin_pp=pp,
                                    group_position=int(codes[indices[0]]), rxn_class=str(refs.rxn_class.iloc[indices[0]]),
                                    n_full_positive=len(pos), n_full_negative=len(neg),
                                    min_single_target_cost=min(map(len, minimal)), candidate_sets_json=encode(minimal)))
            counts[f"{goal}|{pp}"] = dict(full_targets=full, actual_retained=retained, actual_residual=full - retained)
    targets = pd.DataFrame(targets).sort_values(["margin_pp", "goal", "group_position"]).reset_index(drop=True)
    targets.to_csv(out / "crossgoal_targets.csv.gz", index=False,
                   compression={'method': 'gzip', 'mtime': 0})
    res = dict(n_residual_targets=len(targets), by_goal_margin=counts)
    (out / "crossgoal_target_counts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
