"""Fixed held-out directional judgments and cross-goal recovery for retained record sets.

Leave-out replay targets are all
references in exact structural groups with >= 3 identity pairs; a training reference is a member of
the same group with a different identity pair/core that shares none of the target's four records; a
judgment needs >= 2 retained training references (each one distinct pair and core), and the majority
of strong training events gives the direction (ties/no strong event abstain). Cross-goal recovery
uses the same fixed target groups: a missing (actual-residual) target is restored
when the record set retains a positive and a negative reference of its group.
Writes out/heldout_replay.csv and out/check_heldout.json.
"""
import argparse
from importlib import import_module
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from m2common import data_dir, out_dir

goal_codes = import_module("41_crossgoal_targets").goal_codes
HOLES = ["record_position_low_A", "record_position_low_B", "record_position_high_A", "record_position_high_B"]
GOALS = ["structural", "partner", "held_recipe"]
TOL = 1e-12


def unpack(packed, n):
    return np.unpackbits(packed, bitorder="big")[:n].astype(bool)


def events(low, high, pp):
    return np.where((low >= pp / 100 - TOL) & (high <= -pp / 100 + TOL), 1,
                    np.where((low <= -pp / 100 + TOL) & (high >= pp / 100 - TOL), -1, 0)).astype(np.int8)


def leaveout_graph(refs):
    sizes = refs.groupby("group_position").size()
    targets = refs.loc[refs.group_position.map(sizes).ge(3)].reset_index(drop=True)
    endpoint = targets[HOLES].to_numpy(int)
    raw_sets = [set(row) for row in endpoint]
    rr, cc = [], []
    for positions in targets.groupby("group_position", sort=True).indices.values():
        for t in positions:
            for u in positions:
                if t != u and not raw_sets[t] & raw_sets[u]:
                    rr.append(t); cc.append(u)
    rr, cc = np.asarray(rr), np.asarray(cc)
    assert (targets.identity_pair_id.to_numpy()[rr] != targets.identity_pair_id.to_numpy()[cc]).all()
    core = targets.core_ids_json.to_numpy()
    assert (core[rr] != core[cc]).all()
    adjacency = csr_matrix((np.ones(len(rr), np.int32), (rr, cc)), shape=(len(targets), len(targets)))
    return targets, endpoint, adjacency


def leaveout(masks, targets, endpoint, adjacency, pp=5):
    keep = masks[:, endpoint].all(axis=2)
    truth = events(targets.d_variable_low.to_numpy(), targets.d_variable_high.to_numpy(), pp)
    n_train = np.asarray(adjacency @ keep.T.astype(np.int32)).T
    positive = np.asarray(adjacency @ (keep & (truth == 1)).T.astype(np.int32)).T
    negative = np.asarray(adjacency @ (keep & (truth == -1)).T.astype(np.int32)).T
    prediction = np.where(n_train >= 2, np.sign(positive - negative), 0)
    strong = truth != 0
    answered = strong & (prediction != 0)
    correct = answered & (prediction == truth)
    return int(strong.sum()), answered.sum(axis=1), correct.sum(axis=1)


def crossgoal(masks, refs, codes, residual, pp=5):
    keep = masks[:, refs[HOLES].to_numpy(int)].all(axis=2)
    sign = events(refs.d_variable_low.to_numpy(), refs.d_variable_high.to_numpy(), pp)
    result = {}
    for goal in GOALS:
        c, n = codes[goal], int(codes[goal].max()) + 1
        pos = np.vstack([np.bincount(c[k & (sign == 1)], minlength=n) for k in keep])
        neg = np.vstack([np.bincount(c[k & (sign == -1)], minlength=n) for k in keep])
        groups = residual.loc[residual.goal.eq(goal), "group_position"].to_numpy(int)
        result[goal] = ((pos[:, groups] > 0) & (neg[:, groups] > 0)).sum(axis=1)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    out, start = out_dir(v), time.time()
    raw = pd.read_parquet(data_dir(v) / "observations.parquet", columns=["analysis_valid"])
    n = len(raw)
    actual = unpack(np.load(out / "rule_masks.npz")["rule_union"], n)
    full = raw.analysis_valid.to_numpy(bool)
    refs = pd.read_parquet(out / "structural_refs.parquet")
    residual = pd.read_csv(out / "crossgoal_targets.csv.gz", keep_default_na=False)
    residual = residual.loc[residual.margin_pp.eq(5)]
    joint_pref = json.loads((out / "joint_preference_counterexample.json").read_text(encoding="utf-8"))["solutions"]

    def plus(positions):
        mask = actual.copy(); mask[np.asarray(positions, int)] = True
        return mask

    sets = {"rule_union": actual, "full": full}
    with np.load(out / "restoration_joint_returned_mask.npz") as packed:
        sets["joint_full_strength_M2"] = unpack(packed["mask"], n)
    sets["joint_preference_m5_f1_M2"] = plus(joint_pref["preference_m5_f1_three_goals"]["selected_record_positions"])
    family_path = out / "restoration_joint_family.csv"
    n_family = 0
    if family_path.exists():
        family = pd.read_csv(family_path)
        base = family.loc[family.row_type.eq("record") & family.in_I_star.astype(bool), "record_position"].astype(int)
        opt = family.loc[family.row_type.eq("optimum")]
        for sid, part in opt.groupby("set_id", sort=True):
            sets[f"family_{int(sid)}"] = plus(np.r_[base.to_numpy(), part.record_position.astype(int).to_numpy()])
            n_family += 1
    names = list(sets)
    masks = np.vstack([sets[k] for k in names])

    targets, endpoint, adjacency = leaveout_graph(refs)
    n_strong, answered, correct = leaveout(masks, targets, endpoint, adjacency)
    recovered = crossgoal(masks, refs, goal_codes(refs), residual)
    table = pd.DataFrame(dict(recordset=names, n_added_to_actual=(masks & ~actual).sum(axis=1),
                              answered=answered, correct=correct, coverage=answered / n_strong,
                              **{f"recovered_{g}": recovered[g] for g in GOALS}))
    table.to_csv(out / "heldout_replay.csv", index=False)
    row = table.set_index("recordset")
    n_res = residual.goal.value_counts().to_dict()

    def entry(name, manuscript, recomputed):
        recomputed = recomputed.item() if hasattr(recomputed, "item") else recomputed
        return dict(name=name, manuscript=manuscript, recomputed=recomputed, match=manuscript == recomputed)

    pct = lambda k: round(100 * float(row.loc[k, "coverage"]), 1)
    joint = "joint_full_strength_M2"
    checks = [entry("fixed held-out targets", 20391, len(targets)),
              entry("strong targets at 5 pp", 1109, n_strong),
              entry("rule union answered", 69, row.loc["rule_union", "answered"]),
              entry("rule union correct", 36, row.loc["rule_union", "correct"]),
              entry("rule union coverage %", 6.2, pct("rule_union")),
              entry("joint full-strength answered", 72, row.loc[joint, "answered"]),
              entry("joint full-strength correct", 38, row.loc[joint, "correct"]),
              entry("joint full-strength coverage %", 6.5, pct(joint)),
              entry("full measured answered", 118, row.loc["full", "answered"]),
              entry("full measured correct", 59, row.loc["full", "correct"]),
              entry("full measured coverage %", 10.6, pct("full")),
              entry("joint full-strength added records", 2224, row.loc[joint, "n_added_to_actual"]),
              entry("missing structural counterexample groups", 50, n_res.get("structural", 0)),
              entry("missing partner-background groups", 227, n_res.get("partner", 0)),
              entry("missing remaining-recipe groups", 48, n_res.get("held_recipe", 0)),
              entry("joint recovery restores structural", 3, row.loc[joint, "recovered_structural"]),
              entry("joint recovery restores partner", 29, row.loc[joint, "recovered_partner"]),
              entry("joint recovery restores remaining recipe", 2, row.loc[joint, "recovered_held_recipe"]),
              entry("preference+counterexample recovery records", 1018,
                    joint_pref["preference_m5_f1_three_goals"]["minimum_additions"]),
              entry("1,018 recovery restores structural", 50, row.loc["joint_preference_m5_f1_M2", "recovered_structural"]),
              entry("1,018 recovery restores partner", 227, row.loc["joint_preference_m5_f1_M2", "recovered_partner"]),
              entry("1,018 recovery restores remaining recipe", 48, row.loc["joint_preference_m5_f1_M2", "recovered_held_recipe"])]
    res = dict(checks=checks, all_match=all(c["match"] for c in checks), n_family_optima=n_family)
    if n_family:
        fam = table.loc[table.recordset.str.startswith("family_")]
        res["family_range"] = {c: [int(fam[c].min()), int(fam[c].max())] for c in
                               ["answered", "correct", *[f"recovered_{g}" for g in GOALS]]}
    res["other_recordsets"] = table.loc[~table.recordset.str.startswith("family_")].drop(columns="coverage").to_dict("records")
    (out / "check_heldout.json").write_text(json.dumps(res, indent=1, default=int), encoding="utf-8")
    print(json.dumps({k: res[k] for k in res if k != "checks"}, indent=1, default=int))
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
