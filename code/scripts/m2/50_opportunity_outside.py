"""Measured opportunity outside qualified screens.

Compute exact B=12 expected best outcomes and within-class Q75/Q90 tails.
A pair is visible when any record is in any of the 360 construction masks.
Nearest qualified pairs use fingerprints; shared measured conditions define
local comparisons.
"""
import argparse
from functools import lru_cache
import json
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import gammaln
from m2common import data_dir, out_dir

RDLogger.DisableLog("rdApp.*")
KEYS = ["rxn_class", "subpair"]
BUDGET = 12
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MANUSCRIPT = {"q75_outside_tail": 534, "q75_tail_absent_from_rule_union": 472,
              "q90_outside_tail": 247, "q90_tail_absent_from_rule_union": 210,
              "outside_absent_from_rule_union": 1486,
              "absent_sharing_ge12_with_nearest_qualified": 1059}


def log_choose(n, b):
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore"):
        return gammaln(n + 1) - gammaln(b + 1) - gammaln(n - b + 1)


@lru_cache(maxsize=None)
def max_weights(n, b):
    # The largest selected order index j has C(j-1,b-1)/C(n,b) mass.
    j = np.arange(b, n + 1)
    w = np.exp(log_choose(j - 1, b - 1) - log_choose(n, b))
    assert abs(w.sum() - 1) < 1e-9
    return w / w.sum()


def expected_best(values, b=BUDGET):
    y = np.sort(np.asarray(values, dtype=float))
    return float(np.dot(max_weights(len(y), b), y[b - 1:]))


def fingerprint(smiles, cache):
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if text not in cache:
        mol = Chem.MolFromSmiles(text) if text else None
        cache[text] = GEN.GetFingerprint(mol) if mol is not None and mol.GetNumAtoms() else None
    return cache[text]


def pair_table(frame):
    return frame.groupby(KEYS, as_index=False, sort=True).agg(
        sub_1_smiles=("sub_1_smiles", "first"), sub_2_smiles=("sub_2_smiles", "first"),
        n_conditions=("condition_id", "nunique"))


def nearest_reference(outside, anchors):
    rows = []
    for cls, queries in outside.groupby("rxn_class", sort=True):
        q = anchors.loc[anchors.rxn_class.eq(cls) & anchors.valid_fp].reset_index(drop=True)
        q1, q2 = list(q.fp1), list(q.fp2)
        for query in queries.itertuples(index=False):
            similarity, nearest = np.nan, ""
            if query.valid_fp and len(q):
                s = (np.asarray(DataStructs.BulkTanimotoSimilarity(query.fp1, q1))
                     + np.asarray(DataStructs.BulkTanimotoSimilarity(query.fp2, q2))) / 2
                pos = int(s.argmax())
                similarity, nearest = float(s[pos]), q.subpair.iloc[pos]
            rows.append({"rxn_class": cls, "subpair": query.subpair,
                         "nearest_q_similarity": similarity, "nearest_q_subpair": nearest})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    t0 = time.time()
    u = pd.read_parquet(data_dir(v) / "observations.parquet")
    d = u.loc[u.analysis_valid].copy()
    assert not d.duplicated(KEYS + ["condition_id"]).any() and d["yield"].between(0, 1).all()
    q_cells = d.loc[d.qualified_row]
    o_cells = d.loc[d.pair_group.eq("outside_only")]

    # Exact expected best on each pair's own measured menu (Q pairs: qualified cells only).
    eb = {}
    for label, cells in (("Q", q_cells), ("O", o_cells)):
        for (cls, pair), g in cells.groupby(KEYS, sort=True):
            if len(g) >= BUDGET:
                eb[(label, cls, pair)] = expected_best(g["yield"].to_numpy())
    qeb = pd.DataFrame([(c, p, x) for (l, c, p), x in eb.items() if l == "Q"], columns=KEYS + ["expected_best"])
    quant = qeb.groupby("rxn_class").expected_best.quantile([.75, .90]).unstack()
    quant.columns = ["q75", "q90"]

    outside = pair_table(o_cells)
    anchors = pair_table(q_cells)
    outside["expected_best_12"] = [eb.get(("O", c, p), np.nan) for c, p in zip(outside.rxn_class, outside.subpair)]
    outside = outside.merge(quant, on="rxn_class", how="left", validate="many_to_one")
    outside["q_reference_available"] = outside.q75.notna() & outside.q90.notna()
    anchored = outside.expected_best_12.notna() & outside.q_reference_available
    for tag in ("q75", "q90"):
        outside[f"above_{tag}"] = anchored & outside.expected_best_12.ge(outside[tag])

    # Rule visibility: any record of the pair in any of the 360 construction masks.
    z = np.load(out_dir(v) / "rule_masks.npz")
    n = int(z["n_records"])
    masks = np.unpackbits(z["design_masks"], axis=1, bitorder="big")[:, :n].astype(bool)
    designs = pd.read_csv(out_dir(v) / "rule_designs.csv")
    assert len(designs) == 360 and designs.mask_index.nunique() == len(masks)
    union = masks[designs.mask_index.unique()].any(axis=0)
    assert np.array_equal(union, np.unpackbits(z["rule_union"], bitorder="big")[:n].astype(bool))
    visible = u.loc[union, KEYS].drop_duplicates()
    outside["rule_union_visible"] = pd.MultiIndex.from_frame(outside[KEYS]).isin(pd.MultiIndex.from_frame(visible))

    # Nearest same-class qualified pair (mean of the two role Tanimotos) and shared measured conditions.
    cache = {}
    for table in (outside, anchors):
        for role in (1, 2):
            table[f"fp{role}"] = [fingerprint(x, cache) for x in table[f"sub_{role}_smiles"]]
        table["valid_fp"] = table.fp1.notna() & table.fp2.notna()
    outside = outside.merge(nearest_reference(outside, anchors), on=KEYS, validate="one_to_one")
    q_sets = q_cells.groupby(KEYS).condition_id.agg(frozenset).to_dict()
    o_sets = o_cells.groupby(KEYS).condition_id.agg(frozenset).to_dict()
    shared = []
    for r in outside.itertuples(index=False):
        anchor = (r.rxn_class, r.nearest_q_subpair)
        ok = r.q_reference_available and r.nearest_q_subpair != "" and anchor in q_sets
        shared.append(len(o_sets[(r.rxn_class, r.subpair)] & q_sets[anchor]) if ok else 0)
    outside["n_shared_conditions"] = shared

    absent = ~outside.rule_union_visible
    recomputed = {"q75_outside_tail": int(outside.above_q75.sum()),
                  "q75_tail_absent_from_rule_union": int((outside.above_q75 & absent).sum()),
                  "q90_outside_tail": int(outside.above_q90.sum()),
                  "q90_tail_absent_from_rule_union": int((outside.above_q90 & absent).sum()),
                  "outside_absent_from_rule_union": int(absent.sum()),
                  "absent_sharing_ge12_with_nearest_qualified": int((absent & outside.n_shared_conditions.ge(12)).sum())}
    assert len(outside) > 0 and len(anchors) > 0
    cols = KEYS + ["n_conditions", "expected_best_12", "q75", "q90", "above_q75", "above_q90",
                   "rule_union_visible", "nearest_q_subpair", "nearest_q_similarity", "n_shared_conditions"]
    out = out_dir(v)
    outside[cols].to_csv(out / "opportunity_outside_pairs.csv", index=False)
    check = [{"name": k, "manuscript": MANUSCRIPT[k], "recomputed": recomputed[k],
              "match": MANUSCRIPT[k] == recomputed[k]} for k in MANUSCRIPT]
    (out / "check_opportunity.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    print(json.dumps(dict(variant=v, outside_pairs=len(outside), qualified_pairs=len(anchors),
                          B12_anchored=int(anchored.sum()), seconds=round(time.time() - t0, 1),
                          check=check), indent=1))


if __name__ == "__main__":
    main()
