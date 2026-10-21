"""Rule union: replay the two screen-construction rule families (numeric and attempted
condition menus) over 5 x 3 x 3 x 2 x 2 = 180 settings each, and write record masks.

Condition-menu blocks use Jaccard similarity 0.90. Every construction is
recomputed from the chosen endpoint domain.
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from m2common import data_dir, out_dir

RDLogger.DisableLog("rdApp.*")
TWINS = {"Hiyama": [("C43", "C46")], "Suzuki": [("C74", "C77")],
         "Finkelstein": [("C34", "C37")]}
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
KEYS = ["rxn_class", "subpair"]
BASELINE = "s50_c12_f70_o1_q1"


def menu_blocks(sub, jac=0.90):
    sig = sub.groupby("subpair")["condition_id"].apply(lambda s: frozenset(s.unique()))
    cnt = Counter(sig.values)
    sigs = [s for s, _ in cnt.most_common()]
    used, res = set(), []
    for i, a in enumerate(sigs):
        if i in used:
            continue
        grp = [i]; used.add(i)
        for j in range(i + 1, len(sigs)):
            if j in used:
                continue
            if len(a & sigs[j]) / max(len(a | sigs[j]), 1) >= jac:
                grp.append(j); used.add(j)
        keep = {sigs[k] for k in grp}
        res.append(([sp for sp, s in sig.items() if s in keep], sorted(set().union(*keep))))
    res.sort(key=lambda x: -len(x[0]))
    return res


def first_valid(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None and GEN.GetFingerprint(mol).GetNumOnBits() > 0


def block_cache(d, valid_lookup):
    blocks = []
    for (cls, bid), a in d.groupby(["rxn_class", "block_id"], sort=True):
        p = a.pivot(index="subpair", columns="condition_id", values="yield")
        rc = p.index.get_indexer(a.subpair)
        cc = p.columns.get_indexer(a.condition_id)
        parse = np.array([valid_lookup.get(s.split(" >> ")[0], False) for s in p.index])
        rowfill = p.notna().mean(axis=1).to_numpy()
        colfill = p.notna().mean(axis=0).to_numpy()
        cores = {}
        for fill in (.5, .7, .9):
            rr, co = rowfill >= fill, colfill >= fill
            cores[fill] = dict(rows=rr, cols=co, valid_rows=rr & parse,
                               nrows=int(rr.sum()), ncols=int(co.sum()),
                               nvalid=int((rr & parse).sum()))
        blocks.append(dict(cls=cls, bid=bid, index=a.index.to_numpy(),
                           rc=rc, cc=cc, subs=np.array(p.index), conds=np.array(p.columns),
                           nrows=p.shape[0], ncols=p.shape[1], cores=cores))
    return blocks


def replay(blocks, n, min_sub, min_cond, fill, suppress=True, qc=True, stages=False):
    candidates, stage_masks = {}, {}
    if stages:
        stage_masks = {k: np.zeros(n, dtype=bool) for k in
                       ("03_block_size", "04_dense_core", "05_fingerprint", "06_qc_twins", "07_overlap_suppression")}
    for b in blocks:
        if b["nrows"] < min_sub or b["ncols"] < min_cond:
            continue
        if stages:
            stage_masks["03_block_size"][b["index"]] = True
        core = b["cores"][fill]
        if core["nrows"] < min_sub or core["ncols"] < min_cond:
            continue
        dense_rows = core["rows"][b["rc"]] & core["cols"][b["cc"]]
        if stages:
            stage_masks["04_dense_core"][b["index"][dense_rows]] = True
        if core["nvalid"] < min_sub:
            continue
        fp_rows = core["valid_rows"][b["rc"]] & core["cols"][b["cc"]]
        if stages:
            stage_masks["05_fingerprint"][b["index"][fp_rows]] = True
        condkeep = core["cols"].copy()
        if qc:
            for first, second in TWINS.get(b["cls"], []):
                if first in b["conds"][condkeep] and second in b["conds"][condkeep]:
                    condkeep[b["conds"] == second] = False
        indices = b["index"][core["valid_rows"][b["rc"]] & condkeep[b["cc"]]]
        if stages:
            stage_masks["06_qc_twins"][indices] = True
        candidates[b["bid"]] = dict(cls=b["cls"], subs=b["subs"][core["valid_rows"]],
                                     conds=b["conds"][condkeep], indices=indices)
    dropped = set()
    if suppress:
        keys = sorted(candidates)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                sa, sb = candidates[a], candidates[b]
                if sa["cls"] != sb["cls"] or a in dropped or b in dropped:
                    continue
                if set(sa["conds"]) & set(sb["conds"]):
                    dropped.add(a if len(sa["subs"]) < len(sb["subs"]) else b)
    kept = {key: value for key, value in candidates.items() if key not in dropped}
    mask = np.zeros(n, dtype=bool)
    for s in kept.values():
        mask[s["indices"]] = True
    if stages:
        stage_masks["07_overlap_suppression"] = mask.copy()
    return mask, kept, dropped, stage_masks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="clfzero")
    v = ap.parse_args().variant
    raw = pd.read_parquet(data_dir(v) / "observations.parquet")
    eligible = raw.numeric & ~raw.duplicate_cell & raw.rxn_class.ne("Finkelstein")
    data = raw.loc[eligible].copy()
    lookup = {s: first_valid(s) for s in data.sub_1_smiles.dropna().unique()}
    numeric = block_cache(data, lookup)
    attempts = raw.loc[raw.rxn_class.ne("Finkelstein") & ~raw.duplicate_cell]
    labels = {}
    for cls, group in attempts.groupby("rxn_class", sort=True):
        for number, (subs, _) in enumerate(menu_blocks(group), 1):
            labels.update({(cls, pair): f"{cls}|{number}" for pair in subs})
    data["block_id"] = [labels[k] for k in data[KEYS].itertuples(index=False, name=None)]
    attempt = block_cache(data, lookup)
    masks, ids, designs, union = [], {}, [], {}
    for family, blocks in [("numeric", numeric), ("attempt", attempt)]:
        fam = np.zeros(len(raw), dtype=bool)
        for ms, mc, fill, sup, qc in product((20, 30, 50, 75, 100), (6, 12, 24), (.5, .7, .9), (False, True), (False, True)):
            sid = f"s{ms}_c{mc}_f{int(fill*100)}_o{int(sup)}_q{int(qc)}"
            mask, kept, _, _ = replay(blocks, len(raw), ms, mc, fill, sup, qc)
            h = sha256(np.packbits(mask, bitorder="big").tobytes()).hexdigest()
            if h not in ids:
                ids[h] = len(masks); masks.append(mask.copy())
            if family == "numeric" and sid == BASELINE:
                assert np.array_equal(mask, raw.qualified_row.to_numpy())
            fam |= mask
            designs.append(dict(design_id=f"{family}|{sid}", signature_mode=family, scenario_id=sid,
                                mask_index=ids[h], n_records=int(mask.sum()), n_screens=len(kept)))
        union[family] = fam
    two = union["numeric"] | union["attempt"]
    valid = raw.analysis_valid.to_numpy()
    res = dict(variant=v, constructions=len(designs), distinct_masks=len(masks),
               rule_union_records=int(two.sum()), rule_union_valid_records=int((two & valid).sum()),
               numeric_family_records=int(union["numeric"].sum()), attempt_family_records=int(union["attempt"].sum()))
    out = out_dir(v)
    pd.DataFrame(designs).to_csv(out / "rule_designs.csv", index=False)
    np.savez_compressed(out / "rule_masks.npz", design_masks=np.packbits(np.vstack(masks), axis=1, bitorder="big"),
                        rule_union=np.packbits(two, bitorder="big"), n_records=np.array(len(raw)),
                        record_id=raw.record_id.to_numpy(dtype=str))
    (out / "rule_union_counts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
