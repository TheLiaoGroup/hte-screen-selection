"""Classify the local raw tables and summarize both analytical tiers.

Run from any directory: python run_triage.py [--raw DIR] [--out DIR].
All yields are fractions. Numerical zero and blank DESI-negative endpoints are
available outcomes; blank DESI-positive and blank unclassified outcomes remain
missing. The pair-maximum analysis uses numerical yields, preserves every tie,
and reports the first-row calculation separately for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanup import apply_cleanup, assert_clean
from release_inputs import read_release_table


RDLogger.DisableLog("rdApp.*")
BASE = Path(__file__).resolve().parents[2]
STATES = ("numeric", "blank_desi_negative", "blank_desi_positive", "blank_no_flag")
MEASUREMENT_STATUS = dict(zip(STATES, ("quantified", "zero_by_rule",
                                      "detected_not_quantified", "no_screen_record")))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def write_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def classify(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for raw, numeric in (("yield", "y"), ("yield_clf", "clf")):
        values = frame[raw].replace(r"^\s*$", pd.NA, regex=True)
        parsed = pd.to_numeric(values, errors="coerce")
        if (values.notna() & parsed.isna()).any():
            raise ValueError(f"Non-numerical, nonblank {raw} value")
        frame[numeric] = parsed.astype(float)
    if (~frame.clf.dropna().isin([0, 1])).any():
        raise ValueError("DESI calls must be 0, 1, or blank")
    if (~np.isfinite(frame.y.dropna())).any() or (frame.y.dropna() < 0).any():
        raise ValueError("Yield must be a finite, nonnegative fraction or blank")
    if frame.y.max() > 1.05:
        raise ValueError("Yield exceeds the expected fractional scale (maximum >1.05)")
    frame["cls"] = np.select(
        [frame.y.notna(), frame.clf.eq(0), frame.clf.eq(1)],
        STATES[:3], default=STATES[3])
    frame["yy"] = frame.y.mask(frame.cls.eq("blank_desi_negative"), 0.0)
    frame["measurement_status"] = frame.cls.map(MEASUREMENT_STATUS)
    assert frame.loc[frame.cls.isin(STATES[2:]), "yy"].isna().all()
    return frame


def plate_ids(names: pd.Series) -> pd.Series:
    return names.astype("string").str.extract(r"^(.*?-PL\d+)(?:-|$)", expand=False)


def load_raw(raw: Path, *, exclusions: pd.DataFrame | None = None) -> pd.DataFrame:
    files = sorted(path for path in raw.iterdir()
                   if path.suffix.lower() in {".xlsx", ".csv"}
                   and not path.name.startswith(("0_", "~$")))
    if not files:
        raise ValueError("No reaction tables matched the raw directory")
    frames = []
    required = {"yield", "yield_clf", "sub_1_smiles", "name0"}
    for path in files:
        release_frame = read_release_table(path)
        frame = (release_frame if release_frame is not None else
                 pd.read_csv(path, dtype="string", low_memory=False)
                 if path.suffix.lower() == ".csv"
                 else pd.read_excel(path, dtype="string"))
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"Empty or incompatible reaction table: {path.name}")
        if release_frame is None:
            frame["source_file"] = path.name
            frame["source_row"] = np.arange(2, len(frame) + 2, dtype=np.int64)
        reaction_source = str(frame.source_file.iloc[0]) if release_frame is not None else path.name
        frame["rxn"] = re.split(r"_\d|\.xlsx|_\.|_$", reaction_source)[0]
        frame["array_id"] = frame.rxn
        if "sub_2_smiles" not in frame:
            if not frame.rxn.eq("Finkelstein").all():
                raise ValueError(f"Missing second substrate in coupling table: {path.name}")
            frame["sub_2_smiles"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frames.append(frame)
        print(f"Read {path.name}: {len(frame):,} rows", flush=True)
    cleaned, cleanup = apply_cleanup(pd.concat(frames, ignore_index=True, sort=False),
                                     reaction_column="rxn", exclusions=exclusions)
    result = classify(cleaned)
    cache = {}

    def canonical(value):
        if pd.isna(value):
            return None
        if value not in cache:
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                raise ValueError(f"Invalid substrate SMILES: {value}")
            cache[value] = Chem.MolToSmiles(molecule)
        return cache[value]

    for column in ("sub_1_smiles", "sub_2_smiles"):
        result[column + "_c"] = result[column].map(canonical).astype("string")
    coupling = result.rxn.ne("Finkelstein")
    if result.loc[coupling, ["sub_1_smiles_c", "sub_2_smiles_c"]].isna().any().any():
        raise ValueError("Pair identity requires both substrate SMILES")
    result["pair"] = (result.rxn + "|" + result.sub_1_smiles_c + "|"
                      + result.sub_2_smiles_c)
    result["plate_id"] = plate_ids(result.name0)
    result["include_primary"] = result.rxn.ne("Finkelstein")
    result.attrs["cleanup"] = cleanup
    return result


def inventory(frame: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(frame.rxn, frame.cls).reindex(columns=STATES, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["available_endpoints"] = counts.numeric + counts.blank_desi_negative
    return counts.rename_axis("rxn").reset_index()


def pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("pair", sort=True, observed=True)
    result = grouped.agg(rxn=("rxn", "first"), wells=("y", "size"),
                         numeric_wells=("y", "count"), endpoint_wells=("yy", "count"),
                         best=("y", "max"), endpoint_best=("yy", "max"))
    numeric = frame.loc[frame.y.notna()].copy()
    numeric["pair_best"] = numeric.groupby("pair", observed=True).y.transform("max")
    maxima = numeric.loc[numeric.y.eq(numeric.pair_best)].copy()
    maxima["positive"] = maxima.clf.eq(1).astype(int)
    maxima["negative"] = maxima.clf.eq(0).astype(int)
    maxima["unknown"] = maxima.clf.isna().astype(int)
    ties = maxima.groupby("pair", observed=True).agg(
        best_tie_wells=("y", "size"), best_tie_positive=("positive", "sum"),
        best_tie_negative=("negative", "sum"), best_tie_unknown=("unknown", "sum"))
    # head(1) retains the first row even when its call is missing.
    first = maxima.groupby("pair", sort=False, observed=True).head(1).set_index("pair")
    first = first[["clf", "source_file", "source_row", "name0"]].rename(columns={
        "clf": "first_best_clf", "source_file": "first_best_source_file",
        "source_row": "first_best_source_row", "name0": "first_best_name0"})
    result = result.join(ties).join(first)
    for column in ties.columns:
        result[column] = result[column].fillna(0).astype(int)
    result["all_best_calls_negative"] = (result.best_tie_wells.gt(0)
                                         & result.best_tie_negative.eq(result.best_tie_wells))
    result["any_best_call_positive"] = result.best_tie_positive.gt(0)
    return result.reset_index()


def threshold_summary(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well_column in ("wells", "numeric_wells"):
        eligible = best.loc[best[well_column].ge(20) & best.best.notna()]
        for operator, threshold in ((">", 0.0), (">", 0.05), (">", 0.20), (">=", 0.20)):
            selected = eligible.loc[eligible.best.gt(threshold) if operator == ">"
                                    else eligible.best.ge(threshold)]
            rows.append({
                "minimum_20_definition": well_column, "operator": operator,
                "threshold_fraction": threshold, "pairs": len(selected),
                "first_best_negative": int(selected.first_best_clf.eq(0).sum()),
                "first_best_positive": int(selected.first_best_clf.eq(1).sum()),
                "first_best_unknown": int(selected.first_best_clf.isna().sum()),
                "pairs_with_tied_maximum": int(selected.best_tie_wells.gt(1).sum()),
                "any_maximum_negative": int(selected.best_tie_negative.gt(0).sum()),
                "all_maxima_negative": int(selected.all_best_calls_negative.sum()),
                "any_maximum_positive": int(selected.any_best_call_positive.sum()),
                "any_maximum_unknown": int(selected.best_tie_unknown.gt(0).sum()),
                "no_known_positive_maximum": int((~selected.any_best_call_positive).sum()),
            })
    return pd.DataFrame(rows)


def negative_quantification_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    negative = frame.loc[frame.clf.eq(0) & frame.y.notna()].copy()
    positive = negative.loc[negative.y.gt(0)].copy()
    if negative.plate_id.isna().any():
        raise ValueError("DESI-negative numerical wells require a parseable plate identifier")
    for name, threshold in (("ge1pct", .01), ("ge5pct", .05), ("ge20pct", .20)):
        negative[name] = negative.y.ge(threshold)
    negative["positive_numeric"] = negative.y.gt(0)
    plates = negative.groupby(["rxn", "plate_id"], sort=True, observed=True).agg(
        negative_numeric_wells=("y", "size"),
        negative_positive_numeric_wells=("positive_numeric", "sum"),
        negative_ge1pct_wells=("ge1pct", "sum"),
        negative_ge5pct_wells=("ge5pct", "sum"),
        negative_ge20pct_wells=("ge20pct", "sum")).reset_index()
    counts = plates.loc[plates.negative_positive_numeric_wells.gt(0),
                        "negative_positive_numeric_wells"]
    above_three = counts.gt(3)
    denominator = len(positive)
    numerator = int(negative.y.ge(.05).sum())
    audit = {
        "negative_numeric_wells": len(negative),
        "negative_positive_numeric_wells": denominator,
        "negative_ge1pct_wells": int(negative.y.ge(.01).sum()),
        "negative_ge5pct_wells": numerator,
        "negative_gt5pct_wells": int(negative.y.gt(.05).sum()),
        "negative_ge20pct_wells": int(negative.y.ge(.20).sum()),
        "negative_gt20pct_wells": int(negative.y.gt(.20).sum()),
        "positive_numeric_plate_count": len(counts),
        "median_positive_numeric_wells_per_plate": float(counts.median()) if len(counts) else None,
        "max_positive_numeric_wells_per_plate": int(counts.max()) if len(counts) else 0,
        "plates_with_more_than_three_positive_numeric_wells": int(above_three.sum()),
        "positive_numeric_wells_on_plates_with_more_than_three": int(counts.loc[above_three].sum()),
        "ge5pct_rate_upper_bound_pct": 100 * numerator / denominator if denominator else None,
        "bound_formula": "100 * DESI-negative wells with yield >= 0.05 / DESI-negative wells with yield > 0",
        "bound_scope": "among quantitatively assessed DESI-negative wells in this analysis scope",
        "bound_assumption": "positive numerical yields identify quantitatively assessed wells and all such yields >= 0.05 are recorded",
        "bound_excludes": "all-well false-negative rate, assay sensitivity, and unassessed DESI-negative wells",
        "plate_identity": "reaction array and complete plate prefix from name0",
    }
    return audit, plates, positive


def write_report(scope: dict, path: Path) -> None:
    audit = scope["negative_quantification_audit"]
    strict = next(row for row in scope["pair_maximum_thresholds"]
                  if row["minimum_20_definition"] == "wells"
                  and row["operator"] == ">" and row["threshold_fraction"] == .20)
    inclusive = next(row for row in scope["pair_maximum_thresholds"]
                     if row["minimum_20_definition"] == "wells"
                     and row["operator"] == ">=" and row["threshold_fraction"] == .20)
    bound = audit["ge5pct_rate_upper_bound_pct"]
    bound_text = (f"{audit['negative_ge5pct_wells']:,}/{audit['negative_positive_numeric_wells']:,} "
                  f"(approximately {bound:.3f}%)" if bound is not None else "undefined (no positive numerical yields)")
    text = f"""# Triage and Quantification

This analysis contains {scope['arrays']:,} arrays and {scope['wells']:,} cleaned reaction records, excluding Finkelstein.

## Pair maxima

Among substrate pairs with at least 20 recorded wells, {strict['pairs']:,} have a maximum numerical yield strictly greater than 20%. None has a DESI-negative maximum. At the inclusive threshold (at least 20%), {inclusive['pairs']:,} pairs qualify and {inclusive['any_maximum_negative']:,} has a DESI-negative maximum. The Chan-Lam boundary maximum is exactly 20.0%, so it enters only the inclusive threshold. All tied numerical maxima are retained in these counts.

## DESI-negative wells

Of {audit['negative_numeric_wells']:,} DESI-negative wells with numerical yield entries, {audit['negative_positive_numeric_wells']:,} have yield greater than zero; {audit['negative_ge1pct_wells']:,}, {audit['negative_ge5pct_wells']:,}, and {audit['negative_ge20pct_wells']:,} have yield at least 1%, 5%, and 20%, respectively. Strictly greater than 5% and 20% gives {audit['negative_gt5pct_wells']:,} and {audit['negative_gt20pct_wells']:,} wells, respectively.

The positive numerical DESI-negative wells occur on {audit['positive_numeric_plate_count']:,} plates, with a median of {audit['median_positive_numeric_wells_per_plate']:g} per plate and a maximum of {audit['max_positive_numeric_wells_per_plate']:,}. Of these plates, {audit['plates_with_more_than_three_positive_numeric_wells']:,} contain more than three such wells, comprising {audit['positive_numeric_wells_on_plates_with_more_than_three']:,} wells. These data do not establish a fixed number of negative-well UPLC checks per plate; the authors must confirm the sampling protocol.

## SI-ready observation and bound

DESI-negative wells with positive numerical yields provide a minimum count of quantitatively assessed negatives. Assuming that these positive numerical yields identify quantitatively assessed wells and that all recorded yields of at least 5% are represented, the fraction with yield at least 5% among quantitatively assessed DESI-negative wells is at most {bound_text}. Quantitatively assessed wells with zero yield would increase the denominator and reduce this bound. Recorded numerical zeros cannot distinguish such measurements from zeros assigned by the screening rule. This bound is not an all-well false-negative rate or an estimate of sensitivity; it makes no claim about unassessed DESI-negative wells.

## Quantification of detected products

Of {scope['desi_pos']:,} DESI-positive wells, {scope['pos_unquantified']:,} lack numerical yields ({scope['pos_unquantified_pct']:.1f}%). These endpoints remain missing and are never imputed as zero. Half of the unquantifiable wells occur in {scope['unquantifiable_concentration']['0.5']['pairs']:,} of {scope['pairs']:,} substrate pairs ({scope['unquantifiable_concentration']['0.5']['pct_all_pairs']:.1f}%).
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def analyze(frame: pd.DataFrame, out: Path) -> dict:
    if frame.empty:
        raise ValueError("Analysis scope matched zero rows")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(inventory(frame), out / "raw_inventory_by_reaction.csv")
    rows = []
    for reaction, group in frame.groupby("rxn", sort=True, observed=True):
        positive, negative = group.clf.eq(1), group.clf.eq(0)
        has = group.y.notna()
        neg_numeric = group.loc[negative & has, "y"]
        rows.append({"rxn": reaction, "wells": len(group), "desi_pos": int(positive.sum()),
                     "desi_neg": int(negative.sum()), "desi_unknown": int(group.clf.isna().sum()),
                     "pos_unquantified": int((positive & ~has).sum()),
                     "pos_unquantified_pct": 100 * (positive & ~has).sum() / positive.sum()
                     if positive.any() else np.nan,
                     "neg_with_number": len(neg_numeric), "neg_gt0": int(neg_numeric.gt(0).sum()),
                     "neg_ge1pct": int(neg_numeric.ge(0.01).sum()),
                     "neg_ge5pct": int(neg_numeric.ge(0.05).sum()),
                     "neg_ge20pct": int(neg_numeric.ge(0.20).sum())})
    reactions = pd.DataFrame(rows)
    write_csv(reactions, out / "triage_by_reaction.csv")
    write_csv(reactions[["rxn", "desi_pos", "pos_unquantified", "pos_unquantified_pct"]],
              out / "unquantifiable_by_reaction.csv")
    best = pair_summary(frame)
    write_csv(best, out / "pair_best_well_desi_call.csv")
    thresholds = threshold_summary(best)
    write_csv(thresholds, out / "pair_maximum_thresholds.csv")
    unquantified = frame.loc[frame.clf.eq(1) & frame.y.isna()]
    concentration = (unquantified.groupby("pair", sort=True, observed=True).size()
                     .rename("unquantified_wells").reset_index()
                     .sort_values(["unquantified_wells", "pair"], ascending=[False, True],
                                  kind="stable"))
    concentration["cumulative_unquantified_wells"] = concentration.unquantified_wells.cumsum()
    write_csv(concentration, out / "unquantifiable_by_pair.csv")
    totals = reactions.select_dtypes(include="number").drop(columns="pos_unquantified_pct").sum()
    scope = {
        "arrays": int(frame.array_id.nunique()), "wells": len(frame),
        "pairs": int(best.shape[0]), "states": {state: int(frame.cls.eq(state).sum()) for state in STATES},
        "measurement_status": {status: int(frame.measurement_status.eq(status).sum())
                               for status in MEASUREMENT_STATUS.values()},
        **{key: int(value) for key, value in totals.items() if key != "wells"},
        "pos_unquantified_pct": 100 * len(unquantified) / int(totals.desi_pos),
        "pairs_ge20_total_wells_with_numeric_yield": int((best.wells.ge(20) & best.best.notna()).sum()),
        "pairs_ge20_total_wells_without_numeric_yield": int((best.wells.ge(20) & best.best.isna()).sum()),
        "pairs_ge20_numeric_wells": int(best.numeric_wells.ge(20).sum()),
        "pairs_with_unquantifiable_wells": len(concentration),
        "unquantifiable_concentration": {}, "pair_maximum_thresholds": thresholds.to_dict("records"),
    }
    if len(unquantified):
        for fraction in (0.5, 0.8, 0.9):
            k = int((concentration.cumulative_unquantified_wells < fraction * len(unquantified)).sum()) + 1
            scope["unquantifiable_concentration"][str(fraction)] = {
                "pairs": k, "pct_all_pairs": 100 * k / len(best)}
    boundary = best.loc[best.rxn.eq("Chan-Lam") & best.best.eq(0.20)
                        & best.best_tie_negative.gt(0) & best.wells.ge(20)]
    detail = frame.loc[frame.pair.isin(boundary.pair)].copy()
    detail["is_numeric_pair_maximum"] = detail.y.eq(0.20)
    write_csv(detail, out / "chan_lam_20pct_boundary_all_rows.csv")
    write_csv(boundary, out / "chan_lam_20pct_boundary_pairs.csv")
    scope["chan_lam_20pct_boundary"] = boundary.replace({np.nan: None}).to_dict("records")
    audit, plates, positive_negative = negative_quantification_audit(frame)
    scope["negative_quantification_audit"] = audit
    write_csv(plates, out / "negative_numeric_by_plate.csv")
    write_csv(positive_negative, out / "negative_positive_numeric_wells.csv")
    write_json(scope, out / "summary.json")
    write_report(scope, out / "REPORT.md")
    return scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=BASE / "data" / "raw")
    parser.add_argument("--out", type=Path, default=BASE / "out" / "triage")
    args = parser.parse_args()
    frame = load_raw(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    assert_clean(frame, reaction_column="rxn")
    cleanup = frame.attrs["cleanup"]
    frame.to_parquet(args.out / "raw_classified.parquet", index=False, compression="zstd")
    write_csv(inventory(frame), args.out / "all_table_inventory.csv")
    for name in ("excluded_plate_rows.csv", "excluded_plate_summary.csv"):
        (args.out / name).unlink(missing_ok=True)
    write_csv(pd.DataFrame(cleanup["by_plate"]), args.out / "cleanup_by_plate.csv")
    write_json(cleanup, args.out / "cleanup_summary.json")
    reference = frame.loc[frame.rxn.ne("Finkelstein")]
    primary = frame.loc[frame.include_primary]
    summary = {
        "definitions": {
            "numeric_yield": "y: a numerical raw yield, including numerical zero",
            "analysis_endpoint": "yy: y, or zero only for blank yield with DESI call 0",
            "pair": "array and ordered canonical substrate SMILES",
            "maximum": "maximum numerical yield; every exactly tied row is retained",
            "minimum_20": "separately reported for all recorded wells and numerical-yield wells",
            "reference": "all cleaned reaction tables except Finkelstein",
            "primary": "all cleaned reaction tables except Finkelstein",
        },
        "all_tables": {"arrays": int(frame.array_id.nunique()), "wells": len(frame),
                       "states": {state: int(frame.cls.eq(state).sum()) for state in STATES},
                       "measurement_status": {status: int(frame.measurement_status.eq(status).sum())
                                              for status in MEASUREMENT_STATUS.values()}},
        "cleanup": cleanup,
        "reference": analyze(reference, args.out / "reference"),
        "primary": analyze(primary, args.out / "primary"),
    }
    write_json(summary, args.out / "summary.json")
    (args.out / "REPORT.md").write_text(
        (args.out / "primary" / "REPORT.md").read_text(encoding="utf-8"),
        encoding="utf-8", newline="\n")
    print(json.dumps({key: {field: summary[key][field]
                            for field in ("arrays", "wells", "pairs", "pos_unquantified", "pos_unquantified_pct")}
                      for key in ("reference", "primary")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
