"""Summarize existing counterexample margins and paired measured-yield differences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
THRESHOLDS_PP = (5, 10, 15, 20)
REPLICATES = (('Hiyama', 'C43', 'C46'), ('Suzuki', 'C74', 'C77'))
RECIPE_FIELDS = tuple(f'reagent{i}_id' for i in range(1, 7)) + ('solvent_id', 'temp_id', 'time_id')
TOLERANCE = 1e-12


def _require(frame: pd.DataFrame, columns, label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{label} missing columns: {sorted(missing)}')


def summarize_thresholds(strength: pd.DataFrame, edges: pd.DataFrame, *,
                         expected_targets: int = 177) -> tuple[list[dict], pd.DataFrame]:
    _require(edges, ('edge_id', 'rxn_class', 'new_contradiction'), 'counterexample targets')
    _require(strength, ('edge_id', 'rxn_class', 'universe', 'recordset_id', 'rho'), 'strengths')
    flags = edges.new_contradiction
    if flags.isna().any() or not flags.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError('Counterexample target flags must be nonmissing booleans')
    targets = edges.loc[flags, ['edge_id', 'rxn_class']].copy()
    if len(targets) != expected_targets or not targets.edge_id.is_unique:
        raise ValueError(f'Expected {expected_targets} unique counterexample targets, found {len(targets)}')
    full = strength.loc[strength.universe.eq('frozen') & strength.recordset_id.eq('full')]
    if not full.edge_id.is_unique:
        raise ValueError('Duplicate frozen/full strength edge')
    merged = targets.merge(full[['edge_id', 'rxn_class', 'rho']], on='edge_id', how='left',
                           validate='one_to_one', suffixes=('', '_strength'))
    if (merged.rxn_class.ne(merged.rxn_class_strength).any() or
            not np.isfinite(merged.rho.to_numpy(float)).all()):
        raise ValueError('Missing, nonfinite, or identity-mismatched target strength')
    if (merged.rho.to_numpy(float) < 0.05 - TOLERANCE).any():
        raise ValueError('A frozen target does not survive its original five-point threshold')
    merged = merged.drop(columns='rxn_class_strength').sort_values('edge_id', kind='stable').reset_index(drop=True)
    merged['rho_pp'] = 100 * merged.rho
    summary = []
    for threshold in THRESHOLDS_PP:
        surviving = merged.rho.to_numpy(float) >= threshold / 100 - TOLERANCE
        merged[f'survives_{threshold}_pp'] = surviving
        summary.append({'threshold_pp': threshold, 'target_comparisons': len(merged),
                        'surviving_comparisons': int(surviving.sum()),
                        'surviving_fraction': float(surviving.mean())})
    return summary, merged


def _normal(value):
    return None if pd.isna(value) else str(value).strip()


def _signature(frame: pd.DataFrame, condition: str) -> tuple:
    records = frame.loc[frame.condition_id.eq(condition)]
    if records.empty:
        raise ValueError(f'Missing replicate condition: {condition}')
    identities = {tuple(_normal(value) for value in row)
                  for row in records.loc[:, RECIPE_FIELDS].itertuples(index=False, name=None)}
    if len(identities) != 1:
        raise ValueError(f'Replicate condition has inconsistent recipe identities: {condition}')
    return next(iter(identities))


def _statistics(first: np.ndarray, second: np.ndarray) -> dict:
    difference = 100 * (first - second)
    if len(difference) < 2 or not np.isfinite(difference).all():
        raise ValueError('At least two finite paired measured yields are required')
    correlation = (float(np.corrcoef(first, second)[0, 1])
                   if np.ptp(first) > 0 and np.ptp(second) > 0 else None)
    return {'difference_mean_pp': float(difference.mean()),
            'difference_sample_sd_pp': float(difference.std(ddof=1)),
            'absolute_difference_ge5_count': int((np.abs(difference) >= 5 - 100 * TOLERANCE).sum()),
            'absolute_difference_ge10_count': int((np.abs(difference) >= 10 - 100 * TOLERANCE).sum()),
            'absolute_difference_ge5_fraction': float((np.abs(difference) >= 5 - 100 * TOLERANCE).mean()),
            'absolute_difference_ge10_fraction': float((np.abs(difference) >= 10 - 100 * TOLERANCE).mean()),
            'pearson_r': correlation}


def replicate_statistics(observations: pd.DataFrame, replicates=REPLICATES) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    columns = ('rxn_class', 'condition_id', 'subpair', 'sub_1_smiles', 'sub_2_smiles',
               'product_smiles', 'record_id', 'source_file', 'source_row', 'raw_yield',
               'yield', 'measurement_status', 'source_campaign', 'plate_id', 'qc_pass', *RECIPE_FIELDS)
    _require(observations, columns, 'replicate observations')
    summaries, pair_rows, identities = [], [], []
    for reaction, first, second in replicates:
        frame = observations.loc[observations.rxn_class.eq(reaction) &
                                 observations.condition_id.isin((first, second))].copy()
        if frame.duplicated(['subpair', 'condition_id']).any() or not frame.record_id.is_unique:
            raise ValueError(f'Duplicate replicate cell or record identity: {reaction}')
        first_signature, second_signature = _signature(frame, first), _signature(frame, second)
        if first_signature != second_signature:
            raise ValueError(f'Replicate recipe identities differ: {reaction} {first}/{second}')
        identities.append({'rxn_class': reaction, 'condition_A': first, 'condition_B': second,
                           'identical_recipe_signature': True,
                           **dict(zip(RECIPE_FIELDS, first_signature))})
        numeric_raw = pd.to_numeric(frame.raw_yield, errors='coerce')
        measured = frame.measurement_status.eq('quantified')
        if (not np.isfinite(numeric_raw.loc[measured].to_numpy(float)).all() or
                not np.isfinite(frame.loc[measured, 'yield'].to_numpy(float)).all() or
                not np.array_equal(numeric_raw.loc[measured].to_numpy(float),
                                   frame.loc[measured, 'yield'].to_numpy(float))):
            raise ValueError(f'A quantified replicate must have its original finite numeric yield: {reaction}')
        by_condition = {condition: frame.loc[frame.condition_id.eq(condition)].set_index('subpair')
                        for condition in (first, second)}
        shared = sorted(set(by_condition[first].index) & set(by_condition[second].index))
        left, right = (by_condition[condition].loc[shared] for condition in (first, second))
        for role in ('sub_1_smiles', 'sub_2_smiles', 'product_smiles'):
            if left[role].map(_normal).ne(right[role].map(_normal)).any():
                raise ValueError(f'Paired replicate molecular identity differs: {reaction}, {role}')
        for condition, records in by_condition.items():
            expected_pairs = records.sub_1_smiles.astype(str) + ' >> ' + records.sub_2_smiles.astype(str)
            if not np.array_equal(expected_pairs.to_numpy(), records.index.to_numpy()):
                raise ValueError(f'Replicate substrate-pair key differs from its identities: {reaction} {condition}')
        complete_measured = left.measurement_status.eq('quantified') & right.measurement_status.eq('quantified')
        for field in ('source_campaign', 'plate_id'):
            if (left.loc[complete_measured, field].map(_normal).ne(
                    right.loc[complete_measured, field].map(_normal)).any() or
                    left.loc[complete_measured, field].isna().any()):
                raise ValueError(f'Paired measured replicates must share {field}: {reaction}')
        numeric_both = np.isfinite(left['yield'].to_numpy(float)) & np.isfinite(right['yield'].to_numpy(float))
        rule_zero = left.measurement_status.eq('zero_by_rule') | right.measurement_status.eq('zero_by_rule')
        measured_left, measured_right = left.loc[complete_measured], right.loc[complete_measured]
        for records in (measured_left, measured_right):
            if not records.qc_pass.map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value)).all():
                raise ValueError(f'Paired measured replicates must pass recorded QC: {reaction}')
            if not records['yield'].between(0, 1).all():
                raise ValueError(f'Paired measured yields must be within the unit interval: {reaction}')
        stats = _statistics(measured_left['yield'].to_numpy(float), measured_right['yield'].to_numpy(float))
        row = {'rxn_class': reaction, 'condition_A': first, 'condition_B': second,
               'condition_A_recorded_pairs': len(by_condition[first]),
               'condition_B_recorded_pairs': len(by_condition[second]),
               'shared_recorded_pairs': len(shared),
               'finite_endpoint_pairs_including_rule_zeros': int(numeric_both.sum()),
               'rule_zero_pairs_excluded': int((numeric_both & rule_zero.to_numpy()).sum()),
               'measured_pairs': int(complete_measured.sum()),
               'paired_qc_pass_all': True, 'paired_yields_in_unit_interval': True,
               'paired_same_campaign_all': True, 'paired_same_plate_all': True, **stats}
        inclusive = _statistics(left.loc[numeric_both, 'yield'].to_numpy(float),
                                right.loc[numeric_both, 'yield'].to_numpy(float))
        row.update({f'including_rule_zeros_{key}': value for key, value in inclusive.items()})
        summaries.append(row)
        for pair in measured_left.index:
            a, b = measured_left.loc[pair], measured_right.loc[pair]
            result = {'rxn_class': reaction, 'condition_A': first, 'condition_B': second,
                      'subpair': pair, 'sub_1_smiles': a.sub_1_smiles,
                      'sub_2_smiles': a.sub_2_smiles, 'product_smiles': a.product_smiles}
            for suffix, record in (('A', a), ('B', b)):
                for field in ('record_id', 'source_file', 'source_row', 'source_campaign',
                              'plate_id', 'measurement_status', 'qc_pass', 'raw_yield', 'yield'):
                    result[f'{field}_{suffix}'] = record[field]
            result['difference_pp'] = 100 * (float(a['yield']) - float(b['yield']))
            result['absolute_difference_pp'] = abs(result['difference_pp'])
            pair_rows.append(result)
    return summaries, pd.DataFrame(pair_rows), identities


def headline_fields(summary: dict) -> dict:
    fields = {'noise_counterexamples': summary['target_comparisons'],
              'noise_rho_median_pp': f"{summary['target_rho_median_pp']:.1f}"}
    fields.update({f"noise_survive_{row['threshold_pp']}": row['surviving_comparisons']
                   for row in summary['thresholds']})
    for row in summary['replicates']:
        prefix = f"noise_{row['rxn_class'].lower()}"
        fields.update({f'{prefix}_pairs': row['measured_pairs'],
                       f'{prefix}_analysis_pairs': row['finite_endpoint_pairs_including_rule_zeros'],
                       f'{prefix}_excluded_rule_zero_pairs': row['rule_zero_pairs_excluded'],
                       f'{prefix}_sd_pp': f"{row['difference_sample_sd_pp']:.1f}",
                       f'{prefix}_r': f"{row['pearson_r']:.2f}" if row['pearson_r'] is not None else 'undefined'})
        for threshold in (5, 10):
            fields[f'{prefix}_ge{threshold}'] = row[f'absolute_difference_ge{threshold}_count']
            fields[f'{prefix}_ge{threshold}_pct'] = f"{100 * row[f'absolute_difference_ge{threshold}_fraction']:.1f}"
    return fields


def table_markdown(summary: dict) -> str:
    lines = ['(a) Survival of the original counterexample comparisons.', '',
             '| Minimum yield difference (percentage points) | Surviving recipe comparisons |',
             '|---:|---:|']
    lines.extend(f"| {row['threshold_pp']} | {row['surviving_comparisons']} |" for row in summary['thresholds'])
    lines.extend(['', '(b) Paired measured yields under duplicated conditions.', '',
                  '| Array | Condition pair | Pairs | SD (points) | Absolute difference >= 5 points | Absolute difference >= 10 points | r |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    for row in summary['replicates']:
        correlation = f"{row['pearson_r']:.2f}" if row['pearson_r'] is not None else 'undefined'
        lines.append(f"| {row['rxn_class']} | {row['condition_A']}/{row['condition_B']} | {row['measured_pairs']} | "
                     f"{row['difference_sample_sd_pp']:.1f} | {row['absolute_difference_ge5_count']} "
                     f"({100 * row['absolute_difference_ge5_fraction']:.1f}%) | {row['absolute_difference_ge10_count']} "
                     f"({100 * row['absolute_difference_ge10_fraction']:.1f}%) | {correlation} |")
    return '\n'.join(lines)


def verify_summary(summary: dict) -> None:
    expected = {'noise_counterexamples': 177, 'noise_rho_median_pp': '7.7',
                'noise_survive_5': 177, 'noise_survive_10': 58,
                'noise_survive_15': 17, 'noise_survive_20': 9,
                'noise_hiyama_pairs': 121, 'noise_hiyama_analysis_pairs': 122,
                'noise_hiyama_excluded_rule_zero_pairs': 1,
                'noise_hiyama_ge5': 5, 'noise_hiyama_ge10': 1,
                'noise_hiyama_sd_pp': '2.1', 'noise_hiyama_r': '0.93',
                'noise_suzuki_pairs': 10, 'noise_suzuki_ge5': 6, 'noise_suzuki_ge10': 2,
                'noise_suzuki_sd_pp': '8.5', 'noise_suzuki_r': '0.89'}
    observed = headline_fields(summary)
    differences = {name: {'expected': value, 'observed': observed.get(name)}
                   for name, value in expected.items() if observed.get(name) != value}
    if differences:
        raise ValueError('Computed noise-margin checks failed: ' + json.dumps(differences, sort_keys=True))


def build(out: Path, destination: Path | None = None) -> dict:
    out = Path(out)
    destination = Path(destination) if destination is not None else out / 'noise_margin'
    m2 = out / 'm2'
    manifest_path = m2 / 'processed/reconstruction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for reaction, first, second in REPLICATES:
        if [first, second] not in manifest['qc_twins'].get(reaction, []):
            raise ValueError(f'Replicate definition differs from ingestion manifest: {reaction}')
    edges = pd.read_csv(m2 / 'partner_score_edges.csv', dtype={'new_contradiction': 'boolean'})
    strength = pd.read_csv(m2 / 'constraint_strength_edges.csv', float_precision='round_trip')
    thresholds, targets = summarize_thresholds(strength, edges)
    observations = pd.read_parquet(m2 / 'processed/observations.parquet')
    replicates, pairs, identities = replicate_statistics(observations)
    inputs = ('m2/partner_score_edges.csv', 'm2/constraint_strength_edges.csv',
              'm2/processed/observations.parquet', 'm2/processed/reconstruction_manifest.json')
    summary = {'target_comparisons': len(targets), 'target_rho_median_pp': float(targets.rho_pp.median()),
               'thresholds': thresholds, 'replicates': replicates,
               'definitions': {
                   'targets': 'Original new_contradiction recipe comparisons only',
                   'strength_scope': 'frozen/full',
                   'rho': 'Maximum over all directed cycles of the minimum absolute yield difference on a cycle',
                   'thresholds': 'Inclusive >=, in percentage points; original yield fractions retained',
                   'comparison_tolerance_yield_fraction': TOLERANCE,
                   'paired_measurements': 'Both records have quantified status, finite original numeric yields in [0,1], and passing recorded QC; partners share a campaign and plate',
                   'difference': '100 * (yield_condition_A - yield_condition_B)',
                   'standard_deviation': 'Sample SD of signed paired differences, ddof=1',
                   'correlation': 'Pearson correlation of the paired numerical yields',
                   'rule_zero_sensitivity': 'Separately reported finite-pair statistics include DESI-negative blanks assigned zero',
                   'replicate_scope': 'Two named duplicated-condition groups, reported separately without pooling'},
               'source_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in inputs}}
    verify_summary(summary)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(thresholds).to_csv(destination / 'threshold_survival.csv', index=False)
    pd.DataFrame(replicates).to_csv(destination / 'replicate_noise.csv', index=False)
    pairs.to_csv(destination / 'replicate_pairs.csv', index=False)
    targets.to_csv(destination / 'target_strengths.csv', index=False)
    pd.DataFrame(identities).to_csv(destination / 'replicate_condition_identities.csv', index=False)
    (destination / 'noise_margin_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE / 'out')
    parser.add_argument('--destination', type=Path)
    arguments = parser.parse_args()
    result = build(arguments.out, arguments.destination)
    print(json.dumps(headline_fields(result), indent=2, allow_nan=False))
