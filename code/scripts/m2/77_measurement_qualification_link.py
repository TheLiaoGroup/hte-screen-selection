"""Link analytical well status to recalculated pair qualification."""
import argparse
import json
import numpy as np
import pandas as pd
from m2common import SCOPE, assert_clean, data_dir, out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='clfzero')
    variant = parser.parse_args().variant
    u = pd.read_parquet(data_dir(variant) / 'observations.parquet')
    assert_clean(u, reaction_column='rxn_class')
    u = u.loc[u.rxn_class.ne('Finkelstein')].copy()
    missing = u.measurement_status.isin(['detected_not_quantified', 'no_screen_record'])
    assert len(u) > 0 and missing.any()
    assert u.loc[missing, 'yield'].isna().all() and not u.loc[missing, 'analysis_valid'].any()
    assert u.loc[u.measurement_status.eq('zero_by_rule'), 'yield'].eq(0).all()
    pair_in = u.groupby(['rxn_class', 'subpair']).qualified_row.any()
    idx = pd.MultiIndex.from_frame(u[['rxn_class', 'subpair']])
    u['pair_in_Q'] = pair_in.reindex(idx).to_numpy()
    u['detected_missing'] = u.measurement_status.eq('detected_not_quantified')
    u['zero_by_rule'] = u.measurement_status.eq('zero_by_rule')
    pairs = u.groupby(['rxn_class', 'subpair'], sort=True).agg(
        wells=('record_id', 'size'), included_in_Q=('pair_in_Q', 'first'),
        detected_not_quantified=('detected_missing', 'sum'), zero_by_rule=('zero_by_rule', 'sum'),
        observed=('numeric', 'sum'))
    pairs['unquantifiable_fraction'] = pairs.detected_not_quantified / pairs.wells
    rows = []
    for included, group in u.groupby('pair_in_Q', sort=True):
        p = pairs.loc[pairs.included_in_Q.eq(included)]
        rows.append(dict(included_in_Q=bool(included), pairs=len(p), wells=len(group),
                         detected_not_quantified=int(group.detected_missing.sum()),
                         detected_not_quantified_fraction=float(group.detected_missing.mean()),
                         zero_by_rule=int(group.zero_by_rule.sum())))
    grouped = pd.DataFrame(rows)
    total_missing = int(u.detected_missing.sum())
    assert total_missing > 0 and set(grouped.included_in_Q) == {False, True}
    excluded_missing = int(grouped.loc[~grouped.included_in_Q, 'detected_not_quantified'].iloc[0])
    positive = pairs.loc[pairs.detected_not_quantified.gt(0)].sort_values(
        'detected_not_quantified', ascending=False, kind='stable')
    cumulative = positive.detected_not_quantified.cumsum().to_numpy()
    summary = dict(scope=SCOPE, definition='pair is included if at least one observed record enters recalculated Q',
                   well_denominator='all cleaned non-Finkelstein records',
                   wells=len(u), pairs=len(pairs), detected_not_quantified=total_missing,
                   detected_missing_on_excluded_pairs=excluded_missing,
                   fraction_detected_missing_on_excluded_pairs=excluded_missing / total_missing,
                   pairs_with_detected_missing=len(positive),
                   pairs_covering_half_detected_missing=int(np.searchsorted(cumulative, total_missing * .5) + 1),
                   pairs_covering_four_fifths_detected_missing=int(np.searchsorted(cumulative, total_missing * .8) + 1),
                   qualification=rows, missing_filled_zero=0,
                   zero_by_rule_observed=int(u.zero_by_rule.sum()))
    out = out_dir(variant)
    pairs.reset_index().to_csv(out / 'measurement_qualification_pairs.csv', index=False)
    grouped.to_csv(out / 'measurement_qualification_groups.csv', index=False)
    (out / 'measurement_qualification_link.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
