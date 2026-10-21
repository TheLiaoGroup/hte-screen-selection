"""Describe analytical missingness in the outside pairs of selected minimum cycle certificates."""
import argparse
from importlib import import_module
import json
import networkx as nx
import pandas as pd
from m2common import data_dir, out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='clfzero')
    variant = parser.parse_args().variant
    out = out_dir(variant)
    u = pd.read_parquet(data_dir(variant) / 'observations.parquet')
    observed = u.set_index('record_id')
    edges = pd.read_csv(out / 'partner_score_edges.csv')
    new = edges.loc[edges.new_contradiction]
    assert len(new) > 0 and new.edge_id.is_unique
    links = []
    for edge in new.itertuples(index=False):
        for dual in json.loads(edge.certificate_dual_ids):
            ids = dual.split('|')
            assert len(ids) == 2
            rows = observed.loc[ids]
            assert rows.analysis_valid.all() and rows.subpair.nunique() == 1
            assert rows.rxn_class.eq(edge.rxn_class).all() and rows.pair_group.nunique() == 1
            if rows.pair_group.iloc[0] == 'outside_only':
                links.append(dict(edge_id=edge.edge_id, rxn_class=edge.rxn_class,
                                  subpair=rows.subpair.iloc[0], dual_id=dual))
    links = pd.DataFrame(links)
    assert links.edge_id.nunique() == len(new)
    pairs = pd.read_csv(out / 'measurement_qualification_pairs.csv')
    valid_pairs = u.loc[u.analysis_valid & u.pair_group.eq('outside_only'), ['rxn_class', 'subpair']].drop_duplicates()
    pairs = pairs.merge(valid_pairs, on=['rxn_class', 'subpair'], validate='one_to_one')
    membership = links[['rxn_class', 'subpair']].drop_duplicates().assign(certificate_pair=True)
    pairs = pairs.merge(membership, on=['rxn_class', 'subpair'], how='left', validate='one_to_one')
    pairs['certificate_pair'] = pairs.certificate_pair.eq(True)
    by_class = []
    for cls, group in pairs.groupby('rxn_class', sort=True):
        row = dict(rxn_class=cls)
        for label, flag in [('certificate', True), ('remaining', False)]:
            g = group.loc[group.certificate_pair.eq(flag)]
            row.update({f'{label}_pairs': len(g), f'{label}_wells': int(g.wells.sum()),
                        f'{label}_detected_not_quantified': int(g.detected_not_quantified.sum()),
                        f'{label}_pair_mean': float(g.unquantifiable_fraction.mean())})
        row['comparison_eligible'] = row['certificate_pairs'] > 0 and row['remaining_pairs'] > 0
        by_class.append(row)
    by_class = pd.DataFrame(by_class)
    eligible = by_class.loc[by_class.comparison_eligible]
    assert len(eligible) > 0 and pairs.certificate_pair.any() and (~pairs.certificate_pair).any()
    joined = links.merge(pairs[['rxn_class', 'subpair', 'detected_not_quantified']],
                         on=['rxn_class', 'subpair'], validate='many_to_one')
    has_missing = joined.groupby('edge_id').detected_not_quantified.max().gt(0)
    affected_pairs = set(pairs.loc[pairs.detected_not_quantified.gt(0), ['rxn_class', 'subpair']].itertuples(index=False, name=None))
    duals = pd.read_parquet(out / 'recipe_duals.parquet')
    ps = import_module('10_partner_score_counterexamples')
    strong = duals.loc[duals.core_outside_domain & duals.edge_id.isin(new.edge_id) &
                       duals.d.abs().ge(ps.MARGIN / 100 - ps.TOL)].copy()
    strong['dual_id'] = strong.record_id_A + '|' + strong.record_id_B
    witnesses = []
    new_index = new.set_index('edge_id')
    for edge_id, group in strong.groupby('edge_id', sort=True):
        graph = ps.build_graph(group)
        components = {node: i for i, nodes in enumerate(nx.strongly_connected_components(graph)) for node in nodes}
        scale = graph.number_of_nodes() + 1
        for a, b, attributes in graph.edges(data=True):
            attributes['distance'] = scale * (attributes['cohort'] == 'O') + 1
        affected = group.loc[group.identity_A.eq('O') &
            pd.Series([(cls, pair) in affected_pairs for cls, pair in zip(group.rxn_class, group.subpair)], index=group.index)]
        n_cycle = n_minimum = 0
        expected = int(new_index.loc[edge_id, 'minimum_O']) * scale + int(new_index.loc[edge_id, 'certificate_length'])
        for row in affected.itertuples(index=False):
            left, right = (1, str(row.sub_1_smiles)), (2, str(row.sub_2_smiles))
            source, target = (right, left) if row.d > 0 else (left, right)
            if components[source] == components[target]:
                n_cycle += 1
                distance = nx.shortest_path_length(graph, target, source, weight='distance')
                n_minimum += distance + scale + 1 == expected
        witnesses.append(dict(edge_id=edge_id, strong_outside_affected_duals=len(affected),
                              affected_duals_on_any_cycle=n_cycle,
                              affected_duals_on_a_minimum_certificate=n_minimum))
    witnesses = pd.DataFrame(witnesses)
    assert len(witnesses) == len(new)
    selected_affected = has_missing.reindex(witnesses.edge_id).to_numpy()
    assert (witnesses.affected_duals_on_a_minimum_certificate.gt(0).to_numpy() | ~selected_affected).all()
    assert (witnesses.affected_duals_on_any_cycle >= witnesses.affected_duals_on_a_minimum_certificate).all()
    summary = dict(analysis_status='post hoc selected-certificate comparison and exhaustive cycle-existence analysis',
                   new_counterexamples=len(new),
                   counterexamples_with_unquantifiable_wells_in_an_outside_certificate_pair=int(has_missing.sum()),
                   counterexamples_with_any_cycle_using_an_affected_outside_pair=int(witnesses.affected_duals_on_any_cycle.gt(0).sum()),
                   counterexamples_with_a_minimum_cycle_using_an_affected_outside_pair=int(witnesses.affected_duals_on_a_minimum_certificate.gt(0).sum()),
                   affected_pair_definition='outside-only pair has at least one detected-not-quantified well in its complete raw history',
                   minimum_cycle_definition='minimum number of outside duals, then minimum cycle length; all identity tie-breaks admitted',
                   outside_pairs=len(pairs), selected_certificate_pairs=int(pairs.certificate_pair.sum()),
                   comparison_classes=int(len(eligible)), groups=[])
    for label, flag in [('certificate', True), ('remaining', False)]:
        g = pairs.loc[pairs.certificate_pair.eq(flag)]
        summary['groups'].append(dict(group=label, pairs=len(g), wells=int(g.wells.sum()),
            detected_not_quantified=int(g.detected_not_quantified.sum()),
            pooled_well_fraction=float(g.detected_not_quantified.sum() / g.wells.sum()),
            pair_mean_fraction=float(g.unquantifiable_fraction.mean()),
            class_equal_pair_mean_fraction=float(eligible[f'{label}_pair_mean'].mean())))
    pairs.to_csv(out / 'certificate_measurement_pairs.csv', index=False)
    by_class.to_csv(out / 'certificate_measurement_by_class.csv', index=False)
    joined.to_csv(out / 'certificate_measurement_links.csv', index=False)
    witnesses.to_csv(out / 'certificate_measurement_all_witnesses.csv', index=False)
    (out / 'certificate_measurement_link.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
