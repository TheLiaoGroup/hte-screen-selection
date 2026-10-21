"""Campaign halves: repeat the preference-graph analysis within each half of the
source campaigns. Labels are ordered by SHA-256 of the label and dealt alternately to H1 and H2;
records without a label form their own label. A dual belongs to a half only if both records do.
"""
from hashlib import sha256
from importlib import import_module
import json

import networkx as nx
import pandas as pd
from m2common import out_dir

ps = import_module('10_partner_score_counterexamples')
MARGIN = ps.MARGIN / 100 - ps.TOL


def main():
    d = pd.read_parquet(out_dir() / 'recipe_duals.parquet')
    d = d.loc[d.core_outside_domain].copy()
    d['dual_id'] = d.record_id_A.astype(str) + '|' + d.record_id_B.astype(str)
    for s in 'AB':
        d[f'c{s}'] = d[f'source_campaign_{s}'].astype('string').fillna('<unlabelled>').replace('', '<unlabelled>')
    labels = sorted(set(d.cA) | set(d.cB), key=lambda x: sha256(x.encode('utf-8')).hexdigest())
    half = {lab: i % 2 for i, lab in enumerate(labels)}
    d['hA'], d['hB'] = d.cA.map(half), d.cB.map(half)
    strong = d.loc[d.d.abs().ge(MARGIN)]
    rows = []
    for h in (0, 1):
        z = strong.loc[strong.hA.eq(h) & strong.hB.eq(h)]
        for (cls, e), g in z.groupby(['rxn_class', 'edge_id'], sort=True):
            q = g.loc[g.identity_A.eq('Q')]
            new = len(q) > 0 and nx.is_directed_acyclic_graph(ps.build_graph(q)) and \
                not nx.is_directed_acyclic_graph(ps.build_graph(g))
            if new:
                rows.append(dict(half=f'H{h + 1}', rxn_class=cls, edge_id=e))
    res = pd.DataFrame(rows)
    full = pd.read_csv(out_dir() / 'partner_score_edges.csv')
    full179 = set(full.loc[full.new_contradiction, 'edge_id'])
    h1, h2 = (set(res.loc[res.half.eq(h), 'edge_id']) for h in ('H1', 'H2'))
    summary = dict(campaign_labels=len(labels), H1_labels=sum(v == 0 for v in half.values()),
                   H1_new=len(h1), H2_new=len(h2), both_halves=len(h1 & h2),
                   H1_in_full179=len(h1 & full179), H2_in_full179=len(h2 & full179),
                   H1_classes=res.loc[res.half.eq('H1')].rxn_class.value_counts().to_dict(),
                   H2_classes=res.loc[res.half.eq('H2')].rxn_class.value_counts().to_dict())
    res.to_csv(out_dir() / 'campaign_halves_edges.csv', index=False)
    (out_dir() / 'campaign_halves.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
