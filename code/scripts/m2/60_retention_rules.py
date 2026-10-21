"""Build yield-blind retention rules and report their record budgets."""
import json
import numpy as np
from m2common import out_dir
from retention import Corpus, build_rules, anchor_R4, anchor_R4p

c = Corpus()
assert c.pool.sum() > 0 and c.Q.sum() > 0
rules = build_rules(c)
res = {}
for k, a in rules.items():
    assert c.pool[a].all() and len(np.unique(a)) == len(a)
    res[k] = dict(added_records=int(len(a)), added_pairs=int(c.u.iloc[a][['rxn_class', 'subpair']].drop_duplicates().shape[0]))
for k in ('R1-12',):  # anchor self-check at one budget
    b = len(rules[k])
    assert len(anchor_R4(c, b, 0)) == b and len(anchor_R4p(c, b, 0)) == b
np.savez_compressed(out_dir() / 'retention_rules.npz', **{k: v for k, v in rules.items()})
(out_dir() / 'retention_rule_budgets.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps(res, indent=1))
