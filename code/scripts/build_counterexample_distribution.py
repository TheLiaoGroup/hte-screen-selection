"""Export Figure 4c array counts from the computed counterexample classifications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def distribution(edges: pd.DataFrame, recipes: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ['rxn_class', 'edge_id', 'new_contradiction', 'minimum_O']
    if not set(columns).issubset(edges):
        raise ValueError('Counterexample classifications lack required fields')
    if edges.edge_id.isna().any() or edges.edge_id.duplicated().any():
        raise ValueError('Recipe comparison identity must be unique and nonmissing')
    flags = edges.new_contradiction
    if flags.isna().any() or not flags.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError('Counterexample target flags must be nonmissing booleans')
    if recipes is not None:
        keys = ['rxn_class', 'edge_id']
        if not set(keys).issubset(recipes) or len(recipes) != len(edges):
            raise ValueError('Recipe inventory differs from the classified comparisons')
        edges = recipes[keys].merge(edges[columns], on=keys, how='left', validate='one_to_one')
        if edges.new_contradiction.isna().any():
            raise ValueError('Recipe inventory does not reconcile with the classifications')
        flags = edges.new_contradiction
    members = edges.loc[flags, columns].copy()
    minimum = members.minimum_O.to_numpy(float)
    if (members.rxn_class.isna().any() or not np.isfinite(minimum).all()
            or (minimum < 1).any() or (minimum != np.floor(minimum)).any()):
        raise ValueError('Counterexample target classification is invalid')
    members['single'] = members.minimum_O.eq(1).astype(int)
    members['collective'] = members.minimum_O.gt(1).astype(int)
    counts = members.groupby('rxn_class')[['single', 'collective']].sum()
    counts['total'] = counts.sum(axis=1)
    counts = counts.sort_values(['total', 'single'], ascending=[False, False]).reset_index()
    return counts, members


def build(out: Path, destination: Path | None = None) -> dict:
    out = Path(out)
    edges = pd.read_csv(out / 'm2/partner_score_edges.csv')
    recipes = pd.read_csv(out / 'm2/recipe_edges.csv', usecols=['rxn_class', 'edge_id'])
    counts, members = distribution(edges, recipes)
    scope = (len(members), int(counts.single.sum()), int(counts.collective.sum()), len(counts))
    if scope != (177, 111, 66, 11):
        raise ValueError('Required 177/111/66/11 counterexample scope differs')
    destination = Path(destination) if destination is not None else out / 'counterexample_distribution'
    destination.mkdir(parents=True, exist_ok=True)
    counts.to_csv(destination / 'Figure_4c_array_distribution.csv', index=False)
    members.to_csv(destination / 'Figure_4c_classification_members.csv', index=False)
    return dict(zip(('counterexamples', 'single', 'collective', 'arrays'), scope))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE / 'out')
    parser.add_argument('--destination', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.out, arguments.destination), indent=2))
