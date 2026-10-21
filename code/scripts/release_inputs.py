"""Read the published cleaned corpus without applying endpoint correction twice."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_ENV = 'HTE_CLEANED_RELEASE_MANIFEST'
PROVENANCE = ['source_file', 'source_row']
MARKER = 'verified_cleaned_release_manifest'


def active_manifest() -> Path | None:
    value = os.environ.get(MANIFEST_ENV)
    return Path(value).resolve() if value else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def _manifest(path: str, digest: str) -> dict:
    location = Path(path)
    result = json.loads(location.read_text(encoding='utf-8'))
    if (not result.get('files') or result.get('excluded_name0_present') != 0
            or result.get('endpoint_disagreements') != 0
            or result.get('original_field_disagreements') != 0):
        raise ValueError('Cleaned-release manifest lacks successful input verification')
    entries = result['files']
    names = [entry['file'] for entry in entries]
    if len(set(names)) != len(names) or len(entries) != result['arrays']:
        raise ValueError('Cleaned-release manifest has an ambiguous table inventory')
    for entry in entries:
        relative = Path(entry['file'])
        if relative.parts[:1] != ('data',) or len(relative.parts) != 2:
            raise ValueError('Cleaned-release tables must be directly inside data/')
        target = (location.parent / relative).resolve()
        if target.parent != (location.parent / 'data').resolve() or target.suffix != '.csv':
            raise ValueError('Unsafe cleaned-release table path')
        if _sha256(target) != entry['sha256']:
            raise ValueError(f'Cleaned-release table hash mismatch: {entry["file"]}')
    if sum(entry['records'] for entry in entries) != result['records']:
        raise ValueError('Cleaned-release manifest record total disagrees with its tables')
    actual = {p.relative_to(location.parent).as_posix() for p in
              (location.parent / 'data').iterdir() if p.is_file()}
    if actual != set(names):
        raise ValueError('Cleaned-release data directory differs from its manifest')
    return result


def verified_manifest(path: Path) -> dict:
    return _manifest(str(path.resolve()), _sha256(path))


def read_release_table(path: Path) -> pd.DataFrame | None:
    """Return original measurement fields, or None for the archival raw route."""
    manifest_path = active_manifest()
    if manifest_path is None:
        return None
    manifest = verified_manifest(manifest_path)
    relative = path.resolve().relative_to(manifest_path.parent).as_posix()
    entries = [row for row in manifest['files'] if row['file'] == relative]
    if len(entries) != 1:
        raise ValueError(f'Input table is absent from the cleaned-release manifest: {path.name}')
    if _sha256(path) != entries[0]['sha256']:
        raise ValueError(f'Cleaned-release table hash mismatch: {path.name}')
    header = pd.read_csv(path, nrows=0)
    dtypes = {column: 'string' for column in header if column != 'yield'}
    frame = pd.read_csv(path, dtype={**dtypes, 'yield': float}, keep_default_na=False,
                        na_values=[''], float_precision='round_trip')
    required = {*PROVENANCE, 'array_id', 'name0', 'raw_yield', 'yield',
                'measurement_status', 'yield_clf'}
    if not required.issubset(frame) or len(frame) != entries[0]['records']:
        raise ValueError(f'Cleaned-release table has incompatible rows or fields: {path.name}')
    if frame.source_file.nunique() != 1 or frame.array_id.nunique() != 1:
        raise ValueError('Cleaned-release table must represent one original source and array')
    frame['source_row'] = pd.to_numeric(frame.source_row, errors='raise').astype(np.int64)
    if (frame[PROVENANCE].duplicated().any() or frame.source_row.lt(2).any()
            or not frame.source_row.is_monotonic_increasing):
        raise ValueError('Cleaned-release source provenance is invalid')
    raw = pd.to_numeric(frame.raw_yield, errors='raise').astype(float)
    clf = pd.to_numeric(frame.yield_clf, errors='raise').astype(float)
    expected = raw.mask(raw.isna() & clf.eq(0), 0.0)
    corrected = pd.to_numeric(frame['yield'], errors='raise').astype(float)
    if not np.allclose(expected, corrected, atol=0, rtol=0, equal_nan=True):
        raise ValueError('Cleaned-release endpoints disagree with the original measurement fields')
    status = np.select([raw.notna(), clf.eq(0), clf.eq(1)],
                       ['quantified', 'zero_by_rule', 'detected_not_quantified'],
                       default='no_screen_record')
    if not frame.measurement_status.eq(status).all():
        raise ValueError('Cleaned-release measurement states disagree with the original fields')
    frame = frame.drop(columns=['yield', 'measurement_status', 'array_id']).rename(
        columns={'raw_yield': 'yield'})
    frame.attrs[MARKER] = {'path': str(manifest_path), 'sha256': _sha256(manifest_path)}
    return frame


def validate_precleaned_frame(frame: pd.DataFrame) -> bool:
    """Accept precleaned input only after checking the complete manifest corpus."""
    manifest_path = active_manifest()
    if manifest_path is None:
        if MARKER in frame.attrs:
            raise ValueError('Cleaned-release marker requires an explicit manifest')
        return False
    marker = {'path': str(manifest_path), 'sha256': _sha256(manifest_path)}
    if frame.attrs.get(MARKER) != marker:
        raise ValueError('Cleaned-release input did not pass the manifest reader')
    manifest = verified_manifest(manifest_path)
    if len(frame) != manifest['records'] or frame[PROVENANCE].duplicated().any():
        raise ValueError('Cleaned-release ingestion requires the complete unique corpus')
    expected_keys = set()
    for entry in manifest['files']:
        expected = read_release_table(manifest_path.parent / entry['file'])
        source = expected.source_file.iloc[0]
        actual = frame.loc[frame.source_file.eq(source)].reset_index(drop=True)
        if len(actual) != len(expected):
            raise ValueError(f'Cleaned-release source record count differs: {source}')
        for column in expected:
            if column == 'idx' and column not in actual:
                continue
            if (column not in actual or not actual[column].astype('string').fillna('').eq(
                    expected[column].astype('string').fillna('')).all()):
                raise ValueError(f'Cleaned-release source field differs: {source}, {column}')
        expected_keys.add(source)
    if set(frame.source_file) != expected_keys:
        raise ValueError('Unexpected source in cleaned-release corpus')
    return True
