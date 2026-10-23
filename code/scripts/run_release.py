"""Replay primary analyses using only a local code-and-cleaned-data release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from release_inputs import MANIFEST_ENV, verified_manifest


def verify_bundle(root: Path) -> dict:
    record = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    if not record.get('files'):
        raise ValueError('Release manifest has no files')
    for relative, expected in record['files'].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError(f'Unsafe or missing release file: {relative}')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Release file hash mismatch: {relative}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'
              and p != root / 'release_manifest.json'}
    if actual != set(record['files']):
        raise ValueError('Release file inventory differs from the manifest')
    verified_manifest(root / 'cleaned_corpus' / 'manifest.json')
    return record


def compare_reference(root: Path, output: Path, stage: str) -> dict:
    expected = json.loads((root / 'reference_outputs.json').read_text(encoding='utf-8'))
    matched, different = [], []
    for relative, digest in expected['files'].items():
        target = (output / relative).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError(f'Unsafe reference file: {relative}')
        if stage == 'ingest' and relative not in INGEST_OUTPUTS:
            continue
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            different.append(relative)
        else:
            matched.append(relative)
    if different or not matched:
        raise ValueError(f'Release replay differs from reference science outputs: {different}')
    return {'stage': stage, 'matched_files': len(matched), 'byte_identical': True,
            'files': matched}


INGEST_OUTPUTS = {
    'triage/all_table_inventory.csv',
    *(f'triage/{scope}/{name}' for scope in ('primary', 'reference') for name in (
        'raw_inventory_by_reaction.csv', 'triage_by_reaction.csv', 'unquantifiable_by_reaction.csv',
        'pair_best_well_desi_call.csv', 'pair_maximum_thresholds.csv', 'unquantifiable_by_pair.csv',
        'chan_lam_20pct_boundary_all_rows.csv', 'chan_lam_20pct_boundary_pairs.csv',
        'negative_numeric_by_plate.csv', 'negative_positive_numeric_wells.csv', 'summary.json')),
    'm2/processed/block_inventory.csv', 'm2/processed/pair_inventory.csv',
    'm2/processed/qualification_inventory.csv', 'm2/processed/legacy_outside_reconstruction.csv',
    'm2/domain_counts.json', 'm2/rule_designs.csv', 'm2/rule_masks.npz',
    'm2/rule_union_counts.json', 'm2/recipe_dual_counts.json', 'm2/recipe_duals.parquet',
    'm2/recipe_edges.csv',
    'm2/measurement_loss_summary.json', 'm2/measurement_loss_by_contrast.csv',
    'm2/measurement_loss_comparisons.parquet', 'm2/measurement_loss_rectangles.csv',
    'm2/measurement_qualification_groups.csv', 'm2/measurement_qualification_link.json',
    'm2/measurement_qualification_pairs.csv',
}


def science_commands(root: Path, output: Path, stage: str) -> list[list[str]]:
    scripts, corpus = root / 'code/scripts', root / 'cleaned_corpus'
    commands = [
        [sys.executable, str(scripts / 'triage/run_triage.py'),
         '--raw', str(corpus / 'data'), '--out', str(output / 'triage')],
        [sys.executable, str(scripts / 'm2/run_analysis.py'), '--stage', stage,
         '--raw-dir', str(corpus / 'data'), '--output-dir', str(output / 'm2')],
    ]
    if stage == 'primary':
        commands.append([
            sys.executable, str(scripts / 'build_science_extensions.py'),
            '--input-dir', str(output / 'm2'),
            '--output-dir', str(output / 'science_extensions'),
            '--compound-catalogue', str(root / 'code/data/catalogues/0_Compound ID_260810.xlsx'),
            '--only', 'all'])
        commands.extend([
            [sys.executable, str(scripts / 'build_noise_margin.py'), '--out', str(output)],
            [sys.executable, str(scripts / 'build_counterexample_distribution.py'),
             '--out', str(output)],
        ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['ingest', 'primary'], default='primary')
    parser.add_argument('--verify-reference', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root, output = args.release_root.resolve(), args.out.resolve()
    manifest = verify_bundle(root)
    if args.verify_only:
        print(json.dumps({'verified_files': len(manifest['files']), 'status': 'verified'}))
        return
    if output.is_relative_to(root):
        parser.error('--out must be outside the immutable release bundle')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty')
    corpus = root / 'cleaned_corpus'
    env = dict(os.environ, **{MANIFEST_ENV: str(corpus / 'manifest.json')},
               M2_RAW_DIR=str(corpus / 'data'), M2_OUTPUT_DIR=str(output / 'm2'),
               M2_CONDITION_CATALOGUE=str(root / 'code' / 'data' / 'catalogues' /
                                         '0_total condition_260810.xlsx'),
               M2_COMPOUND_CATALOGUE=str(root / 'code' / 'data' / 'catalogues' /
                                        '0_Compound ID_260810.xlsx'),
               PYTHONDONTWRITEBYTECODE='1',
               PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    for command in science_commands(root, output, args.stage):
        subprocess.run(command, check=True, cwd=root, env=env)
    if args.verify_reference:
        result = compare_reference(root, output, args.stage)
        (output / 'REPLAY_VERIFICATION.json').write_text(
            json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
