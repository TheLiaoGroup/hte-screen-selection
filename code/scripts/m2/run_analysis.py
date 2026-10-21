"""Recompute qualification and retained counterexamples from local raw tables."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parents[1]
STAGES = {
    'ingest': ['01_build_observations', '02_domain_counts', '04_rule_union', '05_recipe_duals',
               '06_measurement_loss',
               '77_measurement_qualification_link'],
    'counterexamples': ['10_partner_score_counterexamples', '78_certificate_measurement_link',
                        '79_measurement_loss_link', '80_chemical_example', '20_constraint_strength'],
    'restoration': ['21_joint_restoration', '22_restoration_family'],
    'retention': [('74_primary_retention', 'rules'), ('74_primary_retention', 'e1')],
    'supporting': ['30_fixed_change_backgrounds', '40_structural_series', '41_crossgoal_targets',
                   '42_joint_preference_counterexample', '43_heldout_crossgoal_replay',
                   '50_opportunity_outside', '63_campaign_halves', '23_restoration_tolerance'],
    'secondary-retention': [('74_primary_retention', 'e2'), ('74_primary_retention', 'e3'),
                            ('74_primary_retention', 'e4'), '68_rules_crossgoal',
                            '69_retention_evidence_summary'],
    'molecular': ['71_molecule_failure', '72_failure_common_conditions', '74_missing_well_types'],
}
PRIMARY = ['ingest', 'counterexamples', 'restoration', 'retention']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=[*STAGES, 'primary', 'all'], default='primary')
    parser.add_argument('--raw-dir', type=Path, default=REPRODUCE / 'data' / 'raw')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    default_output = REPRODUCE / 'out' / 'm2'
    output = (args.output_dir or default_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, M2_VARIANT='clfzero', M2_SCOPE='cleaned',
               M2_RAW_DIR=str(args.raw_dir.resolve()), M2_OUTPUT_DIR=str(output),
               PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    stages = list(STAGES) if args.stage == 'all' else PRIMARY if args.stage == 'primary' else [args.stage]
    completed = []
    for stage in stages:
        for task in STAGES[stage]:
            name, endpoint = task if isinstance(task, tuple) else (task, None)
            cmd = [sys.executable, str(HERE / f'{name}.py')]
            if name != '63_campaign_halves':
                cmd += ['--variant', 'clfzero']
            if endpoint:
                cmd += ['--stage', endpoint, '--workers', str(args.workers)]
            print(f'Running {name}' + (f' ({endpoint})' if endpoint else ''), flush=True)
            subprocess.run(cmd, check=True, cwd=HERE, env=env)
            completed.append(name + (f':{endpoint}' if endpoint else ''))
    print(f'Completed {len(completed)} tasks in cleaned scope.', flush=True)


if __name__ == '__main__':
    main()
