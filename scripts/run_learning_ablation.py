"""Run six predeclared short experiments using at most two CPU workers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def run(item):
    variant, seed = item
    output = Path('runs/learning-ablation') / f'{variant}-seed-{seed}'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'training.log').open('w') as log:
        subprocess.run([sys.executable, '-u', '-m', 'abr.learning_experiment', '--variant', variant,
                        '--seed', str(seed), '--episodes', '120', '--output', str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((output / 'metrics.json').read_text())
    report = metrics['selected_validation']
    print(json.dumps({'variant': variant, 'seed': seed, 'completed': True,
                      'selected_validation': None if report is None else
                      {k: v for k, v in report.items() if k != 'episodes'}}), flush=True)
    return metrics


def main():
    schedule = [(variant, seed) for variant in ('baseline', 'credit', 'queue') for seed in (7, 11)]
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(run, schedule))
    output = Path('runs/learning-ablation')
    (output / 'selection_inputs.json').write_text(json.dumps([
        {k: r[k] for k in ('variant', 'seed', 'episodes', 'accepted_checkpoint', 'selected_validation',
                          'final_validation', 'max_episode_return_shift_error', 'selected_q_calibration', 'final_q_calibration')}
        for r in results], indent=2) + '\n')


if __name__ == '__main__':
    main()
