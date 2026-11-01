"""Build reproducible controlled-capacity traces for a fixed ABR diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from abr.data import Dataset, Trace


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'experiments/scenarios/controlled_capacity_v1.json'
OUTPUT = ROOT / 'data/controlled-capacity'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(rng, bounds):
    low, high = bounds
    return float(rng.uniform(low, high))


def draw_int(rng, bounds):
    low, high = bounds
    return int(rng.integers(low, high + 1))


def profile(family, split, index, rng, horizon, generation):
    capacity = np.full(horizon, 4500.0, dtype=np.float64)
    details = {'family': family, 'split': split, 'index': index}
    if family == 'step_drop_recovery':
        baseline = draw(rng, generation['baseline_kbps'])
        start = draw_int(rng, generation['drop_start_s'])
        duration = draw_int(rng, generation['drop_duration_s'])
        drop_end = min(horizon, start + duration)
        recovery = draw(rng, generation['recovery_fraction_of_baseline']) * baseline
        capacity[:] = baseline
        capacity[start:drop_end] = draw(rng, generation['drop_kbps'])
        capacity[drop_end:] = recovery
        details.update({'baseline_kbps': baseline, 'drop_start_s': start,
                        'drop_end_s': drop_end, 'recovery_kbps': recovery})
    elif family == 'repeated_capacity_changes':
        transition_count = int(generation['transition_count'])
        levels = rng.uniform(*generation['plateau_kbps'], transition_count + 1)
        points = []
        cursor = draw_int(rng, generation['first_transition_s'])
        for _ in range(transition_count):
            points.append(cursor)
            cursor += draw_int(rng, generation['subsequent_transition_spacing_s'])
        boundaries = [0, *points, horizon]
        for start, end, level in zip(boundaries[:-1], boundaries[1:], levels):
            capacity[start:end] = level
        details.update({'transition_s': points, 'plateau_kbps': levels.tolist()})
    elif family == 'zero_outage_then_degraded_recovery':
        baseline = draw(rng, generation['baseline_kbps'])
        start = draw_int(rng, generation['outage_start_s'])
        outage_duration = draw_int(rng, generation['outage_duration_s'])
        degraded_duration = draw_int(rng, generation['degraded_duration_s'])
        degraded_level = draw(rng, generation['degraded_kbps'])
        recovery = draw(rng, generation['recovery_fraction_of_baseline']) * baseline
        outage_end = min(horizon, start + outage_duration)
        degraded_end = min(horizon, outage_end + degraded_duration)
        capacity[:] = baseline
        capacity[start:outage_end] = 0.0
        capacity[outage_end:degraded_end] = degraded_level
        capacity[degraded_end:] = recovery
        details.update({'baseline_kbps': baseline, 'outage_start_s': start,
                        'outage_end_s': outage_end, 'degraded_end_s': degraded_end,
                        'degraded_kbps': degraded_level, 'recovery_kbps': recovery})
    else:
        raise ValueError(f'Unknown controlled scenario family: {family}')
    if not np.isfinite(capacity).all() or (capacity < 0).any():
        raise ValueError(f'Invalid generated capacity values for {family}/{split}/{index}')
    return capacity, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    output_root = args.output.resolve()
    spec = json.loads(spec_path.read_text())
    horizon = int(spec['trace_duration_s'] / spec['interval_s'])
    traces_by_split = {split: [] for split in ('train', 'validation', 'test')}
    records = []
    family_names = spec['families']
    split_offsets = {'train': 11, 'validation': 29, 'test': 47}
    for split in ('train', 'validation', 'test'):
        count = spec['split_sizes_per_family'][split]
        for family_index, family in enumerate(family_names):
            rng = np.random.default_rng(spec['generator_seed'] + split_offsets[split] + family_index * 1009)
            for index in range(count):
                values, details = profile(family, split, index, rng, horizon,
                                          spec['generation'][family])
                name = f'{split}-{family}-{index:02d}'
                trace = Trace(
                    name=name,
                    durations_s=np.full(horizon, float(spec['interval_s'])),
                    bandwidth_kbps=values,
                    rtt_s=float(spec['rtt_s']),
                    source_group=f'{spec["scenario_set_id"]}-{name}',
                    source=(f'{spec["scenario_set_id"]}; controlled application goodput; '
                            f'spec_sha256={digest(spec_path)}'),
                    semantics='controlled_capacity',
                )
                traces_by_split[split].append(trace)
                records.append({**details, 'name': name, 'trace_sha256': trace.fingerprint,
                                'minimum_kbps': float(values.min()),
                                'maximum_kbps': float(values.max()),
                                'mean_kbps': float(values.mean())})

    video_sources = {
        'envivio': ROOT / 'data/startup-ladder/training.json',
        'sintel': ROOT / 'data/sintel-ladder/training.json',
    }
    video_records = {}
    for content, source in video_sources.items():
        video_dataset = Dataset.load(source)
        output = output_root / f'{content}.json'
        dataset = Dataset(video_dataset.video,
                          {split: tuple(traces) for split, traces in traces_by_split.items()},
                          f'Controlled capacity scenario set {spec["scenario_set_id"]}; '
                          f'video ladder from {source.relative_to(ROOT)}; synthetic controlled link capacities.')
        dataset.save(output)
        video_records[content] = {
            'source_dataset': str(source.relative_to(ROOT)),
            'source_dataset_sha256': digest(source),
            'dataset': str(output.relative_to(ROOT)),
            'dataset_sha256': dataset.fingerprint,
            'video_bitrates_kbps': dataset.video.bitrates_kbps.tolist(),
            'video_segments': len(dataset.video.durations_s),
            'video_duration_s': float(dataset.video.durations_s.sum()),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        'scenario_set_id': spec['scenario_set_id'],
        'spec': str(spec_path.relative_to(ROOT)),
        'spec_sha256': digest(spec_path),
        'generator': str(Path(__file__).resolve().relative_to(ROOT)),
        'generator_sha256': digest(__file__),
        'generator_seed': spec['generator_seed'],
        'controlled_interval_s': spec['interval_s'],
        'trace_duration_s': spec['trace_duration_s'],
        'rtt_s': spec['rtt_s'],
        'semantics': 'controlled_capacity',
        'split_counts': {split: len(traces) for split, traces in traces_by_split.items()},
        'content_datasets': video_records,
        'scenarios': records,
    }
    (output_root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output_root / 'README.md').write_text(
        '# Controlled capacity diagnostic data\n\n'
        'These traces are generated controlled application-goodput profiles, not measured network data. '
        'They cover step drops, repeated changes, and a zero-capacity outage followed by degraded service and recovery. '
        'The test profiles use independent parameter draws and are not used for training or checkpoint selection.\n\n'
        'Both content datasets use the same network scenarios. Envivio has 49 segments; Sintel has 14. '
        'They share the same seven nominal bitrate choices but have different measured sizes and durations.\n\n'
        'The scenario definition, random seeds, trace hashes, and dataset fingerprints are in `provenance.json`. '
        'Rebuild with `.venv/bin/python -m scripts.prepare_controlled_capacity`.\n')
    print(json.dumps({'scenario_set_id': spec['scenario_set_id'],
                      'datasets': video_records,
                      'split_counts': provenance['split_counts']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
