"""Fetch the first four unused ferry sessions in archive filename order.

Save the experiment settings before downloading or replaying these sessions.
Run: python -m scripts.prepare_buffer_experiment
"""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from abr.data import Dataset, Trace
from abr.env import StreamConfig
from abr.models.safety import ActionSafety
from abr.recovery import RecoveryConfig


SESSIONS = ('2010-09-22_0702CEST', '2010-09-23_1001CEST',
            '2010-09-28_1003CEST', '2010-09-29_0702CEST')
BASE = 'https://datasets.simula.no/downloads/hsdpa-tcp-logs/ferry.nesoddtangen-oslo/'


def main():
    output = Path('runs/buffer-experiment')
    output.mkdir(parents=True, exist_ok=True)
    destination = Path('data/buffer-sessions')
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('runs/learning-ablation/baseline-seed-7/accepted.npz')
    protocol = {
        'hypothesis': 'Larger startup/resume buffers may reduce full-playback stalls at a cost in startup or resume waiting',
        'sessions': list(SESSIONS),
        'session_selection': 'First four unused ferry sessions in archive filename order; selected from directory metadata, before viewing trace values',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'training': False,
        'profiles': [{'name': f'start{startup}-resume{resume}',
                      'config': asdict(StreamConfig(startup_s=startup, resume_s=resume))}
                     for startup in (1., 4., 8.) for resume in (1., 4., 8.)],
        'policies': ['lowest', 'highest_allowed', 'dqn'],
        'recovery': asdict(RecoveryConfig()), 'safety_filter': asdict(ActionSafety()),
        'targets': {'max_startup_s': 3., 'max_rebuffer_s': 8., 'completion_fraction': 1.},
        'decision_rule': 'Compare all predeclared profiles; do not promote or select a winner using these four sessions',
        'primary_dataset': 'data/buffer-sessions/dataset.json',
        'development_control': 'data/learning-holdout/dataset.json',
        'limitations': ['Same VOD and ferry route, four additional sessions only',
                        'Frozen DQN was trained with startup/resume thresholds of 1 s; changed context is untrained',
                        'Thresholds are media seconds; whole segments arrive together (first segment 3.9934 s)',
                        'All nine profiles share these sessions; this is exploratory comparison, not independent validation of a selected profile',
                        'Observed goodput replay, RTT assumed 80 ms; no packet/radio simulation',
                        'Simulator only; browser unchanged']}
    protocol_path = output / 'protocol.json'
    serialized = json.dumps(protocol, indent=2) + '\n'
    if protocol_path.exists() and protocol_path.read_text() != serialized:
        raise ValueError('Existing protocol differs; preserve the predeclared experiment')
    protocol_path.write_text(serialized)
    original = Dataset.load('data/hard_envivio/dataset.json')
    known = [Dataset.load(path) for path in ('data/hard_envivio/dataset.json',
             'data/learning-holdout/dataset.json', 'data/realistic_envivio/dataset.json')]
    known_hashes = {t.fingerprint for d in known for traces in d.splits.values() for t in traces}
    traces, raw_metadata = [], []
    for session in SESSIONS:
        filename = f'report.{session}.log'
        path = destination / filename
        if not path.exists():
            raw = subprocess.check_output(['curl', '--fail', '--silent', '--show-error', '--location',
                                           '--retry', '2', '--max-time', '40', BASE + filename])
            path.write_bytes(raw)
        raw = path.read_bytes()
        values = np.loadtxt(io.BytesIO(raw), ndmin=2)
        if values.shape[1] != 6 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f'Invalid log: {path}')
        durations = np.diff(values[:, 1]) / 1000
        if (durations <= 0).any() or (values[1:, 4] < 0).any():
            raise ValueError(f'Invalid timestamps/bytes: {path}; no imputation allowed')
        sha = hashlib.sha256(raw).hexdigest()
        trace = Trace(f'buffer-ferry-{session}', durations,
                      values[1:, 4] * 8 / (durations * 1000), .08,
                      source_group='buffer-ferry-route',
                      source=f'{BASE + filename}; raw_sha256={sha}; bytes/monotonic timestamp differences; first row discarded; RTT assumed80ms')
        if trace.fingerprint in known_hashes:
            raise ValueError('Additional session duplicates an existing trace')
        traces.append(trace)
        raw_metadata.append({'file': str(path), 'url': BASE + filename, 'raw_sha256': sha,
                             'trace_sha256': trace.fingerprint, 'duration_s': float(durations.sum())})
    dataset = Dataset(original.video, {**original.splits, 'test': tuple(traces)},
                      'Four additional unused Norway ferry sessions, filename order predeclared before replay. '
                      'Same Envivio VOD and ferry route; not new-video or new-route evidence. '
                      'Train/validation retained only for schema compatibility and are not used in this experiment.')
    dataset.save(destination / 'dataset.json')
    (destination / 'provenance.json').write_text(json.dumps({'dataset_sha256': dataset.fingerprint,
                                                          'sessions': raw_metadata}, indent=2) + '\n')
    (destination / 'README.md').write_text(
        'Four unused sessions from the original [Norway HSDPA archive](' + BASE + '). '
        'These are the first four unused ferry sessions in archive filename order. '
        'Dates and buffer profiles were saved before downloads/replay in '
        '`runs/buffer-experiment/protocol.json`. No scaling, imputation or trace repetition. '
        'Source URLs and hashes are in `provenance.json`. Only test split sessions '
        'are replayed; train/validation entries are retained for schema compatibility.\n')
    print(json.dumps({'dataset_sha256': dataset.fingerprint, 'sessions': raw_metadata}), flush=True)


if __name__ == '__main__':
    main()
