"""Frozen 30/60-second prefetch comparison. --prepare fetches fixed new sessions."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from abr.benchmark import evaluate
from abr.data import Dataset, Trace
from abr.env import StreamConfig, StreamingEnv, QoEWeights
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.recovery import RecoveryConfig
from scripts.evaluate_request_recovery import compact

OUT = Path('runs/prefetch-validation')
DATA = Path('data/prefetch-sessions')
SESSIONS = ('2011-01-29_1800CET', '2011-01-31_1830CET', '2011-02-01_0629CET', '2011-02-01_0740CET')
BASE = 'https://datasets.simula.no/downloads/hsdpa-tcp-logs/ferry.nesoddtangen-oslo/'
CHECKPOINT = Path('runs/learning-ablation/baseline-seed-7/accepted.npz')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    protocol = {'sessions': list(SESSIONS), 'selection': 'Next four unused ferry archive filenames after the prior four; chosen before trace values or replay',
                'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'caps_s': [30., 60.], 'startup_s': 1., 'resume_s': 1., 'recovery': asdict(RecoveryConfig()),
                'policies': ['lowest', 'highest_allowed', 'dqn'], 'safety_filter': asdict(ActionSafety()),
                'targets': {'complete': True, 'max_startup_s': 3., 'max_rebuffer_s': 8.},
                'native_cases': [{'label': 'new-first', 'trace_index': 0}, {'label': 'known-hard', 'trace_index': 4}],
                'native_policies': ['dqn', 'highest_allowed'], 'native_wall_limit_s': 240,
                'native_recovery': False,
                'memory': 'Sample whole isolated Chrome process-tree RSS every 2 s, JS heap separately. '
                          'Simulator also reports estimated compressed-media backlog; it is not browser memory.',
                'training_gate': 'Do not train unless 60 s passes every new simulator and native timing/completion gate. '
                                 'Memory suitability still needs an explicit device budget; no budget is assumed.',
                'training_after_gate': {'seeds': [7, 11, 13], 'episodes_per_seed': 120,
                                       'baseline': 'same highest-allowed rule and recovery/configuration'},
                'limitations': ['Same silent VOD and ferry route; historical observed-goodput traces with assumed80ms RTT',
                                'Frozen model trained at 30 s cap; 60 s is an untrained context',
                                'Native player has no recovery; raw partial runs must remain marked partial',
                                'Browser RSS includes runtime/codecs; process-tree sums can count shared pages more than once',
                                'No target-device memory budget supplied; measurement is not a deployment approval']}
    path = OUT / 'protocol.json'
    text = json.dumps(protocol, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing fixed protocol differs')
    path.write_text(text)  # Freeze settings before downloads or inspecting trace values.
    original = Dataset.load('data/hard_envivio/dataset.json')
    known = [Dataset.load(p) for p in ('data/hard_envivio/dataset.json', 'data/learning-holdout/dataset.json',
                                      'data/buffer-sessions/dataset.json', 'data/realistic_envivio/dataset.json')]
    fingerprints = {t.fingerprint for d in known for ts in d.splits.values() for t in ts}
    traces, provenance = [], []
    for session in SESSIONS:
        path = DATA / f'report.{session}.log'
        url = BASE + path.name
        if not path.exists():
            path.write_bytes(subprocess.check_output(['curl', '--fail', '--silent', '--show-error', '--location',
                                                     '--retry', '2', '--max-time', '40', url]))
        raw = path.read_bytes()
        values = np.loadtxt(io.BytesIO(raw), ndmin=2)
        if values.shape[1] != 6 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('Invalid raw log')
        durations = np.diff(values[:, 1]) / 1000
        if (durations <= 0).any() or (values[1:, 4] < 0).any():
            raise ValueError('Invalid times/bytes; no imputation')
        sha = hashlib.sha256(raw).hexdigest()
        trace = Trace(f'prefetch-ferry-{session}', durations, values[1:, 4] * 8 / (durations * 1000), .08,
                      source_group='prefetch-ferry-route', source=f'{url}; raw_sha256={sha}; '
                      'bytes/monotonic timestamp differences; first row discarded; RTT assumed80ms')
        if trace.fingerprint in fingerprints:
            raise ValueError('Session already used')
        traces.append(trace)
        provenance.append({'url': url, 'file': str(path), 'raw_sha256': sha, 'trace_sha256': trace.fingerprint,
                           'duration_s': float(durations.sum())})
    # A fifth test entry is an explicitly labelled, already-inspected stress control.
    hard = Dataset.load('data/learning-holdout/dataset.json').splits['test'][0]
    data = Dataset(original.video, {**original.splits, 'test': tuple(traces) + (hard,)},
                   'Four predeclared unused ferry sessions plus one known failure control; only test split is used. '
                   'Train/validation retained for schema compatibility; same VOD.')
    data.save(DATA / 'dataset.json')
    (DATA / 'provenance.json').write_text(json.dumps({'dataset_sha256': data.fingerprint, 'new_sessions': provenance,
                                                    'known_control': hard.name}, indent=2) + '\n')
    (DATA / 'README.md').write_text('Four additional raw Norway ferry sessions selected in filename order before replay. '
                                  'A fifth test trace is the already-inspected hard control. Raw source URLs and hashes '
                                  'are in provenance.json; experiment settings are in runs/prefetch-validation/protocol.json.\n')
    print(json.dumps(provenance), flush=True)


class PayloadMeter:
    """Offline estimate of encoded backlog; does not allocate fake video memory."""
    def __init__(self, env):
        self.peak_whole_bytes = self.peak_proportional_bytes = 0.
        ends = np.cumsum(env.video.durations_s)
        appended = []
        original = env.player.append
        def append(duration, bitrate):
            original(duration, bitrate)
            index = env.segment
            action = int(np.flatnonzero(env.video.bitrates_kbps == bitrate)[0])
            appended.append((float(ends[index]), float(duration), float(env.video.sizes_bytes[index, action])))
            remaining = [(min(d, max(0., end - env.player.playhead_s)), d, size) for end, d, size in appended]
            whole = sum(size for left, _, size in remaining if left > 1e-9)
            proportional = sum(size * left / duration for left, duration, size in remaining)
            self.peak_whole_bytes = max(self.peak_whole_bytes, whole)
            self.peak_proportional_bytes = max(self.peak_proportional_bytes, proportional)
        env.player.append = append


class MeteredEnv(StreamingEnv):
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.payload_meter = PayloadMeter(self)
        return result

    def metrics(self):
        metrics = super().metrics()
        metrics.update(estimated_peak_encoded_whole_segments_bytes=self.payload_meter.peak_whole_bytes,
                       estimated_peak_encoded_proportional_bytes=self.payload_meter.peak_proportional_bytes)
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    torch.set_num_threads(1)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    data = Dataset.load(DATA / 'dataset.json')
    model, _ = load_model(CHECKPOINT)
    safety = ActionSafety(**protocol['safety_filter'])
    dqn = DecisionPolicy(model, safety=safety)
    results = []
    for cap in protocol['caps_s']:
        for name in protocol['policies']:
            env = MeteredEnv(data, split='test', config=StreamConfig(max_buffer_s=cap),
                             recovery=RecoveryConfig(**protocol['recovery']), weights=QoEWeights(rebuffer=12))
            kwargs = {'baseline': 'lowest'} if name == 'lowest' else {
                'baseline': 'model', 'policy': dqn if name == 'dqn' else
                lambda obs, cfg: int(np.flatnonzero(allowed_actions(obs, safety))[-1]),
                'policy_metadata': {'kind': name, 'frozen_checkpoint_sha256': protocol['checkpoint_sha256']}}
            report = compact(evaluate(env, **kwargs))
            report.update(cap_s=cap, policy=name)
            results.append(report)
            print(json.dumps({'cap_s': cap, 'policy': name,
                              'completed': [e['completed'] for e in report['episodes']],
                              'stalls_s': [round(e['metrics']['rebuffer_s'], 3) for e in report['episodes']],
                              'targets': [e['playback_targets_met'] for e in report['episodes']]}), flush=True)
    payload = {'protocol': protocol, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'results': results}
    (OUT / 'simulator.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
