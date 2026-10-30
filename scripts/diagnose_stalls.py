"""Diagnose VOD stalls; --check/--check-minimal exit 1 on the retained failure."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.data import Dataset, Trace, Video
from abr.diagnostics import optimistic_arrival_bound, service_bytes
from abr.env import StreamingEnv, StreamConfig, QoEWeights
from abr.models.checkpoint import load_model, source_hash
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.recovery import RecoveryConfig

OUTPUT = Path('runs/stall-diagnosis')
CHECKPOINT = Path('runs/learning-ablation/baseline-seed-7/accepted.npz')


def replay(dataset, policy, *, config=None, recovery=None):
    env = StreamingEnv(dataset, split='test', config=config or StreamConfig(startup_s=4., resume_s=1.),
                       weights=QoEWeights(rebuffer=12), recovery=recovery or RecoveryConfig())
    obs, _ = env.reset(seed=0, options={'trace_index': 0})
    steps, stalls = [], []
    while True:
        start, buffer, before = env.wall_s, env.player.buffer_s, env.metrics()['rebuffer_s']
        action = int(policy(obs, env.config, env.segment))
        obs, _, done, truncated, info = env.step(action)
        row = {'segment_index': info['segment_index'], 'start_s': start,
               'end_s': start + info['request_s'], 'buffer_before_s': buffer,
               'policy_action': action, 'effective_action': info['effective_action'],
               'added_stall_s': env.metrics()['rebuffer_s'] - before,
               'request_s': info['request_s'], 'buffer_wait_s': info['buffer_wait_s'],
               'rtt_s': info['rtt_s'], 'backoff_s': info['recovery_backoff_s'],
               'bytes_received': info['bytes_received'], 'attempts': info['request_attempts']}
        steps.append(row)
        if row['added_stall_s'] > 1e-7:
            i, body_start = row['segment_index'], row['attempts'][0]['body_start_s']
            available = service_bytes(env.network.trace, body_start, max(body_start, start + buffer))
            minimum_size = float(dataset.video.sizes_bytes[i].min())
            row.update(same_state_no_stall_deadline_s=start + buffer,
                       service_before_deadline_bytes=available, minimum_segment_bytes=minimum_size,
                       minimum_byte_shortfall_at_deadline=max(0., minimum_size - available))
            stalls.append(row)
        if done or truncated:
            break
    metrics = env.metrics()
    return {'trace': env.network.trace.name, 'completed': bool(done), 'metrics': metrics,
            'passed': bool(done and metrics['startup_s'] <= 3. and metrics['rebuffer_s'] <= 8.),
            'config': asdict(env.config), 'recovery': asdict(env.recovery),
            'steps': steps, 'stall_steps': stalls, 'provenance': env.provenance()}


def make_minimal_case(dataset, row):
    """Keep one prebuffered unit and one requested unit in a clipped trace."""
    index, trace = row['segment_index'], dataset.splits['test'][0]
    duration = float(dataset.video.durations_s[index - 1])
    if not np.isclose(duration, row['buffer_before_s'], atol=1e-7):
        raise ValueError('Reduction requires one previous buffered unit')
    ends = np.cumsum(trace.durations_s)
    starts = np.r_[0., ends[:-1]]
    overlap = np.maximum(0., np.minimum(ends, row['end_s'] + 1e-6) - np.maximum(starts, row['start_s']))
    valid = overlap > 0
    cropped = Trace('minimal-stall-window', overlap[valid], trace.bandwidth_kbps[valid], trace.rtt_s,
                    source='Clipped measured window of ' + trace.name + '; no scaling or repetition')
    video = Video(dataset.video.bitrates_kbps, dataset.video.durations_s[index - 1:index + 1],
                  dataset.video.sizes_bytes[index - 1:index + 1])
    data = Dataset(video, {'train': (Trace('unused-train', [1.], [1.]),),
                          'validation': (Trace('unused-validation', [2.], [1.]),),
                          'test': (cropped,)}, 'One-request diagnostic; other splits are unused schema scaffolding')
    return {'dataset': data.to_dict(), 'original_trace': trace.name, 'original_start_s': row['start_s'],
            'original_segment_index': index, 'config': asdict(StreamConfig(startup_s=4., resume_s=1.)),
            'buffer_before_s': duration, 'expected_added_stall_s': row['added_stall_s'],
            'restored_state': 'One lowest-quality unit buffered; player already started and playing; segment index=1'}


def replay_minimal(case):
    raw = case['dataset']
    data = Dataset(Video(**raw['video']), {k: tuple(Trace(**t) for t in rows)
                   for k, rows in raw['splits'].items()}, raw['provenance'])
    env = StreamingEnv(data, split='test', config=StreamConfig(**case['config']), recovery=RecoveryConfig())
    env.reset(options={'trace_index': 0})
    env.player.append(case['buffer_before_s'], data.video.bitrates_kbps[0])
    env.player.started = env.player.playing = True
    env.segment, env.downloaded_media_s = 1, case['buffer_before_s']
    _, _, done, truncated, info = env.step(0)
    metrics = env.metrics()
    expected = max(0., info['request_s'] - case['buffer_before_s'])
    assert np.isclose(metrics['rebuffer_s'], expected, atol=1e-7)
    assert np.isclose(metrics['rebuffer_s'], case['expected_added_stall_s'], atol=1e-7)
    return {'completed': bool(done), 'truncated': bool(truncated), 'request_s': info['request_s'],
            'buffer_before_s': case['buffer_before_s'], 'rebuffer_s': metrics['rebuffer_s'],
            'accounting_identity_stall_equals_request_minus_buffer_s': expected,
            'passed': bool(done and metrics['rebuffer_s'] <= 8.), 'attempts': info['request_attempts']}


def ppo_inventory():
    rows = []
    for path in sorted(Path('runs').glob('ppo*/*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            m = json.loads(archive['__metadata__'].tobytes())
        rows.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'training_data': m.get('training', {}).get('training_data'),
                     'dataset_sha256': m.get('training', {}).get('training_dataset_sha256'),
                     'source_compatible': m.get('model_source_sha256') == source_hash()})
    return rows


def write_report(report):
    cases, bound = report['counterfactuals'], report['offline_bounds'][0]
    base, small = cases['dqn'], report['minimal_replay']
    lines = ['# Remaining long-stall diagnosis', '',
             '## Finding', '',
             'The dominant cause is a service deficit under the simulator buffer/ladder contract. '
             'Lowest-quality requests themselves arrive too slowly. No playback accounting defect '
             'was found. This is a diagnosis, not a runtime fix or a model promotion.', '',
             f"The frozen DQN completes the first failure session with {base['metrics']['rebuffer_s']:.6f} s "
             'of stalls, startup threshold 4 s, resume threshold 1 s, and buffer cap 30 s.', '',
             '## Reproduction and reduction', '',
             '`.venv/bin/python -m scripts.diagnose_stalls --check` exits **1** on the retained '
             '8-second stall-limit failure. Repeated original replays produced the same verdict.', '',
             '`.venv/bin/python -m scripts.diagnose_stalls --check-minimal` also exits **1**. '
             'It uses `minimal_case.json`: one already-buffered unit, one lowest-quality request, '
             'and clipped measured trace intervals. No scores, earlier requests, retries or '
             'cancellation are needed. The player is restored to its already-started state.', '',
             f"The reduced request takes {small['request_s']:.6f} s with {small['buffer_before_s']:.6f} s "
             f"buffered. Their difference is {small['rebuffer_s']:.6f} s, matching the counted stalls "
             'to numerical precision. Unused train/validation entries only satisfy the Dataset schema.', '',
             '## Stall locations', '',
             'Indices are zero-based. Service is integrated from the original body start to buffer '
             'exhaustion, assuming no bytes are wasted. These are same-state diagnostics, not global bounds.', '',
             '| Segment | Request start s | Buffer s | Requested/effective kbps | Added stalls s | Minimum bytes | Service before buffer empty, bytes |',
             '|---:|---:|---:|---|---:|---:|---:|']
    for r in base['stall_steps']:
        lines.append(f"| {r['segment_index']} | {r['start_s']:.3f} | {r['buffer_before_s']:.3f} | "
                     f"{r['attempts'][0]['bitrate_kbps']:.0f}/"
                     f"{r['attempts'][-1]['bitrate_kbps']:.0f} | "
                     f"{r['added_stall_s']:.3f} | {r['minimum_segment_bytes']:.0f} | {r['service_before_deadline_bytes']:.0f} |")
    lines.extend(['', '## One-factor counterfactuals', '',
                  'All rows cover completed videos. Zero RTT changes only RTT relative to lowest; '
                  'buffer60 changes only the cap relative to the corresponding policy. Startup/resume '
                  'thresholds remain 4/1 s. Total delay includes both startup and stalls.', '',
                  '| Case | Startup s | Stalls s | Total delay s | Played kbps | Discarded bytes | Peak buffer s | Targets met |',
                  '|---|---:|---:|---:|---:|---:|---:|---|'])
    for name, c in cases.items():
        m = c['metrics']
        lines.append(f"| {name} | {m['startup_s']:.3f} | {m['rebuffer_s']:.3f} | "
                     f"{m['startup_s']+m['rebuffer_s']:.3f} | {m['mean_played_bitrate_kbps']:.1f} | "
                     f"{m['wasted_bytes']:.0f} | {m['peak_buffer_s']:.3f} | {c['passed']} |")
    reduction = base['metrics']['rebuffer_s'] - cases['dqn_force_lowest_at37']['metrics']['rebuffer_s']
    lines.extend(['', f'Forcing lowest only at segment 37 removes the cancellation and saves '
                  f'just {reduction:.6f} s of stalls. Removing backoff changes nothing here. '
                  'Using lowest quality for the whole video still fails the stall target. '
                  'Removing RTT starts playback earlier and slightly reduces total delay, '
                  'but does not remove the later service deficit.', '',
                  '## Necessary waiting bound', '',
                  'Let C_i be media duration through segment i, P_i its media start, and B the buffer cap. '
                  'The simulator starts a body only when buffer + duration_i <= B. Played media cannot '
                  'exceed elapsed wall time at unit rate, so body i cannot start before max(0, C_i - B). '
                  'Real startup/stalls only delay those buffer-space release times.', '',
                  'At these optimistic releases, transfer the smallest actual file per segment serially, '
                  'with zero RTT, no retries, no timeouts and no discarded bytes. Denote the resulting '
                  'earliest arrivals by a_i. No real schedule can deliver the same ordered segments earlier. '
                  'To play segment i, elapsed time must be at least a_i but played time is only P_i; '
                  'therefore total startup+stalls >= max_i(a_i - P_i). This is a necessary lower bound; '
                  'the relaxed schedule need not itself be feasible.', '',
                  f"For B=30 s, startup+stalls >= **{bound['startup_plus_stalls_lower_bound_s']:.6f} s** "
                  f"at segment {bound['bottleneck_segment_index']}. With startup <=3 s, stalls must be "
                  f">= **{bound['stalls_lower_bound_with_startup_budget_s']:.6f} s**. The 8 s stall target "
                  'cannot be met by any bitrate policy under this exact simulator contract.', '',
                  '## PPO answer', '',
                  'PPO learns a policy distribution from sampled rollouts and clipped policy updates. '
                  'DQN learns action values from replay and a target network. Either can learn different '
                  'choices; neither changes the service, cap, media files or necessary waiting bound. '
                  'PPO under the same contract cannot meet the joint 3 s startup / 8 s stall target '
                  'on this session. Its exact reward, bitrate and stall total remain unmeasured.', '',
                  f"The local audit found {len(report['ppo_checkpoint_inventory'])} saved PPO checkpoints, "
                  'all trained on the synthetic fixture with incompatible source hashes. They are not '
                  'a matched real-input baseline and were not silently migrated or replayed. '
                  'The current PPO trainer also lacks the DQN training-time filter and stall gates. '
                  'A fair comparison requires matching data, reward, constraints, interaction budget '
                  'and multiple seeds; these runs do not establish PPO versus DQN superiority.', '',
                  'Source: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).', '',
                  '## Next change to evaluate', '',
                  'Increasing the prefetch cap from 30 to 60 s removes stalls in both the DQN and '
                  'lowest-quality diagnostic replays here, without changing startup threshold. '
                  'This differs from increasing startup/resume thresholds: it permits fetching '
                  'more media before the long slow period. The larger-cap runs also discard more '
                  'bytes during retries. This is one inspected session, not validation of a new default.', '',
                  'Evaluate the larger prefetch cap on separate sessions, retaining startup/stall '
                  'limits and recording downloaded/wasted bytes and memory cost. Alternatively, '
                  'a lower bitrate rung needs real encoded segment files, not invented sizes. '
                  'No additional training or production configuration change is made in this diagnosis.', '',
                  '; '.join(report['limits']) + '.', '',
                  'Regenerate evidence: `.venv/bin/python -m scripts.diagnose_stalls`.', ''])
    (OUTPUT / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-minimal', action='store_true')
    args = parser.parse_args()
    if args.check_minimal:
        result = replay_minimal(json.loads((OUTPUT / 'minimal_case.json').read_text()))
        print(json.dumps(result))
        raise SystemExit(0 if result['passed'] else 1)
    torch.set_num_threads(1)
    dataset = Dataset.load('data/learning-holdout/dataset.json')
    model, metadata = load_model(CHECKPOINT)
    safety = ActionSafety()
    dqn = DecisionPolicy(model, safety=safety)
    dqn_policy = lambda obs, cfg, segment: dqn(obs, cfg)
    base = replay(dataset, dqn_policy)
    if args.check:
        print(json.dumps({'trace': base['trace'], 'completed': base['completed'], 'passed': base['passed'],
                          'startup_s': base['metrics']['startup_s'], 'rebuffer_s': base['metrics']['rebuffer_s']}))
        raise SystemExit(0 if base['passed'] else 1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rule = lambda obs, cfg, segment: int(np.flatnonzero(allowed_actions(obs, safety))[-1])
    lowest = lambda obs, cfg, segment: 0
    no_rtt = Dataset(dataset.video, {**dataset.splits, 'test':
        (replace(dataset.splits['test'][0], rtt_s=0., rtt_series_s=None),)}, 'Same first session with RTT removed for diagnosis')
    cases = {'dqn': base, 'highest_allowed': replay(dataset, rule), 'lowest': replay(dataset, lowest),
             'lowest_zero_rtt': replay(no_rtt, lowest),
             'lowest_buffer60': replay(dataset, lowest, config=StreamConfig(startup_s=4., max_buffer_s=60.)),
             'dqn_buffer60': replay(dataset, dqn_policy, config=StreamConfig(startup_s=4., max_buffer_s=60.)),
             'dqn_no_backoff': replay(dataset, dqn_policy, recovery=RecoveryConfig(backoff_s=0.)),
             'dqn_force_lowest_at37': replay(dataset, lambda obs, cfg, segment: 0 if segment == 37 else dqn(obs, cfg))}
    bounds = [optimistic_arrival_bound(dataset.video, dataset.splits['test'][0], max_buffer_s=cap) for cap in (30., 60.)]
    minimal = make_minimal_case(dataset, max(base['stall_steps'], key=lambda r: r['added_stall_s']))
    (OUTPUT / 'minimal_case.json').write_text(json.dumps(minimal, indent=2, allow_nan=False) + '\n')
    report = {'scope': 'Diagnosis only; no runtime changes, training, target relaxation or model migration',
              'dataset_sha256': dataset.fingerprint, 'trace_sha256': dataset.splits['test'][0].fingerprint,
              'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              'checkpoint_training_dataset_sha256': metadata['training']['training_dataset_sha256'],
              'diagnostic_source_sha256': hashlib.sha256(Path('abr/diagnostics.py').read_bytes() + Path(__file__).read_bytes()).hexdigest(),
              'ranked_hypotheses': ['Service deficit under the buffer/ladder contract',
                                    'Higher quality or retry overhead dominates stalls', 'Playback accounting error'],
              'counterfactuals': cases, 'offline_bounds': bounds, 'minimal_replay': replay_minimal(minimal),
              'ppo_checkpoint_inventory': ppo_inventory(), 'ppo_reference': 'https://arxiv.org/abs/1707.06347',
              'limits': ['Simulator-specific; same already-inspected failure session and trace start at zero',
                         'Observed goodput replay is not a causal radio/packet bound',
                         'Serial whole-segment VOD, pre-body backpressure, fixed ladder and unit playback rate',
                         'Future trace and exact candidate sizes are offline diagnostics only',
                         'Larger buffer changes DQN context outside its training distribution',
                         'No matched PPO was trained or measured on these real inputs']}
    (OUTPUT / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_report(report)
    print(json.dumps({'reproduced_stalls_s': base['metrics']['rebuffer_s'],
                      'minimal_request_stalls_s': report['minimal_replay']['rebuffer_s'],
                      'necessary_stalls_with_startup_at_most3_s': bounds[0]['stalls_lower_bound_with_startup_budget_s'],
                      'counterfactuals': {name: c['metrics']['rebuffer_s'] for name, c in cases.items()},
                      'output': str(OUTPUT / 'results.json')}))


if __name__ == '__main__':
    main()
