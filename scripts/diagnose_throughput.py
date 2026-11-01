"""Audit observed throughput history; future service is never a policy input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from abr.models.safety import ActionSafety, allowed_actions

OUT = Path('runs/throughput-diagnosis')


def window_observation(observation, window):
    """Change only the safety filter's valid history; keep model input intact."""
    history = observation['history'].copy()
    valid = np.flatnonzero(history[:, -1] > 0)
    history[valid[:-window], -1] = 0
    return {**observation, 'history': history}


def estimate_kbps(observation, safety, window=8):
    history = window_observation(observation, window)['history']
    rates = history[history[:, -1] > 0, 0]
    return float(safety.throughput_factor * len(rates) / np.sum(1 / np.maximum(rates, 1e-12)) * 1000) if len(rates) else None


def native_audit():
    results = []
    for content in ('envivio', 'sintel'):
        for path in sorted((Path('runs/native-quality-gate') / content).glob('*s.json')):
            capture = json.loads(path.read_text())
            context = capture['context']
            safety = ActionSafety(**context['safety_filter'])
            history = np.zeros((8, 5), dtype=np.float32)
            rows = []
            for segment, decision in zip(capture['segments'], capture['transport_audit']['decisions'], strict=True):
                index = segment['segment_index']
                duration = context['durations_s'][index]
                ladder = np.array([r['bitrate_kbps'] for r in context['representations']])
                observation = {'history': history, 'state': np.array(decision['state'], dtype=np.float32),
                               'candidates': np.column_stack((ladder / 1000, ladder * duration / 8000,
                                                            np.full(len(ladder), duration))).astype(np.float32)}
                original = allowed_actions(observation, safety)
                if np.flatnonzero(original).tolist() != decision['allowed_actions']:
                    raise ValueError(f'Could not reproduce actual filter mask in {path}, segment {index}')
                floor = int(np.flatnonzero(ladder >= 300)[0])
                masks = {w: allowed_actions(window_observation(observation, w), safety) for w in (8, 3, 1)}
                rates = history[history[:, -1] > 0, 0]
                rows.append({'segment_index': index, 'buffer_s': float(observation['state'][0]),
                             'action': segment['action'], 'history_kbps': (rates * 1000).tolist(),
                             'estimates_kbps': {str(w): estimate_kbps(observation, safety, w) for w in (8, 3, 1)},
                             'allowed_actions': {str(w): np.flatnonzero(masks[w]).tolist() for w in masks},
                             'floor_blocked_with_8_but_allowed_with_3': bool(not masks[8][floor] and masks[3][floor]),
                             'floor_blocked_even_with_latest_rate': bool(not masks[1][floor]),
                             'received_at_s': segment['received_at_s']})
                received = sum(a['bytes_received'] for a in segment['attempts'])
                history = np.roll(history, -1, axis=0)
                history[-1] = [received * 8 / segment['body_s'] / 1e6, 0, 0, 0, 1]
            results.append({'capture': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'content': content, 'case': path.stem, 'safety': context['safety_filter'],
                            'all_actual_masks_reproduced': True, 'decisions': rows})
    return results


def run_replays():
    """Vary only safety history length; the neural model still receives all 8 rows."""
    import torch
    from abr.data import Dataset
    from abr.models.checkpoint import load_model
    from abr.models.policy import DecisionPolicy
    from abr.startup_training import make_env
    from scripts.diagnose_quality_gate import probe_upgrade

    torch.set_num_threads(1)
    original = json.loads(Path('runs/startup-ladder/protocol.json').read_text())
    checkpoint = Path(original['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    inputs = [('envivio-validation', 'data/startup-ladder/training.json', 'validation'),
              ('envivio-development', 'data/startup-ladder/development.json', 'test'),
              ('sintel-development', 'data/sintel-ladder/development.json', 'test')]
    protocol = {'question': 'Does eight-sample harmonic history delay quality recovery?',
                'checkpoint_sha256': original['checkpoint_sha256'], 'stream_protocol': original,
                'fixed_min_buffer_s': 4., 'fixed_throughput_factor': .65,
                'safety_windows': [8, 3, 1], 'model_history_length': 8,
                'policies': ['dqn', 'highest_allowed'],
                'inputs': [{'label': label, 'path': path, 'split': split,
                            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for label, path, split in inputs],
                'sources': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                            ('scripts/diagnose_throughput.py', 'abr/models/safety.py', 'abr/env.py')},
                'success': 'Paired complete replay, no extra stalls, no worse startup, less emergency media. '
                           'Retain negative outcomes and bitrate/wasted-byte changes. Diagnostic only; no promotion.',
                'limits': 'Known development/validation inputs, simulator only. Offline one-request probes use '
                          'future service only in a deep-copy diagnostic, never to choose policy actions.'}
    text = json.dumps(protocol, indent=2) + '\n'
    path = OUT / 'protocol.json'
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing throughput protocol differs')
    path.write_text(text)
    model, _ = load_model(checkpoint)
    scorer = DecisionPolicy(model)
    safety = ActionSafety(min_buffer_s=4., throughput_factor=.65)
    results = []
    for label, filename, split in inputs:
        dataset = Dataset.load(filename)
        for window in (8, 3, 1):
            for policy in ('dqn', 'highest_allowed'):
                for index in range(len(dataset.splits[split])):
                    env = make_env(dataset, split, original)
                    obs, _ = env.reset(seed=0, options={'trace_index': index})
                    sample_times = []
                    rows = []
                    emergency = streak = longest = reward_sum = 0.
                    while True:
                        mask = allowed_actions(window_observation(obs, window), safety)
                        scores = scorer.scores(obs, env.config)
                        action = int(np.where(mask, scores, -np.inf).argmax()) if policy == 'dqn' else int(np.flatnonzero(mask)[-1])
                        segment = env.segment
                        duration = float(dataset.video.durations_s[segment])
                        record = {'segment_index': segment, 'wall_s': env.wall_s,
                                  'buffer_s': float(obs['state'][0]),
                                  'history_kbps': (obs['history'][obs['history'][:, -1] > 0, 0] * 1000).tolist(),
                                  'sample_ages_s': [env.wall_s - t for t in sample_times[-8:]],
                                  'estimates_kbps': {str(w): estimate_kbps(obs, safety, w) for w in (8, 3, 1)},
                                  'allowed_actions': np.flatnonzero(mask).tolist(), 'policy_action': action}
                        if window == 8 and policy == 'dqn' and action == 0:
                            record['floor_probe'] = probe_upgrade(env, 1)
                        obs, reward, done, truncated, info = env.step(action)
                        reward_sum += reward
                        record.update(effective_action=info['effective_action'],
                                      buffer_wait_s=info['buffer_wait_s'], request_s=info['request_s'])
                        if info['segment_complete']:
                            sample_times.append(env.wall_s - info['playback_tail_s'])
                        if info['segment_complete'] and info['effective_action'] == 0:
                            emergency += duration
                            streak += duration
                            longest = max(longest, streak)
                        else:
                            streak = 0.
                        rows.append(record)
                        if done or truncated:
                            break
                    m = env.metrics()
                    results.append({'dataset': label, 'trace': env.network.trace.name,
                        'trace_sha256': env.network.trace.fingerprint, 'window': window, 'policy': policy,
                        'completed': bool(done), 'startup_s': m['startup_s'], 'rebuffer_s': m['rebuffer_s'],
                        'mean_played_bitrate_kbps': m['mean_played_bitrate_kbps'],
                        'wasted_bytes': m['wasted_bytes'], 'reward': reward_sum,
                        'emergency_media_s': emergency, 'emergency_fraction': emergency / env.total_s,
                        'longest_emergency_s': longest,
                        'playback_targets_met': bool(done and m['startup_s'] <= 3 and m['rebuffer_s'] <= 8),
                        'quality_limits_met': bool(done and emergency / env.total_s <= .1 + 1e-9 and longest <= 12 + 1e-9),
                        'steps': rows if window == 8 and policy == 'dqn' else None})
            print(json.dumps({'dataset': label, 'window': window, 'finished': len(results)}), flush=True)
    comparisons = []
    for row in results:
        if row['window'] == 8:
            continue
        base = next(b for b in results if b['window'] == 8 and
                    all(b[k] == row[k] for k in ('dataset', 'trace', 'policy')))
        comparisons.append({'dataset': row['dataset'], 'trace': row['trace'], 'policy': row['policy'],
            'window': row['window'], 'emergency_reduction_s': base['emergency_media_s'] - row['emergency_media_s'],
            'stall_delta_s': row['rebuffer_s'] - base['rebuffer_s'],
            'startup_delta_s': row['startup_s'] - base['startup_s'],
            'bitrate_delta_kbps': row['mean_played_bitrate_kbps'] - base['mean_played_bitrate_kbps'],
            'wasted_bytes_delta': row['wasted_bytes'] - base['wasted_bytes'],
            'no_playback_regression': bool(row['completed'] and row['rebuffer_s'] <= base['rebuffer_s'] + 1e-8 and
                                           row['startup_s'] <= base['startup_s'] + 1e-8)})
    summary = []
    for window in (8, 3, 1):
        for policy in ('dqn', 'highest_allowed'):
            group = [r for r in results if r['window'] == window and r['policy'] == policy]
            paired = [r for r in comparisons if r['window'] == window and r['policy'] == policy]
            summary.append({'window': window, 'policy': policy, 'sessions': len(group),
                'complete': sum(r['completed'] for r in group),
                'playback_targets_met': sum(r['playback_targets_met'] for r in group),
                'quality_limits_met': sum(r['quality_limits_met'] for r in group),
                'emergency_media_s': sum(r['emergency_media_s'] for r in group),
                'rebuffer_s': sum(r['rebuffer_s'] for r in group),
                'wasted_bytes': sum(r['wasted_bytes'] for r in group),
                'paired_no_regression': all(r['no_playback_regression'] for r in paired) if paired else None})
    payload = {'summary': summary, 'results': results, 'comparisons': comparisons,
               'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'runtime_defaults_changed': False, 'production_approved': False}
    (OUT / 'replays.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-stale', action='store_true', help='Exit 1 if captured 8-vs-3 sample lockouts remain; diagnostic, not a production acceptance test')
    parser.add_argument('--replay', action='store_true', help='Run frozen-model 8/3/1 sample safety-history ablation at fixed 4s buffer threshold')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = native_audit()
    count = sum(r['floor_blocked_with_8_but_allowed_with_3'] for c in cases for r in c['decisions'])
    payload = {'cases': cases, 'all_actual_masks_reproduced': True,
               'captured_8_vs_3_floor_lockouts': count,
               'scope': 'Observed measurements only; shorter-window permission does not prove a safe download',
               'runtime_defaults_changed': False}
    (OUT / 'native-audit.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'captured_8_vs_3_floor_lockouts': count, 'captures': len(cases), 'all_actual_masks_reproduced': True}))
    if args.replay:
        run_replays()
    if args.check_stale and count:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
