"""Fixed frozen-weight comparison of baseline and buffer-aware floor recovery."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.data import Dataset
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety
from abr.buffer_safety import BufferAwareSafety, allowed_actions
from abr.startup_training import make_env

OUT = Path('runs/buffer-aware-recovery')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    original = json.loads(Path('runs/startup-ladder/protocol.json').read_text())
    safety = {'baseline': ActionSafety(min_buffer_s=4.), 'buffer_aware': BufferAwareSafety(min_buffer_s=4.)}
    inputs = [(f'{content}-{split}', f'data/{folder}/{filename}.json', split)
              for content, folder in [('envivio', 'startup-ladder'), ('sintel', 'sintel-ladder')]
              for split, filename in [('validation', 'training'), ('test', 'development')]]
    inputs += [(f'{c}-transfer', f'data/transfer-sessions/{c}.json', 'test') for c in ('envivio', 'sintel')]
    protocol = {'checkpoint': original['checkpoint'], 'checkpoint_sha256': original['checkpoint_sha256'],
                'stream_protocol': original, 'variants': {k: asdict(v) for k, v in safety.items()},
                'inputs': [{'label': label, 'path': p, 'split': split, 'sha256': sha(p)} for label, p, split in inputs],
                'sources': {p: sha(p) for p in ('scripts/evaluate_buffer_recovery.py', 'abr/models/safety.py', 'abr/buffer_safety.py', 'abr/env.py')},
                'gate': 'Diagnostic candidate only. Paired complete, no added stalls or startup delay, '
                        'no added emergency media or lost draft quality passes. Retain bitrate and wasted-byte changes. '
                        'Any gate failure prevents promotion. No checkpoint or default changes.',
                'native_plan': 'If candidate improves a measured recovery case without extra stalls there, '
                               'compare baseline and candidate in Chrome on slow-first test trace 0 and controlled validation trace 2, '
                               'on Envivio and Sintel. Retain negative native outcomes; no automatic production approval.',
                'limits': 'Known measured sessions and synthetic controls; transfer sessions were already used in prior experiments. '
                          'Not a new independent acceptance set. Nominal sizes and observed history only, no future service inputs.'}
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(protocol, indent=2) + '\n'
    if (OUT / 'protocol.json').exists() and (OUT / 'protocol.json').read_text() != text:
        raise ValueError('Existing recovery protocol differs')
    (OUT / 'protocol.json').write_text(text)
    if sha(original['checkpoint']) != original['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    model, _ = load_model(original['checkpoint'])
    scorer = DecisionPolicy(model)
    results = []
    for label, filename, split in inputs:
        dataset = Dataset.load(filename)
        for variant, rule in safety.items():
            for policy in ('dqn', 'highest_allowed'):
                for index in range(len(dataset.splits[split])):
                    env = make_env(dataset, split, original)
                    obs, _ = env.reset(seed=0, options={'trace_index': index})
                    rows = []
                    emergency = streak = longest = reward_sum = 0.
                    while True:
                        baseline_mask = allowed_actions(obs, safety['baseline'])
                        mask = allowed_actions(obs, rule)
                        scores = scorer.scores(obs, env.config)
                        action = int(np.where(mask, scores, -np.inf).argmax()) if policy == 'dqn' else int(np.flatnonzero(mask)[-1])
                        duration = float(dataset.video.durations_s[env.segment])
                        row = {'segment_index': env.segment, 'wall_s': env.wall_s,
                               'buffer_s': float(obs['state'][0]), 'policy_action': action,
                               'baseline_allowed': np.flatnonzero(baseline_mask).tolist(),
                               'allowed': np.flatnonzero(mask).tolist(),
                               'recovery_selected': bool(not baseline_mask[action])}
                        obs, reward, done, truncated, info = env.step(action)
                        reward_sum += reward
                        if info['segment_complete'] and info['effective_action'] == 0:
                            emergency += duration
                            streak += duration
                            longest = max(longest, streak)
                        else:
                            streak = 0.
                        row.update(effective_action=info['effective_action'], request_s=info['request_s'])
                        rows.append(row)
                        if done or truncated: break
                    m = env.metrics()
                    results.append({'dataset': label, 'trace': env.network.trace.name,
                        'trace_sha256': env.network.trace.fingerprint, 'variant': variant, 'policy': policy,
                        'completed': bool(done), 'startup_s': m['startup_s'], 'rebuffer_s': m['rebuffer_s'],
                        'mean_played_bitrate_kbps': m['mean_played_bitrate_kbps'], 'wasted_bytes': m['wasted_bytes'],
                        'reward': reward_sum, 'emergency_media_s': emergency,
                        'emergency_fraction': emergency / env.total_s, 'longest_emergency_s': longest,
                        'playback_targets_met': bool(done and m['startup_s'] <= 3 and m['rebuffer_s'] <= 8),
                        'quality_limits_met': bool(done and emergency / env.total_s <= .1 + 1e-9 and longest <= 12 + 1e-9),
                        'recovery_selections': sum(r['recovery_selected'] for r in rows), 'steps': rows})
        print(json.dumps({'dataset': label, 'rollouts': len(results)}), flush=True)
    comparisons = []
    for row in results:
        if row['variant'] == 'baseline': continue
        base = next(b for b in results if b['variant'] == 'baseline' and
                    all(b[k] == row[k] for k in ('dataset', 'trace', 'policy')))
        playback = bool(row['completed'] and row['rebuffer_s'] <= base['rebuffer_s'] + 1e-8 and row['startup_s'] <= base['startup_s'] + 1e-8)
        quality = bool(row['emergency_media_s'] <= base['emergency_media_s'] + 1e-8 and
                       (not base['quality_limits_met'] or row['quality_limits_met']))
        comparisons.append({'dataset': row['dataset'], 'trace': row['trace'], 'policy': row['policy'],
            'emergency_reduction_s': base['emergency_media_s'] - row['emergency_media_s'],
            'stall_delta_s': row['rebuffer_s'] - base['rebuffer_s'], 'startup_delta_s': row['startup_s'] - base['startup_s'],
            'bitrate_delta_kbps': row['mean_played_bitrate_kbps'] - base['mean_played_bitrate_kbps'],
            'wasted_bytes_delta': row['wasted_bytes'] - base['wasted_bytes'],
            'playback_gate': playback, 'quality_gate': quality, 'passed': playback and quality})
    groups = []
    for variant in safety:
        for policy in ('dqn', 'highest_allowed'):
            group = [r for r in results if r['variant'] == variant and r['policy'] == policy]
            groups.append({'variant': variant, 'policy': policy, 'sessions': len(group),
                'completed': sum(r['completed'] for r in group), 'targets_met': sum(r['playback_targets_met'] for r in group),
                'quality_limits_met': sum(r['quality_limits_met'] for r in group),
                'emergency_media_s': sum(r['emergency_media_s'] for r in group),
                'rebuffer_s': sum(r['rebuffer_s'] for r in group), 'wasted_bytes': sum(r['wasted_bytes'] for r in group)})
    payload = {'protocol_sha256': sha(OUT / 'protocol.json'), 'results': results, 'comparisons': comparisons,
               'summary': groups, 'all_paired_gates_passed': all(c['passed'] for c in comparisons),
               'native_confirmation_warranted': any(c['dataset']=='envivio-test' and c['emergency_reduction_s']>0 and c['playback_gate'] for c in comparisons),
               'defaults_changed': False, 'production_approved': False}
    (OUT / 'simulator.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': groups, 'all_paired_gates_passed': payload['all_paired_gates_passed']}), flush=True)


if __name__ == '__main__': main()
