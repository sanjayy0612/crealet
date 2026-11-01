"""Frozen-policy diagnosis of emergency quality; future service is audit-only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.data import Dataset
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.startup_training import make_env


OUT = Path('runs/quality-gate-diagnosis')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_upgrade(env, action):
    """One real-size request from identical state; never changes the policy run."""
    branch = deepcopy(env)
    before = branch.metrics()
    size = float(branch.video.sizes_bytes[branch.segment, action])
    _, _, _, truncated, info = branch.step(action)
    after = branch.metrics()
    extra_stall = after['rebuffer_s'] - before['rebuffer_s']
    delivered = info['segment_complete'] and info['effective_action'] == action
    return {'actual_segment_bytes': size, 'request_s': info['request_s'],
            'effective_action': info['effective_action'], 'delivered_requested_quality': bool(delivered),
            'added_stall_s': extra_stall, 'startup_s': after['startup_s'],
            'locally_safe': bool(delivered and not truncated and extra_stall <= 1e-8 and
                                 (before['played_s'] > 0 or after['startup_s'] <= 3. + 1e-8)),
            'scope': 'Single-request counterfactual only; not a safe full-session policy or future-service input'}


def rollout(dataset, split, index, protocol, model, threshold, policy_name, audit):
    env = make_env(dataset, split, protocol)
    obs, _ = env.reset(seed=0, options={'trace_index': index})
    scorer = DecisionPolicy(model)
    safety = ActionSafety(min_buffer_s=threshold, throughput_factor=protocol['safety']['throughput_factor'])
    rows = []
    emergency = streak = longest = total_reward = 0.
    while True:
        mask = allowed_actions(obs, safety)
        scores = scorer.scores(obs, env.config)
        raw = int(scores.argmax())
        action = int(np.where(mask, scores, -np.inf).argmax()) if policy_name == 'dqn' else int(np.flatnonzero(mask)[-1])
        valid = obs['history'][obs['history'][:, -1] > 0, 0]
        estimate = float(safety.throughput_factor * len(valid) / np.sum(1 / np.maximum(valid, 1e-12))) if len(valid) else None
        floor = int(np.flatnonzero(dataset.video.bitrates_kbps >= 300)[0])
        # Threshold-only attribution: would the same throughput estimate permit 300 kbps?
        no_buffer_mask = allowed_actions(obs, ActionSafety(min_buffer_s=0., throughput_factor=safety.throughput_factor))
        row = {'segment_index': env.segment, 'wall_s': env.wall_s,
               'buffer_before_s': float(obs['state'][0]), 'scores': scores.tolist(),
               'raw_model_action': raw, 'allowed_actions': np.flatnonzero(mask).tolist(),
               'policy_action': action, 'normal_floor_allowed': bool(mask[floor]),
               'normal_floor_blocked_by_buffer_only': bool(not mask[floor] and no_buffer_mask[floor]),
               'normal_floor_blocked_by_throughput': bool(not no_buffer_mask[floor]),
               'estimated_safe_mbps': estimate, 'history_samples': len(valid)}
        if audit and action == 0:
            row['normal_floor_counterfactual'] = probe_upgrade(env, floor)
        duration = float(dataset.video.durations_s[env.segment])
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        effective = info['effective_action']
        if info['segment_complete'] and dataset.video.bitrates_kbps[effective] < 300:
            emergency += duration
            streak += duration
            longest = max(longest, streak)
        else:
            streak = 0.
        row.update(effective_action=effective, segment_complete=info['segment_complete'])
        rows.append(row)
        if done or truncated:
            break
    metrics = env.metrics()
    return {'trace': env.network.trace.name, 'trace_sha256': env.network.trace.fingerprint,
            'policy': policy_name, 'min_buffer_s': threshold, 'completed': bool(done),
            'startup_s': metrics['startup_s'], 'rebuffer_s': metrics['rebuffer_s'],
            'mean_played_bitrate_kbps': metrics['mean_played_bitrate_kbps'], 'reward': total_reward,
            'emergency_media_s': emergency, 'emergency_fraction': emergency / env.total_s,
            'longest_emergency_s': longest, 'wasted_bytes': metrics['wasted_bytes'],
            'playback_targets_met': bool(done and metrics['startup_s'] <= 3. and metrics['rebuffer_s'] <= 8.),
            'quality_limits_met': bool(done and emergency / env.total_s <= .1 + 1e-9 and longest <= 12. + 1e-9),
            'steps': rows if audit else None}


def main():
    torch.set_num_threads(1)
    original = json.loads(Path('runs/startup-ladder/protocol.json').read_text())
    datasets = [('envivio-validation', 'data/startup-ladder/training.json', 'validation'),
                ('envivio-development', 'data/startup-ladder/development.json', 'test'),
                ('sintel-development', 'data/sintel-ladder/development.json', 'test')]
    protocol = {'hypothesis': 'The 12-second buffer gate may block stall-free upgrades to the normal floor',
                'checkpoint': original['checkpoint'], 'checkpoint_sha256': digest(original['checkpoint']),
                'stream_protocol': original, 'thresholds_s': [12., 4., 0.],
                'policies': ['dqn', 'highest_allowed'],
                'datasets': [{'label': label, 'path': path, 'split': split, 'sha256': digest(path)} for label, path, split in datasets],
                'acceptance': 'Diagnostic only. Compare each changed rollout with the matching 12s policy: complete, no extra stalls, no worse startup, less emergency exposure. No defaults or checkpoint promotion.',
                'limits': ['Known development/validation cases, not an untouched acceptance set',
                           'Simulator only, not a repeat native-browser measurement',
                           'Single-step future-service probes are offline diagnostics, never policy inputs',
                           'Only min_buffer_s changes; throughput factor, weights, recovery and cap stay fixed']}
    OUT.mkdir(parents=True, exist_ok=True)
    # Persist all interventions and inputs before scoring. Do not silently rewrite a protocol.
    text = json.dumps(protocol, indent=2) + '\n'
    target = OUT / 'protocol.json'
    if target.exists() and target.read_text() != text:
        raise ValueError('Existing diagnostic protocol differs')
    target.write_text(text)
    model, _ = load_model(original['checkpoint'])
    if digest(original['checkpoint']) != original['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint no longer matches the original protocol')
    results = []
    for label, path, split in datasets:
        dataset = Dataset.load(path)
        for threshold in protocol['thresholds_s']:
            for policy in protocol['policies']:
                for index in range(len(dataset.splits[split])):
                    result = rollout(dataset, split, index, original, model, threshold, policy,
                                     audit=threshold == 12. and policy == 'dqn')
                    result['dataset'] = label
                    results.append(result)
            print(json.dumps({'dataset': label, 'threshold_s': threshold, 'finished': len(results)}), flush=True)
    comparisons = []
    for row in results:
        if row['min_buffer_s'] == 12.:
            continue
        base = next(x for x in results if x['dataset'] == row['dataset'] and x['trace'] == row['trace'] and
                    x['policy'] == row['policy'] and x['min_buffer_s'] == 12.)
        comparisons.append({'dataset': row['dataset'], 'trace': row['trace'], 'policy': row['policy'],
                            'min_buffer_s': row['min_buffer_s'],
                            'emergency_reduction_s': base['emergency_media_s'] - row['emergency_media_s'],
                            'stall_delta_s': row['rebuffer_s'] - base['rebuffer_s'],
                            'startup_delta_s': row['startup_s'] - base['startup_s'],
                            'no_regression': bool(row['completed'] and row['rebuffer_s'] <= base['rebuffer_s'] + 1e-8 and
                                                  row['startup_s'] <= base['startup_s'] + 1e-8)})
    payload = {'protocol_sha256': digest(target), 'source_sha256': digest(__file__),
               'results': results, 'comparisons': comparisons, 'runtime_defaults_changed': False}
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    decisions = [s for r in results if r['steps'] for s in r['steps'] if s['policy_action'] == 0]
    summary = {
        'rollouts': len(results),
        'video_session_combinations': len({(r['dataset'], r['trace']) for r in results}),
        'baseline_emergency_decisions': len(decisions),
        'model_preferred_upgrade': sum(s['raw_model_action'] > 0 for s in decisions),
        'floor_blocked_by_buffer_only': sum(s['normal_floor_blocked_by_buffer_only'] for s in decisions),
        'floor_blocked_by_throughput_estimate': sum(s['normal_floor_blocked_by_throughput'] for s in decisions),
        'single_request_safe_floor_probes': sum(s['normal_floor_counterfactual']['locally_safe'] for s in decisions),
        'safe_floor_probes_blocked_by_buffer_only': sum(s['normal_floor_blocked_by_buffer_only'] and
            s['normal_floor_counterfactual']['locally_safe'] for s in decisions),
        'groups': [], 'runtime_defaults_changed': False, 'production_approved': False}
    for threshold in protocol['thresholds_s']:
        for policy in protocol['policies']:
            group = [r for r in results if r['policy'] == policy and r['min_buffer_s'] == threshold]
            paired = [r for r in comparisons if r['policy'] == policy and r['min_buffer_s'] == threshold]
            summary['groups'].append({'min_buffer_s': threshold, 'policy': policy, 'sessions': len(group),
                'completed': sum(r['completed'] for r in group),
                'playback_targets_met': sum(r['playback_targets_met'] for r in group),
                'quality_limits_met': sum(r['quality_limits_met'] for r in group),
                'emergency_media_s': sum(r['emergency_media_s'] for r in group),
                'rebuffer_s': sum(r['rebuffer_s'] for r in group),
                'paired_no_regression': all(r['no_regression'] for r in paired) if paired else None})
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
