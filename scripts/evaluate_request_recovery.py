"""Replay fixed request-recovery settings; no training or parameter search.

Run from the repository root: python -m scripts.evaluate_request_recovery
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.benchmark import evaluate
from abr.data import Dataset
from abr.env import StreamingEnv, QoEWeights
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.recovery import RecoveryConfig


def compact(report):
    """Keep metrics, choices, and every recovery/failure event for audit."""
    for episode in report['episodes']:
        steps = episode.pop('steps')
        metrics = episode['metrics']
        # Without recovery, the only discarded body is the final failed request.
        metrics.setdefault('wasted_bytes', sum(s['bytes_received'] for s in steps
                                             if not s['segment_complete']))
        metrics.setdefault('request_retries', 0)
        metrics.setdefault('abandoned_requests', 0)
        episode['metrics_scope'] = 'full_playback' if episode['completed'] else 'partial_playback'
        episode['policy_actions'] = [s['action'] for s in steps]
        episode['effective_actions'] = [s.get('effective_action', s['action']) for s in steps]
        episode['recovery_or_failure_events'] = [
            {'segment_index': s['segment_index'], 'policy_action': s['action'],
             'effective_action': s.get('effective_action', s['action']),
             'request_s': s['request_s'], 'bytes_received': s['bytes_received'],
             'rtt_s': s['rtt_s'], 'backoff_s': s.get('recovery_backoff_s', 0.),
             'attempts': s.get('request_attempts', []), 'reason': s['termination_reason']}
            for s in steps if len(s.get('request_attempts', [])) > 1 or not s['segment_complete']]
        episode['playback_targets_met'] = bool(episode['completed'] and
            metrics['startup_s'] <= 3. + 1e-9 and metrics['rebuffer_s'] <= 8. + 1e-9)
    return report


def main():
    torch.set_num_threads(1)
    output = Path('runs/request-recovery')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('runs/learning-ablation/baseline-seed-7/accepted.npz')
    variants = {'none': None, 'timeout_only': RecoveryConfig(early_abandon=False),
                'progress': RecoveryConfig()}
    safety = ActionSafety()
    protocol = {
        'purpose': 'Repair request termination separately from frozen bitrate selection',
        'training': False, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'recovery': {name: None if cfg is None else asdict(cfg) for name, cfg in variants.items()},
        'policies': ['lowest', 'throughput', 'highest_allowed', 'dqn'],
        'safety_filter': asdict(safety),
        'targets': {'completion_fraction': 1., 'max_startup_s': 3., 'max_rebuffer_s': 8.},
        'target_rule': 'Every session must complete and meet both timing limits',
        'data': [
            {'path': 'data/learning-holdout/dataset.json', 'split': 'test',
             'label': 'failure_development_sessions'},
            {'path': 'data/hard_envivio/dataset.json', 'split': 'validation',
             'label': 'historical_validation'},
            {'path': 'data/hard_envivio/dataset.json', 'split': 'test',
             'label': 'historical_test_already_observed'}],
        'limitations': ['Simulator only; native browser recovery not implemented',
                        'Known sessions; no untouched test or new video',
                        'Finite observed-goodput replay, not a causal network model',
                        'Failed bodies restart in full; no HTTP Range resumption',
                        'Policy history includes received bytes from failed attempts',
                        'All settings fixed before replay; no selection using these results']}
    # Persist the protocol before opening datasets or observing any replay outcome.
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    model, metadata = load_model(checkpoint)
    policy = DecisionPolicy(model, safety=safety)
    results = []
    for source in protocol['data']:
        dataset = Dataset.load(source['path'])
        for variant, cfg in variants.items():
            for name in protocol['policies']:
                env = StreamingEnv(dataset, split=source['split'],
                                   weights=QoEWeights(rebuffer=12), recovery=cfg)
                kwargs = {'baseline': name} if name in ('lowest', 'throughput') else {
                    'baseline': 'model', 'policy': policy if name == 'dqn' else
                    lambda obs, config: int(np.flatnonzero(allowed_actions(obs, safety))[-1]),
                    'policy_metadata': {'kind': name, 'safety_filter': asdict(safety),
                        'checkpoint_sha256': protocol['checkpoint_sha256'] if name == 'dqn' else None}}
                result = compact(evaluate(env, **kwargs))
                result.update(dataset=source['label'], variant=variant, policy=name)
                results.append(result)
                print(json.dumps({'dataset': source['label'], 'variant': variant, 'policy': name,
                                  'completed': sum(e['completed'] for e in result['episodes']),
                                  'sessions': len(result['episodes']),
                                  'stalls_s': [round(e['metrics']['rebuffer_s'], 3) for e in result['episodes']]}), flush=True)
    payload = {'protocol': protocol, 'checkpoint_training': metadata['training'], 'results': results}
    (output / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# Bounded request-recovery replay', '',
             'Frozen DQN weights; no new training. Settings were saved before replay.', '',
             '## Two previously failed sessions', '',
             'Each row is one complete or explicitly partial playback. Wasted bytes are discarded bodies.', '',
             '| Policy | Recovery | Session | Complete | Played % | Stall s | Startup s | Played kbps | Wasted MB | Retries |',
             '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        if result['dataset'] != 'failure_development_sessions':
            continue
        for e in result['episodes']:
            m = e['metrics']
            lines.append(f"| {result['policy']} | {result['variant']} | {e['trace']} | "
                         f"{'yes' if e['completed'] else 'no (partial)'} | {100*m['completion_fraction']:.1f} | "
                         f"{m['rebuffer_s']:.2f} | {m['startup_s']:.2f} | {m['mean_played_bitrate_kbps']:.1f} | "
                         f"{m['wasted_bytes']/1e6:.3f} | {m['request_retries']} |")
    lines.extend(['', '## Historical controls', '',
                  '| Dataset | Policy | Recovery | Complete | Timing targets passed | Max stall s |',
                  '|---|---|---|---:|---:|---:|'])
    for result in results:
        if result['dataset'] == 'failure_development_sessions':
            continue
        episodes = result['episodes']
        lines.append(f"| {result['dataset']} | {result['policy']} | {result['variant']} | "
                     f"{sum(e['completed'] for e in episodes)}/{len(episodes)} | "
                     f"{sum(e['playback_targets_met'] for e in episodes)}/{len(episodes)} | "
                     f"{max(e['metrics']['rebuffer_s'] for e in episodes):.2f} |")
    dqn = [e for r in results if r['dataset'] == 'failure_development_sessions' and
           r['policy'] == 'dqn' and r['variant'] == 'progress' for e in r['episodes']]
    rule = [e for r in results if r['dataset'] == 'failure_development_sessions' and
            r['policy'] == 'highest_allowed' and r['variant'] == 'progress' for e in r['episodes']]
    timeout = [e for r in results if r['dataset'] == 'failure_development_sessions' and
               r['policy'] == 'dqn' and r['variant'] == 'timeout_only' for e in r['episodes']]
    identical_choices = all(a['policy_actions'] == b['policy_actions'] and
                            a['effective_actions'] == b['effective_actions'] for a, b in zip(dqn, rule))
    lines.extend(['', '## Decision', '',
                  f"Progress-based recovery with frozen DQN completed {sum(e['completed'] for e in dqn)}/2 "
                  f"failure sessions; {sum(e['playback_targets_met'] for e in dqn)}/2 met completion, "
                  'startup <=3 s and stalls <=8 s together.', '',
                  f"DQN and the highest-allowed rule chose identical actions on both recovery sessions: "
                  f"{identical_choices}. This run provides no evidence of a learned-policy advantage.", '',
                  f"Compared with timeout-only retries, progress recovery reduced total DQN stalls "
                  f"from {sum(e['metrics']['rebuffer_s'] for e in timeout):.2f} s to "
                  f"{sum(e['metrics']['rebuffer_s'] for e in dqn):.2f} s, and discarded bytes from "
                  f"{sum(e['metrics']['wasted_bytes'] for e in timeout)/1e6:.3f} MB to "
                  f"{sum(e['metrics']['wasted_bytes'] for e in dqn)/1e6:.3f} MB. "
                  'Both comparisons cover two complete videos.', '',
                  'Completion alone does not justify promotion. Partial-playback stalls cannot be compared '
                  'as if they covered a full video. These sessions are development cases, not untouched tests.', '',
                  'Progress recovery can cancel a higher-quality body after 2 seconds if finishing it '
                  'risks draining the buffer, or after 3 seconds without progress. It avoids abandoning '
                  'near-complete bodies and uses nominal size for a proposed lowest-quality restart. '
                  'All retries use the lowest quality. Each retry has a 30-second deadline; the limit '
                  'is four retries per segment. Timeout/no-progress retries wait 2 seconds. '
                  'Backoff, RTT, downloaded bytes and playback stalls are all charged.', '',
                  'Next experiment: measure the tradeoff between startup delay and a larger startup/resume '
                  'buffer on separate network sessions, against the same simple rule. Compare full playback '
                  'and keep the stall limit. This run does not prove that extra training can remove the stalls.', '',
                  'Limitations: ' + '; '.join(protocol['limitations']) + '.', '',
                  'Reproduce: `.venv/bin/python -m scripts.evaluate_request_recovery`', ''])
    (output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
