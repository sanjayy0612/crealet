"""Replay the predeclared buffer grid with frozen policies; no training.

Run: python -m scripts.evaluate_buffer_experiment
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.benchmark import evaluate
from abr.data import Dataset
from abr.env import StreamingEnv, StreamConfig, QoEWeights
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.recovery import RecoveryConfig
from scripts.evaluate_request_recovery import compact


def summarize(episodes):
    completed = [e['metrics'] for e in episodes if e['completed']]
    def mean(key):
        return float(np.mean([m[key] for m in completed])) if completed else None
    return {
        'sessions': len(episodes), 'completed': len(completed),
        'targets_met': sum(e['playback_targets_met'] for e in episodes),
        'mean_startup_s_completed': mean('startup_s'),
        'mean_rebuffer_s_completed': mean('rebuffer_s'),
        'mean_total_delay_s_completed': float(np.mean([m['startup_s'] + m['rebuffer_s']
                                                       for m in completed])) if completed else None,
        'max_rebuffer_s_completed': max((m['rebuffer_s'] for m in completed), default=None),
        'mean_stall_events_completed': mean('stall_events'),
        'mean_played_kbps_completed': mean('mean_played_bitrate_kbps'),
        'total_wasted_bytes_all_sessions': sum(e['metrics']['wasted_bytes'] for e in episodes)}


def main():
    torch.set_num_threads(1)
    output = Path('runs/buffer-experiment')
    protocol = json.loads((output / 'protocol.json').read_text())
    checkpoint = Path(protocol['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    model, _ = load_model(checkpoint)
    safety = ActionSafety(**protocol['safety_filter'])
    dqn = DecisionPolicy(model, safety=safety)
    datasets = {'additional_sessions': Dataset.load(protocol['primary_dataset']),
                'previous_failure_controls': Dataset.load(protocol['development_control'])}
    expected = [f'buffer-ferry-{name}' for name in protocol['sessions']]
    if [t.name for t in datasets['additional_sessions'].splits['test']] != expected:
        raise ValueError('Dataset sessions differ from fixed protocol')
    results = []
    for label, dataset in datasets.items():
        for profile in protocol['profiles']:
            for name in protocol['policies']:
                env = StreamingEnv(dataset, split='test', config=StreamConfig(**profile['config']),
                                   recovery=RecoveryConfig(**protocol['recovery']), weights=QoEWeights(rebuffer=12))
                kwargs = {'baseline': 'lowest'} if name == 'lowest' else {
                    'baseline': 'model', 'policy': dqn if name == 'dqn' else
                    lambda obs, cfg: int(np.flatnonzero(allowed_actions(obs, safety))[-1]),
                    'policy_metadata': {'kind': name, 'safety_filter': protocol['safety_filter'],
                                        'checkpoint_sha256': protocol['checkpoint_sha256'] if name == 'dqn' else None}}
                result = compact(evaluate(env, **kwargs))
                result.update(dataset=label, profile=profile['name'], policy=name)
                for e in result['episodes']:
                    m = e['metrics']
                    e['playback_targets_met'] = bool(e['completed'] and
                        m['startup_s'] <= protocol['targets']['max_startup_s'] + 1e-9 and
                        m['rebuffer_s'] <= protocol['targets']['max_rebuffer_s'] + 1e-9)
                result['summary'] = summarize(result['episodes'])
                results.append(result)
                print(json.dumps({k: result[k] for k in ('dataset', 'profile', 'policy', 'summary')}), flush=True)
    paired = []
    for r in results:
        baseline = next(b for b in results if b['dataset'] == r['dataset'] and
                        b['policy'] == r['policy'] and b['profile'] == 'start1.0-resume1.0')
        for old, new in zip(baseline['episodes'], r['episodes']):
            if old['trace'] != new['trace']:
                raise ValueError('Paired sessions differ')
            pair = {'dataset': r['dataset'], 'profile': r['profile'], 'policy': r['policy'],
                    'trace': new['trace'], 'both_complete': old['completed'] and new['completed']}
            for metric in ('startup_s', 'rebuffer_s', 'stall_events', 'mean_played_bitrate_kbps'):
                pair['delta_' + metric] = (new['metrics'][metric] - old['metrics'][metric]
                                          if pair['both_complete'] else None)
            pair['delta_total_delay_s'] = (pair['delta_startup_s'] + pair['delta_rebuffer_s']
                                           if pair['both_complete'] else None)
            paired.append(pair)
    payload = {'protocol': protocol, 'results': results, 'paired_changes_from_default': paired,
               'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# Startup/resume buffer experiment', '',
             'Frozen DQN, progress-based request recovery, no training. '
             'All nine profiles and four additional sessions were fixed before downloading/replay.', '',
             'Targets: complete every video, startup <=3 seconds, total stalls <=8 seconds. '
             'Buffer thresholds are media seconds, not wall-clock seconds. Segments are indivisible: '
             'the first segments last 3.9934 seconds, so thresholds 4 and 8 require two and three '
             'initial segments respectively. The buffer cap remains 30 seconds.', '',
             '## Four additional sessions', '',
             'Means below cover completed videos only. Completion and timing gates include every session. '
             'Total delay is startup plus all playback stalls; a larger resume threshold can lengthen each pause.', '',
             '| Policy | Startup/resume thresholds | Complete | All targets met | Mean startup s | Mean stalls s | Mean total delay s | Mean played kbps |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'n/a' if value is None else f'{value:.2f}'
    for r in results:
        if r['dataset'] != 'additional_sessions':
            continue
        s = r['summary']
        lines.append(f"| {r['policy']} | {r['profile']} | {s['completed']}/4 | {s['targets_met']}/4 | "
                     f"{fmt(s['mean_startup_s_completed'])} | {fmt(s['mean_rebuffer_s_completed'])} | "
                     f"{fmt(s['mean_total_delay_s_completed'])} | {fmt(s['mean_played_kbps_completed'])} |")
    lines.extend(['', '## DQN per-session timing', '',
                  '| Sessions | Profile | Trace | Complete | Startup s | Stalls s | Stall count | Targets met |',
                  '|---|---|---|---|---:|---:|---:|---|'])
    for r in results:
        if r['policy'] != 'dqn':
            continue
        for e in r['episodes']:
            m = e['metrics']
            lines.append(f"| {r['dataset']} | {r['profile']} | {e['trace']} | {e['completed']} | "
                         f"{m['startup_s']:.2f} | {m['rebuffer_s']:.2f} | {m['stall_events']} | {e['playback_targets_met']} |")
    differences = []
    for r in results:
        if r['policy'] != 'dqn':
            continue
        rule = next(b for b in results if b['dataset'] == r['dataset'] and
                    b['profile'] == r['profile'] and b['policy'] == 'highest_allowed')
        differences.extend({'dataset': r['dataset'], 'profile': r['profile'], 'trace': a['trace']}
                           for a, b in zip(r['episodes'], rule['episodes'])
                           if a['policy_actions'] != b['policy_actions'] or a['effective_actions'] != b['effective_actions'])
    primary_dqn = {r['profile']: r for r in results if r['dataset'] == 'additional_sessions' and r['policy'] == 'dqn'}
    controls = {r['profile']: r for r in results if r['dataset'] == 'previous_failure_controls' and r['policy'] == 'dqn'}
    default = primary_dqn['start1.0-resume1.0']['summary']
    start4 = primary_dqn['start4.0-resume1.0']['summary']
    start8 = primary_dqn['start8.0-resume1.0']['summary']
    def control_values(profile, metric):
        return '/'.join(f"{e['metrics'][metric]:.2f}" for e in controls[profile]['episodes'])
    lines.extend(['', '## Interpretation and limits', '',
                  f'DQN and highest-allowed differed in action sequences in {len(differences)} of '
                  '54 paired profile/session comparisons. Changed startup/resume context was not used '
                  'to train this checkpoint; weights remain frozen.', '',
                  f"The default profile completed {default['completed']}/4 additional sessions, "
                  f"with maximum stalls {fmt(default['max_rebuffer_s_completed'])} s. "
                  'Zero stalls give no opportunity to demonstrate a resume-buffer improvement. '
                  f"Mean startup was {fmt(default['mean_startup_s_completed'])} s, "
                  f"{fmt(start4['mean_startup_s_completed'])} s and {fmt(start8['mean_startup_s_completed'])} s "
                  f"at thresholds 1, 4 and 8 s. The 8 s threshold passed all targets on {start8['targets_met']}/4 sessions. "
                  'Keep the current defaults; these results do not establish a better profile.', '',
                  'On the two previous failure controls, startup=4 s and resume=1 s reduced stalls '
                  f"from {control_values('start1.0-resume1.0', 'rebuffer_s')} s to "
                  f"{control_values('start4.0-resume1.0', 'rebuffer_s')} s, with startup changing from "
                  f"{control_values('start1.0-resume1.0', 'startup_s')} s to "
                  f"{control_values('start4.0-resume1.0', 'startup_s')} s. One session still failed the stall gate. "
                  'Increasing only the resume threshold to 8 s increased stalls to '
                  f"{control_values('start1.0-resume8.0', 'rebuffer_s')} s "
                  'even though it reduced the number of separate pauses. '
                  'A lower pause count does not mean less waiting.', '',
                  'No profile is promoted from this comparison. Choosing a profile after seeing these '
                  'results requires a separate confirmation set. A mean below a limit does not imply '
                  'every session passed it. Previous failure sessions are development controls only.', '',
                  '; '.join(protocol['limitations']) + '.', '',
                  'Reproduce: `.venv/bin/python -m scripts.prepare_buffer_experiment` then '
                  '`.venv/bin/python -m scripts.evaluate_buffer_experiment`.', ''])
    (output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
