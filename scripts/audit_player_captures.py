"""Audit capture integrity separately from whether a policy met playback targets."""
import hashlib
import json
from pathlib import Path

from abr.data import Dataset
from abr.validate_player import player_metrics

CASES = {'dqn-measured': 0, 'throughput-measured': 0,
         'dqn-outage': 2, 'throughput-outage': 2, 'highest-outage': 2}


def event_stalls(events, end_s):
    started, waiting = False, None
    seconds, count = 0., 0
    for event in events:
        if event['type'] == 'playing':
            if waiting is not None:
                dt = event['t_s'] - waiting
                seconds += dt
                count += int(dt > 1e-6)
                waiting = None
            started = True
        elif event['type'] == 'waiting' and started and waiting is None:
            waiting = event['t_s']
    if waiting is not None:
        dt = end_s - waiting
        seconds += dt
        count += int(dt > 1e-6)
    return seconds, count


def main():
    output = Path('runs/player-hard')
    dataset = Dataset.load('data/hard_envivio/dataset.json')
    checkpoint_sha = hashlib.sha256(Path('runs/dqn-hard/dueling_double_dqn_feasible.npz').read_bytes()).hexdigest()
    results = {}
    for name, trace_index in CASES.items():
        data = json.loads((output / (name + '.json')).read_text())
        ctx, metrics = data['context'], data['metrics']
        segments = data['segments']
        decisions = data['transport_audit']['decisions']
        stalls, positive_events = event_stalls(data['events'], metrics['wall_time_s'])
        independent = None
        if data['completed']:
            independent = player_metrics({**data, 'origin': 'browser_capture', 'issues': [],
                                          'end_s': data['events'][-1]['t_s'],
                                          'network_setup_evidence': ctx['network_method']}, 'vod')
        checks = {
            'native_capture': data['origin'] == 'native_mse_policy_capture',
            'checkpoint_matches': ctx['checkpoint_sha256'] == checkpoint_sha,
            'dataset_matches': ctx['dataset_sha256'] == dataset.fingerprint,
            'trace_matches': ctx['trace_sha256'] == dataset.splits['test'][trace_index].fingerprint,
            'visible_samples': bool(data['samples']) and all(s['visibility'] == 'visible' for s in data['samples']),
            'real_decoded_frames': metrics['total_frames'] > 0,
            'segment_bytes_match': all(s['bytes'] == dataset.video.sizes_bytes[s['segment_index'], s['action']] for s in segments),
            'segment_choices_match_server': all(s['action'] == decisions[i]['action'] for i, s in enumerate(segments)),
            'stall_seconds_match_events': abs(stalls - metrics['rebuffer_s']) < .02,
            'complete_has_all_segments': not data['completed'] or len(segments) == len(dataset.video.durations_s),
            'complete_has_full_media': not data['completed'] or abs(metrics['played_s'] - dataset.video.durations_s.sum()) < .2,
            'complete_transport_finished': not data['completed'] or not data['transport_audit']['in_flight'],
            'independent_validator_agrees': independent is None or abs(independent['rebuffer_s'] - metrics['rebuffer_s']) < .02,
        }
        checks = {key: bool(value) for key, value in checks.items()}
        results[name] = {'capture_checks': checks, 'capture_valid': all(checks.values()),
                         'completed': data['completed'], 'error': data['error'],
                         'raw_waiting_event_count': metrics['stall_events'],
                         'positive_duration_stall_events': positive_events,
                         'independent_stall_seconds': stalls,
                         'meets_absolute_targets': bool(data['completed'] and metrics['startup_s'] <= 3 and metrics['rebuffer_s'] <= 8),
                         'independent_completed_playback_metrics': independent}
    report = {'all_captures_valid': all(r['capture_valid'] for r in results.values()), 'cases': results,
              'note': 'Native waiting/playing transitions can have identical timestamps. Raw waiting-event counts include those; positive-duration counts exclude them. Incomplete baseline playback is a policy outcome, not a complete-video result.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['all_captures_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
