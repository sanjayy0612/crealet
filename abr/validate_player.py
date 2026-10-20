"""Prepare fixed-lowest-quality cases and compare native player telemetry.

This tool checks measurements and declared case identity. It cannot verify that
an external network emulator or the claimed media was actually used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_PROTOCOL = Path(__file__).resolve().parent.parent / 'experiments/configs/evaluation_v2.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and nonnegative')
    return float(value)


def prepare_case(report, episode_index=0):
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError('episode index must be a nonnegative integer')
    p = report['provenance']
    if p['environment_version'] != 2 or p['baseline'] != 'lowest':
        raise ValueError('Use a V2 fixed-lowest diagnostic report')
    if p['split'] == 'test':
        raise ValueError('Use calibration/validation data, not the reserved final test split')
    cfg = p['config']
    if cfg['mode'] == 'live' and cfg['catchup']:
        raise ValueError('Disable live catchup for initial timing validation; browser and simulator controllers differ')
    ep = report['episodes'][episode_index]
    if ep['completed'] is not True or not ep['steps'] or any(row['action'] != 0 for row in ep['steps']):
        raise ValueError('Case must be a complete, fixed-lowest episode')
    if any(not row['segment_complete'] for row in ep['steps']):
        raise ValueError('Incomplete requests cannot be a successful reference case')
    identity = {'mode': cfg['mode'], 'baseline': 'lowest',
                'dataset_sha256': p['dataset_sha256'], 'media_sha256': p['media_sha256'],
                'trace_sha256': ep['trace_sha256'], 'config_sha256': digest(cfg),
                'actions_sha256': digest([row['action'] for row in ep['steps']]),
                'simulation_source_sha256': p['source_sha256']}
    return {'schema_version': 1, 'case_id': digest(identity), 'identity': identity,
            'baseline': 'lowest', 'config': cfg, 'expected_metrics': ep['metrics'],
            'trace_source': ep['source'], 'trace_group': ep['source_group'],
            'requirements': ['same packaged media and representation mapping',
                'independently recorded external network setup', 'no seeking, pauses or playback loop',
                'finite event, beginning at media time zero', 'keep manifest/init/decode timing differences visible'],
            'limitations': 'Case identity is declared metadata, not proof of matched browser network conditions'}


def player_metrics(capture, mode):
    if mode not in {'vod', 'live'}:
        raise ValueError('Unknown streaming mode')
    if capture.get('schema_version') != 1 or capture.get('origin') != 'browser_capture':
        raise ValueError('Expected a version 1 browser capture')
    if capture.get('completed') is not True or capture.get('issues'):
        raise ValueError('Incomplete or interrupted capture: inspect raw events and issues')
    if not isinstance(capture.get('network_setup_evidence'), str) or not capture['network_setup_evidence'].strip():
        raise ValueError('External network setup evidence is required')
    events, samples = capture['events'], capture['samples']
    end = number(capture['end_s'], 'end_s')
    if not events or events[0]['type'] != 'start' or abs(number(events[0]['t_s'], 'start')) > .01:
        raise ValueError('Capture must start at wall time zero')
    allowed = {'start', 'playing', 'waiting', 'pause', 'seeking', 'ratechange', 'ended', 'error'}
    for collection in (events, samples):
        previous, previous_media = -1., -1.
        for row in collection:
            t = number(row['t_s'], 'timestamp')
            media = number(row['media_time_s'], 'media_time_s')
            if media + 1e-6 < previous_media:
                raise ValueError('Media clock moved backwards without a valid comparison origin')
            previous_media = media
            if t < previous or t > end + 1e-6:
                raise ValueError('Timestamps must be ordered and within the capture')
            previous = t
    if any(e['type'] not in allowed for e in events):
        raise ValueError('Unknown event type')
    if sum(e['type'] == 'start' for e in events) != 1 or sum(e['type'] == 'ended' for e in events) != 1 or events[-1]['type'] != 'ended':
        raise ValueError('Exactly one start and terminal ended event required')
    if any(e['type'] in {'seeking', 'error'} for e in events):
        raise ValueError('Seeking or playback errors invalidate a timing comparison')
    first = next((e for e in events if e['type'] == 'playing'), None)
    if first is None:
        raise ValueError('No playback start recorded')
    if events[0]['media_time_s'] > .1 or first['media_time_s'] > .1:
        raise ValueError('V2 validation currently requires playback from media time zero')
    final = events[-1]
    if abs(end - final['t_s']) > .1:
        raise ValueError('Capture end does not match playback end')
    if any(e['type'] == 'pause' and (final['t_s'] - e['t_s'] > .1 or abs(e['media_time_s'] - final['media_time_s']) > .1) for e in events):
        raise ValueError('Intentional/external pause invalidates a timing comparison')
    stall_start = None
    rebuffer, count = 0., 0
    started = False
    for e in events:
        if e['type'] == 'playing':
            if stall_start is not None:
                duration = e['t_s'] - stall_start
                if duration > 1e-6:
                    rebuffer += duration
                    count += 1
                stall_start = None
            started = True
        elif e['type'] == 'waiting' and started and stall_start is None:
            stall_start = e['t_s']
    if stall_start is not None:
        raise ValueError('Capture ends with an unresolved stall')
    result = {'startup_s': first['t_s'], 'rebuffer_s': rebuffer, 'stall_events': count,
              'played_s': final['media_time_s'] - events[0]['media_time_s'], 'wall_time_s': final['t_s']}
    if result['played_s'] <= 0:
        raise ValueError('Capture has no positive media progress')
    if mode == 'live':
        observed = [s for s in samples if s['t_s'] >= first['t_s']]
        if len(observed) < 2 or observed[0]['t_s'] - first['t_s'] > .15 or end - observed[-1]['t_s'] > .15:
            raise ValueError('Live latency samples must cover the post-startup interval')
        area = 0.
        for a, b in zip(observed, observed[1:]):
            x = number(a.get('live_latency_s'), 'live latency')
            y = number(b.get('live_latency_s'), 'live latency')
            dt = b['t_s'] - a['t_s']
            if dt > .5:
                raise ValueError('Live telemetry gap exceeds 0.5 seconds')
            area += dt * (x + y) / 2
        span = observed[-1]['t_s'] - observed[0]['t_s']
        if span <= 0:
            raise ValueError('Positive live latency observation interval required')
        result['mean_live_latency_s'] = area / span
    return result


def compare(case, capture, protocol):
    if case.get('schema_version') != 1 or case['case_id'] != digest(case['identity']):
        raise ValueError('Invalid validation case identity')
    if digest(case['config']) != case['identity']['config_sha256']:
        raise ValueError('Case config differs from its identity')
    if capture.get('identity') != case['identity']:
        raise ValueError('Browser and simulator case identities differ')
    metrics = player_metrics(capture, case['identity']['mode'])
    checks = {}
    for name, observed in metrics.items():
        expected = number(case['expected_metrics'][name], f'expected {name}')
        tolerance = protocol['validation_tolerances'][name]
        limit = number(tolerance['absolute'], 'absolute tolerance') + number(tolerance['relative'], 'relative tolerance') * abs(expected)
        error = abs(observed - expected)
        checks[name] = {'simulator': expected, 'browser': observed, 'absolute_error': error,
                        'allowed_error': limit, 'within_tolerance': error <= limit + 1e-9}
    return {'schema_version': 1, 'case_id': case['case_id'],
            'status': 'within_tolerances' if all(x['within_tolerance'] for x in checks.values()) else 'different',
            'checks': checks, 'protocol_id': protocol['protocol_id'], 'protocol_status': protocol['status'],
            'protocol_sha256': digest(protocol), 'capture_sha256': digest(capture),
            'validator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'network_setup_evidence': capture['network_setup_evidence'],
            'real_player_equivalence_established': False,
            'limitations': ['This is one calibration comparison, not broad environment validation.',
                'Media and external network setup identity are self-reported; verify the artifacts independently.',
                'Browser startup includes manifest/init/decode costs absent from the simulator.',
                'Live latency uses sampled player clock estimates; V2 uses an ideal source clock.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--report', required=True)
    prep.add_argument('--episode', type=int, default=0)
    prep.add_argument('--output', required=True)
    check = sub.add_parser('compare')
    check.add_argument('--case', required=True)
    check.add_argument('--capture', required=True)
    check.add_argument('--protocol', default=str(DEFAULT_PROTOCOL))
    check.add_argument('--output', required=True)
    args = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    try:
        if args.command == 'prepare':
            result = prepare_case(read(args.report), args.episode)
        else:
            result = compare(read(args.case), read(args.capture), read(args.protocol))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'output': str(out), 'status': result.get('status', 'case_prepared')}))
        return 1 if result.get('status') == 'different' else 0
    except (ValueError, KeyError, IndexError, TypeError) as error:
        p.exit(2, f'Invalid validation input: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
