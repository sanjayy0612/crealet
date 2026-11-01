"""Paired native Chrome confirmation of the experimental 4-second buffer gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from abr.data import Dataset
from scripts.assess_production import quality_exposure
from scripts.audit_player_captures import event_stalls

OUT = Path('runs/native-quality-gate')
CHECKPOINT = 'runs/learning-ablation/baseline-seed-7/accepted.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    contents = [{'name': 'envivio', 'dataset': 'data/startup-ladder/development.json',
                 'media': 'artifacts/startup-ladder-v1'},
                {'name': 'sintel', 'dataset': 'data/sintel-ladder/development.json',
                 'media': 'artifacts/sintel-ladder-v1'}]
    cases = [{'name': f'{label}-{threshold}s', 'policy': 'dqn', 'split': 'test',
              'trace': index, 'port': port, 'wall_limit_s': 240}
             for label, index in [('slow-first', 0), ('upgrade', 2)]
             for threshold, port in [(12, 8871), (4, 8872)]]
    protocol = {'checkpoint': CHECKPOINT, 'checkpoint_sha256': sha(CHECKPOINT),
                'training_dataset': 'data/hard_envivio/dataset.json', 'max_buffer_s': 60,
                'recovery': True, 'thresholds_s': [12, 4], 'cases': cases,
                'contents': [{**c, 'dataset_sha256': sha(c['dataset']),
                              'media_index_sha256': sha(Path(c['media']) / 'index.json')} for c in contents],
                'sources': {p: sha(p) for p in ('abr/player_server.py', 'abr/models/safety.py',
                                               'scripts/capture_policy_player.mjs', __file__)},
                'comparison': 'Both complete, startup <=3s, stalls <=8s; candidate has no extra stall seconds, '
                              'startup no more than 0.25s above paired baseline (browser timing tolerance). '
                              'Compare emergency exposure; retain failures; no default/checkpoint promotion.',
                'selection': 'Known trace 2 improved in simulator; known slow-first trace 0 did not. Both on both contents, fixed before native playback.',
                'limits': 'Four isolated Chrome processes run concurrently per content on one host. '
                          'This tests paired behavior, not single-player memory approval, a soak, or independent generalization.'}
    text = json.dumps(protocol, indent=2) + '\n'
    target = OUT / 'protocol.json'
    if target.exists() and target.read_text() != text:
        raise ValueError('Existing native protocol differs')
    target.write_text(text)
    (OUT / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    return protocol


def capture(protocol):
    for content in protocol['contents']:
        servers, logs = [], []
        destination = OUT / content['name']
        destination.mkdir(exist_ok=True)
        if (destination / 'summary.json').exists():
            raise ValueError('Capture already exists; use --report-only to avoid overwriting evidence')
        try:
            for threshold, port in [(12, 8871), (4, 8872)]:
                log = (destination / f'server-{threshold}s.log').open('w')
                logs.append(log)
                args = [sys.executable, '-m', 'abr.player_server', '--checkpoint', CHECKPOINT,
                        '--dataset', protocol['training_dataset'], '--media-dir', content['media'],
                        '--replay-dataset', content['dataset'], '--allow-experimental-media',
                        '--experimental-max-buffer-s', '60', '--enable-request-recovery', '--port', str(port)]
                if threshold != 12:
                    args.extend(['--experimental-min-buffer-s', str(threshold)])
                server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                servers.append(server)
                for _ in range(100):
                    if server.poll() is not None:
                        raise RuntimeError(f'Server exited; inspect {log.name}')
                    try:
                        with urlopen(f'http://127.0.0.1:{port}/config', timeout=.5) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError('Local player server readiness timed out')
            subprocess.run(['node', 'scripts/capture_policy_player.mjs', '--port=8871',
                            f'--cases={OUT / "cases.json"}', f'--output={destination}', '--memory'], check=True)
        finally:
            for server in servers:
                if server.poll() is None:
                    server.terminate()
            for server in servers:
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            for log in logs:
                log.close()


def report(protocol):
    rows = []
    for content in protocol['contents']:
        dataset = Dataset.load(content['dataset'])
        for case in protocol['cases']:
            path = OUT / content['name'] / (case['name'] + '.json')
            data = json.loads(path.read_text())
            ctx, metrics = data['context'], data['metrics']
            threshold = 4 if case['name'].endswith('-4s') else 12
            segments = data['segments']
            decisions = data['transport_audit']['decisions']
            stalls, _ = event_stalls(data['events'], metrics['wall_time_s'])
            checks = {
                'checkpoint': ctx['checkpoint_sha256'] == protocol['checkpoint_sha256'],
                'dataset': ctx['dataset_sha256'] == dataset.fingerprint,
                'trace': ctx['trace_sha256'] == dataset.splits['test'][case['trace']].fingerprint,
                'applied_threshold': ctx['safety_filter']['min_buffer_s'] == threshold,
                'original_threshold_preserved': ctx['training_safety_filter']['min_buffer_s'] == 12,
                'throughput_margin_unchanged': ctx['safety_filter']['throughput_factor'] == .65,
                'cap': ctx['config']['max_buffer_s'] == 60,
                'experimental_override_recorded': ctx['experimental_overrides'].get('min_buffer_s') == (4 if threshold == 4 else None),
                'decoded_frames': metrics['total_frames'] > 0,
                'visible': bool(data['samples']) and all(s['visibility'] == 'visible' for s in data['samples']),
                'segment_bytes': all(s['bytes'] == dataset.video.sizes_bytes[s['segment_index'], s['action']] for s in segments),
                'decision_count': len(decisions) == len(segments),
                'filter_respected': all(d['filter_applied'] and d['action'] in d['allowed_actions'] for d in decisions),
                'policy_choices': len(decisions) == len(segments) and all(s['policy_action'] == d['action'] for s, d in zip(segments, decisions)),
                'stall_events_agree': abs(stalls - metrics['rebuffer_s']) < .02,
                'transport_finished': not data['transport_audit']['in_flight']}
            if data['completed']:
                checks['full_segment_sequence'] = [s['segment_index'] for s in segments] == list(range(len(dataset.video.durations_s)))
                checks['full_media'] = abs(metrics['played_s'] - sum(dataset.video.durations_s)) < .2
            exposure = quality_exposure(data, 300) if data['completed'] else None
            rows.append({'content': content['name'], 'case': case['name'], 'threshold_s': threshold,
                         'capture': str(path), 'capture_sha256': sha(path), 'completed': data['completed'],
                         'checks': {k: bool(v) for k, v in checks.items()}, 'valid': all(checks.values()),
                         'metrics': metrics, 'quality_exposure': exposure,
                         'quality_limits_met': bool(exposure and exposure['emergency_fraction'] <= .1 + 1e-9 and
                                                    exposure['longest_continuous_emergency_s'] <= 12 + 1e-9)})
    pairs = []
    for content in protocol['contents']:
        for label in ('slow-first', 'upgrade'):
            base = next(r for r in rows if r['content'] == content['name'] and r['case'] == label + '-12s')
            trial = next(r for r in rows if r['content'] == content['name'] and r['case'] == label + '-4s')
            a, b = base['metrics'], trial['metrics']
            passed = all(r['valid'] and r['completed'] and r['metrics']['startup_s'] <= 3 and r['metrics']['rebuffer_s'] <= 8 for r in (base, trial))
            pairs.append({'content': content['name'], 'case': label,
                          'startup_delta_s': b['startup_s'] - a['startup_s'],
                          'stall_delta_s': b['rebuffer_s'] - a['rebuffer_s'],
                          'wasted_bytes_delta': b['discarded_media_bytes'] - a['discarded_media_bytes'],
                          'emergency_reduction_s': base['quality_exposure']['emergency_media_s'] - trial['quality_exposure']['emergency_media_s'] if base['completed'] and trial['completed'] else None,
                          'paired_playback_gate_passed': bool(passed and b['startup_s'] <= a['startup_s'] + .25 and b['rebuffer_s'] <= a['rebuffer_s'] + 1e-6)})
    payload = {'protocol_sha256': sha(OUT / 'protocol.json'), 'captures': rows, 'pairs': pairs,
               'all_captures_valid': all(r['valid'] for r in rows),
               'all_paired_playback_gates_passed': all(p['paired_playback_gate_passed'] for p in pairs),
               'defaults_changed': False, 'production_approved': False}
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pairs': pairs, 'all_captures_valid': payload['all_captures_valid']}), flush=True)
    if not payload['all_captures_valid']:
        raise RuntimeError('Native capture integrity failed; inspect results')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    protocol = prepare()
    if not args.report_only:
        capture(protocol)
    report(protocol)


if __name__ == '__main__':
    main()
