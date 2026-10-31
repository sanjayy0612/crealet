"""Gate matched training on actual frozen-model media playback, not reward alone."""
import hashlib
import json
from pathlib import Path

from abr.data import Dataset
from scripts.audit_player_captures import event_stalls


def main():
    out = Path('runs/startup-ladder')
    p = json.loads((out / 'protocol.json').read_text())
    d = Dataset.load(p['development_dataset'])
    sim = json.loads((out / 'frozen-simulator.json').read_text())
    rows = []
    for case in json.loads((out / 'cases.json').read_text()):
        x = json.loads((out / 'native' / (case['name'] + '.json')).read_text())
        c, m = x['context'], x['metrics']
        stalls, _ = event_stalls(x['events'], m['wall_time_s'])
        checks = {
            'frozen_checkpoint': c['checkpoint_sha256'] == p['checkpoint_sha256'],
            'media_transfer_explicit': c['experimental_overrides'].get('media_changed') is True,
            'replay_dataset': c['dataset_sha256'] == d.fingerprint,
            'trace': c['trace_sha256'] == d.splits['test'][case['trace']].fingerprint,
            'new_ladder': [r['bitrate_kbps'] for r in c['representations']] == d.video.bitrates_kbps.tolist(),
            'new_cap_and_recovery': c['config']['max_buffer_s'] == 60 and c['request_recovery'] == p['recovery'],
            'actual_frames': m['total_frames'] > 0,
            'visible': bool(x['samples']) and all(s['visibility'] == 'visible' for s in x['samples']),
            'byte_sizes': all(s['bytes'] == d.video.sizes_bytes[s['segment_index'], s['action']] for s in x['segments']),
            'complete_has_all_media': not x['completed'] or len(x['segments']) == len(d.video.durations_s) and abs(m['played_s'] - d.video.durations_s.sum()) < .1,
            'event_stalls': abs(stalls - m['rebuffer_s']) < .02,
            'source_ended': not x['completed'] or any(e['type'] == 'ended' for e in x['events']),
            'transport_quiescent': not x['transport_audit']['in_flight'],
        }
        checks = {k: bool(v) for k, v in checks.items()}
        rows.append({'case': case['name'], 'metrics': m, 'completed': x['completed'], 'error': x['error'],
                     'checks': checks, 'valid': all(checks.values()),
                     'passed': bool(x['completed'] and m['startup_s'] <= 3 and m['rebuffer_s'] <= 8)})
    simulator_pass = all(e['playback_targets_met'] for r in sim['results'] for e in r['episodes'])
    native_pass = all(r['valid'] and r['passed'] for r in rows)
    index_matches = hashlib.sha256(Path('artifacts/startup-ladder-v1/index.json').read_bytes()).hexdigest() == p['media_index_sha256']
    result = {'simulator_pass': simulator_pass, 'native_pass': native_pass, 'media_index_matches': index_matches,
              'training_gate_passed': simulator_pass and native_pass and index_matches, 'cases': rows,
              'limits': 'Known sessions, one content video, intentionally very-low-quality rung; engineering gate only'}
    (out / 'frozen-native-gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['training_gate_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
