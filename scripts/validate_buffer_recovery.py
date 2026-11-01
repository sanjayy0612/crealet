"""Native Chrome confirmation; retain both positive and negative candidate results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np

from abr.buffer_safety import BufferAwareSafety, allowed_actions
from abr.data import Dataset
from abr.models.safety import ActionSafety
from scripts.assess_production import quality_exposure
from scripts.audit_player_captures import event_stalls
from scripts.evaluate_buffer_recovery import OUT, sha


def prepare():
    simulation = json.loads((OUT / 'simulator.json').read_text())
    if not simulation['native_confirmation_warranted']:
        raise ValueError('No measured simulator improvement supports native confirmation')
    base = json.loads((OUT / 'protocol.json').read_text())
    protocol = {'checkpoint': base['checkpoint'], 'checkpoint_sha256': base['checkpoint_sha256'],
        'simulator_protocol_sha256': sha(OUT / 'protocol.json'),
        'variants': base['variants'], 'baseline_promotion_gate_passed': simulation['all_paired_gates_passed'],
        'cases': [{'name': f'{label}-{variant}', 'policy': 'dqn', 'split': split, 'trace': index,
                   'port': port, 'wall_limit_s': 240}
                  for label, split, index in [('slow-first', 'test', 0), ('controlled', 'validation', 2)]
                  for variant, port in [('baseline', 8881), ('candidate', 8882)]],
        'contents': [{'name': name, 'dataset': f'data/{folder}/development.json', 'media': f'artifacts/{media}',
                      'dataset_sha256': sha(f'data/{folder}/development.json'),
                      'media_index_sha256': sha(f'artifacts/{media}/index.json')}
                     for name, folder, media in [('envivio', 'startup-ladder', 'startup-ladder-v1'),
                                                ('sintel', 'sintel-ladder', 'sintel-ladder-v1')]],
        'sources': {p: sha(p) for p in ('scripts/validate_buffer_recovery.py', 'abr/player_server.py',
                                       'abr/buffer_safety.py', 'scripts/capture_policy_player.mjs')},
        'gate': 'Paired completion, startup <=3s and <=baseline+0.25s, no added stalls, no added emergency '
                'media, no lost draft quality pass, no additional discarded media. Retain failures; no promotion '
                'even if native passes because the simulator transfer gate failed.',
        'limits': 'Known cases, one capture per setting, four isolated Chrome processes concurrently per content. '
                  'Not a single-player memory approval, soak, or generalization result.'}
    text = json.dumps(protocol, indent=2) + '\n'
    path = OUT / 'native-protocol.json'
    if path.exists() and path.read_text() != text: raise ValueError('Native protocol differs')
    path.write_text(text)
    (OUT / 'native-cases.json').write_text(json.dumps(protocol['cases'], indent=2) + '\n')
    return protocol


def capture(protocol):
    for content in protocol['contents']:
        destination = OUT / content['name']
        destination.mkdir(exist_ok=True)
        if (destination / 'summary.json').exists(): raise ValueError('Refusing to overwrite complete captures')
        servers, logs = [], []
        try:
            for name, port in [('baseline', 8881), ('candidate', 8882)]:
                log = (destination / f'server-{name}.log').open('w'); logs.append(log)
                command = [sys.executable, '-m', 'abr.player_server', '--dataset', 'data/hard_envivio/dataset.json',
                           '--checkpoint', protocol['checkpoint'], '--media-dir', content['media'],
                           '--replay-dataset', content['dataset'], '--allow-experimental-media',
                           '--experimental-max-buffer-s', '60', '--experimental-min-buffer-s', '4',
                           '--enable-request-recovery', '--port', str(port)]
                if name == 'candidate': command.append('--experimental-buffer-aware-safety')
                server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT); servers.append(server)
                for _ in range(100):
                    if server.poll() is not None: raise RuntimeError(f'Server exited; inspect {log.name}')
                    try:
                        with urlopen(f'http://127.0.0.1:{port}/config', timeout=.5) as response:
                            if response.status == 200: break
                    except OSError: time.sleep(.1)
                else: raise RuntimeError('Player readiness timed out')
            subprocess.run(['node', 'scripts/capture_policy_player.mjs', '--port=8881',
                            f'--cases={OUT / "native-cases.json"}', f'--output={destination}', '--memory'], check=True)
        finally:
            for server in servers:
                if server.poll() is None: server.terminate()
            for server in servers:
                try: server.wait(timeout=5)
                except subprocess.TimeoutExpired: server.kill(); server.wait()
            for log in logs: log.close()


def report(protocol):
    rows = []
    for content in protocol['contents']:
        data = Dataset.load(content['dataset'])
        for case in protocol['cases']:
            path = OUT / content['name'] / (case['name'] + '.json')
            raw = json.loads(path.read_text()); ctx = raw['context']; m = raw['metrics']
            candidate = case['name'].endswith('-candidate')
            rule = BufferAwareSafety(**ctx['safety_filter']) if candidate else ActionSafety(**ctx['safety_filter'])
            expected = protocol['variants']['buffer_aware' if candidate else 'baseline']
            segments, decisions = raw['segments'], raw['transport_audit']['decisions']
            stalls, _ = event_stalls(raw['events'], m['wall_time_s'])
            checks = {'checkpoint': ctx['checkpoint_sha256'] == protocol['checkpoint_sha256'],
                'dataset': ctx['dataset_sha256'] == data.fingerprint,
                'trace': ctx['trace_sha256'] == data.splits[case['split']][case['trace']].fingerprint,
                'safety': asdict(rule) == expected, 'original_gate': ctx['training_safety_filter']['min_buffer_s'] == 12,
                'override': ctx['experimental_overrides'].get('buffer_aware_safety', False) == candidate,
                'buffer_gate': ctx['config']['max_buffer_s'] == 60 and ctx['experimental_overrides']['min_buffer_s'] == 4,
                'decoded': m['total_frames'] > 0,
                'visible': bool(raw['samples']) and all(s['visibility'] == 'visible' for s in raw['samples']),
                'bytes': all(s['bytes'] == data.video.sizes_bytes[s['segment_index'], s['action']] for s in segments),
                'choices': len(segments) == len(decisions) and all(s['policy_action']==d['action'] and d['action'] in d['allowed_actions'] for s,d in zip(segments,decisions)),
                'events': abs(stalls-m['rebuffer_s']) < .02, 'transport_finished': not raw['transport_audit']['in_flight']}
            if candidate:
                for decision in decisions:
                    i = decision['segment_index']; duration = data.video.durations_s[i]; ladder = data.video.bitrates_kbps
                    obs = {'history': np.array(decision['recovery_input_history'],dtype=np.float32),
                           'state': np.array(decision['state'],dtype=np.float32),
                           'candidates': np.column_stack((ladder/1000, ladder*duration/8000, np.full(len(ladder),duration))).astype(np.float32)}
                    checks[f'candidate_mask_{i}'] = np.flatnonzero(allowed_actions(obs,rule)).tolist()==decision['allowed_actions']
            if raw['completed']:
                checks['all_segments'] = [s['segment_index'] for s in segments] == list(range(len(data.video.durations_s)))
                checks['full_duration'] = abs(m['played_s']-sum(data.video.durations_s)) < .2
            q = quality_exposure(raw,300) if raw['completed'] else None
            rows.append({'content':content['name'],'case':case['name'],'capture':str(path),'sha256':sha(path),
                'completed':raw['completed'],'metrics':m,'quality_exposure':q,'checks':{k:bool(v) for k,v in checks.items()},
                'valid':all(checks.values()),'quality_passed':bool(q and q['emergency_fraction']<=.1+1e-9 and q['longest_continuous_emergency_s']<=12+1e-9)})
    pairs=[]
    for content in protocol['contents']:
        for label in ('slow-first','controlled'):
            a=next(r for r in rows if r['content']==content['name'] and r['case']==label+'-baseline')
            b=next(r for r in rows if r['content']==content['name'] and r['case']==label+'-candidate')
            complete=bool(a['completed'] and b['completed'])
            reduction=a['quality_exposure']['emergency_media_s']-b['quality_exposure']['emergency_media_s'] if complete else None
            passed=complete and a['valid'] and b['valid'] and all(r['metrics']['startup_s']<=3 for r in (a,b)) and b['metrics']['startup_s']<=a['metrics']['startup_s']+.25 and b['metrics']['rebuffer_s']<=a['metrics']['rebuffer_s']+1e-6 and reduction>=-1e-8 and (not a['quality_passed'] or b['quality_passed']) and b['metrics']['discarded_media_bytes']<=a['metrics']['discarded_media_bytes']
            pairs.append({'content':content['name'],'case':label,'emergency_reduction_s':reduction,
                'stall_delta_s':b['metrics']['rebuffer_s']-a['metrics']['rebuffer_s'],
                'startup_delta_s':b['metrics']['startup_s']-a['metrics']['startup_s'],
                'bitrate_delta_kbps':b['metrics']['mean_played_bitrate_kbps']-a['metrics']['mean_played_bitrate_kbps'],
                'discarded_bytes_delta':b['metrics']['discarded_media_bytes']-a['metrics']['discarded_media_bytes'], 'passed':bool(passed)})
    result={'protocol_sha256':sha(OUT/'native-protocol.json'),'captures':rows,'pairs':pairs,
            'all_captures_valid':all(r['valid'] for r in rows),'native_gate_passed':all(r['passed'] for r in pairs),
            'defaults_changed':False,'promotion_recommended':False,'production_approved':False}
    (OUT/'native.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pairs':pairs,'all_captures_valid':result['all_captures_valid']}),flush=True)
    if not result['all_captures_valid']: raise RuntimeError('Capture integrity failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    protocol=json.loads((OUT/'native-protocol.json').read_text()) if args.report_only else prepare()
    if not args.report_only: capture(protocol)
    report(protocol)


if __name__=='__main__':main()
