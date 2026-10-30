"""Audit frozen prefetch trials and report the predeclared retraining gate."""
import hashlib
import json
from pathlib import Path

from abr.data import Dataset
from scripts.audit_player_captures import event_stalls

OUT = Path('runs/prefetch-validation')


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    simulator = json.loads((OUT / 'simulator.json').read_text())
    dataset = Dataset.load('data/prefetch-sessions/dataset.json')
    cases = json.loads((OUT / 'native-cases.json').read_text())
    native = []
    for case in cases:
        data = json.loads((OUT / 'native' / (case['name'] + '.json')).read_text())
        memory = json.loads((OUT / 'native' / (case['name'] + '-memory.json')).read_text())
        context, reported_metrics, segments = data['context'], data['metrics'], data['segments']
        requests, decisions = data['transport_audit']['requests'], data['transport_audit']['decisions']
        stalls, _ = event_stalls(data['events'], reported_metrics['wall_time_s'])
        first_playing = next((e['t_s'] for e in data['events'] if e['type'] == 'playing'), None)
        # Canonical event-based metrics, retaining the original counters and
        # their discrepancies. Do not widen the recorder-consistency tolerance.
        metrics = {**reported_metrics, 'startup_s': first_playing, 'rebuffer_s': stalls}
        cap = float(case['name'].split('-')[0][3:])
        completed_files = {f"{context['representations'][s['action']]['id']}/{s['segment_index']+1}.m4s" for s in segments}
        media = [r for r in requests if not r['file'].endswith('Header.m4s')]
        sent = sum(r['bytes_sent'] for r in media)
        discarded = sum(r['bytes_sent'] for r in media if not r['complete'] or r['file'] not in completed_files)
        position = 0.
        unplayed = 0.
        for s in segments:
            duration = context['durations_s'][s['segment_index']]
            fraction = max(0., min(1., (position + duration - metrics['played_s']) / duration))
            unplayed += fraction * s['bytes']
            position += duration
        checks = {
            'native_capture': data['origin'] == 'native_mse_policy_capture',
            'frozen_checkpoint': context['checkpoint_sha256'] == protocol['checkpoint_sha256'],
            'replay_dataset': context['dataset_sha256'] == dataset.fingerprint,
            'trace_identity': context['trace_sha256'] == dataset.splits['test'][case['trace']].fingerprint,
            'trained_cap_retained': context['training_config']['max_buffer_s'] == 30.,
            'experimental_cap_recorded': context['config']['max_buffer_s'] == cap,
            'no_native_recovery_claimed': context['request_recovery'] is False,
            'visible': bool(data['samples']) and all(s['visibility'] == 'visible' for s in data['samples']),
            'segment_bytes': all(s['bytes'] == dataset.video.sizes_bytes[s['segment_index'], s['action']] for s in segments),
            'actions_match_server': all(s['action'] == decisions[i]['action'] for i, s in enumerate(segments)),
            'filter_respected': all(d['filter_applied'] and d['action'] in d['allowed_actions'] for d in decisions),
            'event_clock_ordered': all(a['t_s'] <= b['t_s'] for a, b in zip(data['events'], data['events'][1:])),
            'events_within_capture': all(0 <= e['t_s'] <= metrics['wall_time_s'] for e in data['events']),
            'transport_quiescent': not data['transport_audit']['in_flight'],
            'decoded_if_completed': not data['completed'] or metrics['total_frames'] > 0,
            'full_media_if_completed': not data['completed'] or abs(metrics['played_s'] - dataset.video.durations_s.sum()) < .2,
            'memory_observed': bool(memory['samples']) and all(s['chrome_process_tree_rss_bytes'] is not None for s in memory['samples']),
        }
        checks = {k: bool(v) for k, v in checks.items()}
        native.append({'case': case['name'], 'cap_s': cap, 'trace': context['trace'], 'policy': context['policy'],
                       'completed': data['completed'], 'metrics_scope': 'full_playback' if data['completed'] else 'partial_playback',
                       'error': data['error'], 'metrics': metrics, 'capture_checks': checks,
                       'reported_metrics': reported_metrics,
                       'timing_source': 'native playing/waiting event timestamps; raw recorder counters preserved',
                       'reported_stall_counter_difference_s': reported_metrics['rebuffer_s'] - stalls,
                       'reported_stall_counter_within20ms': abs(reported_metrics['rebuffer_s'] - stalls) < .02,
                       'capture_valid': all(checks.values()), 'server_media_bytes_sent': sent,
                       'discarded_unappended_server_media_bytes': discarded,
                       'estimated_unplayed_appended_bytes': unplayed,
                       'baseline_chrome_rss_bytes': memory['baseline_rss_bytes'],
                       'sampled_peak_chrome_rss_bytes': memory['peak_rss_bytes'],
                       'sampled_peak_javascript_heap_bytes': memory['peak_javascript_heap_bytes'],
                       'targets_met': bool(data['completed'] and metrics['startup_s'] is not None and
                                           metrics['startup_s'] <= 3. and metrics['rebuffer_s'] <= 8.)})
    simulator_gate = all(e['playback_targets_met'] for r in simulator['results'] if r['cap_s'] == 60. and
                         r['policy'] in ('dqn', 'highest_allowed') for e in r['episodes'][:4])
    native_gate = all(n['capture_valid'] and n['targets_met'] for n in native if n['cap_s'] == 60.)
    summary = {'frozen_checkpoint_sha256': protocol['checkpoint_sha256'], 'simulator_60_new_sessions_gate': simulator_gate,
               'native_60_gate': native_gate, 'all_native_captures_valid': all(n['capture_valid'] for n in native),
               'target_device_memory_budget_supplied': False, 'training_performed': False,
               'decision': 'Do not retrain or promote: the predeclared playback gates did not pass. '
                           'Memory measurements are descriptive; no device budget was supplied.',
               'simulator': simulator['results'], 'native': native,
               'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                   ('abr/player_server.py', 'experiments/validation/policy-player.mjs',
                    'scripts/capture_policy_player.mjs', 'scripts/evaluate_prefetch.py', 'scripts/report_prefetch.py')},
               'native_capture_source': json.loads((OUT / 'native' / 'capture-source.json').read_text())}
    if simulator_gate and native_gate:
        raise RuntimeError('Playback unexpectedly passed; review the conditional training step and device memory budget')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# Frozen 30/60-second prefetch validation', '', summary['decision'], '',
             'Settings, four new session filenames, native cases and candidate training budgets were saved '
             'before downloading/replaying the new traces. Both caps keep startup/resume thresholds at 1 s. '
             'Frozen DQN and highest-allowed use the same observed-goodput safety filter.', '',
             '## Simulator: four new sessions and one known control', '',
             'Each row covers a full video. Backlog is an estimate of compressed segment bytes that remain '
             'in the playable queue, counting the whole partially played head segment. It is not allocated '
             'browser RAM and excludes decoded frames, source-buffer overhead and in-flight requests.', '',
             '| Cap s | Policy | Session | Startup s | Stalls s | Downloaded MiB | Discarded KiB | Peak encoded backlog MiB | Targets met |',
             '|---:|---|---|---:|---:|---:|---:|---:|---|']
    for r in simulator['results']:
        for e in r['episodes']:
            m = e['metrics']
            lines.append(f"| {r['cap_s']:.0f} | {r['policy']} | {e['trace']} | {m['startup_s']:.3f} | "
                         f"{m['rebuffer_s']:.3f} | {m['downloaded_bytes']/2**20:.3f} | {m['wasted_bytes']/1024:.3f} | "
                         f"{m['estimated_peak_encoded_whole_segments_bytes']/2**20:.3f} | {e['playback_targets_met']} |")
    lines.extend(['', '## Native Chrome: completion, timing and actual process memory', '',
                  'Eight isolated headless Chrome instances ran in the same capture batch. RSS is sampled '
                  'every 2 s across each browser process tree and can count shared pages multiple times. '
                  'The empty-browser baseline and sampled peak include browser/runtime/codecs; the '
                  'difference is not an isolated measurement of buffer allocation. JS heap excludes '
                  'most native decoder/MSE storage. Python inference-server memory is not included.', '',
                  '| Case | Complete | Startup s | Stalls s | Media sent MiB | Discarded unappended KiB | Baseline RSS MiB | Sampled peak RSS MiB | Peak JS heap MiB | Targets met |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---|'])
    for n in native:
        m = n['metrics']
        startup = 'none' if m['startup_s'] is None else f"{m['startup_s']:.3f}"
        lines.append(f"| {n['case']} | {n['completed']} | {startup} | {m['rebuffer_s']:.3f} | "
                     f"{n['server_media_bytes_sent']/2**20:.3f} | {n['discarded_unappended_server_media_bytes']/1024:.3f} | "
                     f"{n['baseline_chrome_rss_bytes']/2**20:.1f} | {n['sampled_peak_chrome_rss_bytes']/2**20:.1f} | "
                     f"{n['sampled_peak_javascript_heap_bytes']/2**20:.2f} | {n['targets_met']} |")
    lines.extend(['', 'Startup/stall timings in this report are reconstructed from the archived native '
                  'event timestamps. Two original recorder stall counters differed from their event '
                  'logs by about 40 ms because handlers read the clock separately for logging and '
                  'counters. Raw captures/counters remain unchanged; discrepancies are explicit in '
                  'summary.json. A failing handler-clock test reproduced the problem, and the recorder '
                  'now uses one timestamp per event. The timing acceptance limits were not widened.', '',
                  'Failed native runs remain partial. Their stall totals do not describe full-video '
                  'performance. Media bytes above are server-sent bytes; receipt of a failed response '
                  'by the client cannot be proved from this audit. Prefetched but unplayed appended '
                  'bytes are separately estimated in summary.json. Initialization traffic is excluded.', '',
                  '## Gate and next step', '',
                  f'Simulator 60 s gate on new sessions: **{simulator_gate}**. '
                  f'Native 60 s gate: **{native_gate}**. No retraining was started. The fixed seed '
                  'budget in protocol.json is a conditional plan, not an executed training result.', '',
                  'The native player still fails requests after 30 s and has no recovery. A larger '
                  'buffer can protect already-buffered playback while a request is slow, but cannot '
                  'make this player resume after its request loop exits. Port and validate bounded '
                  'request recovery before using native playback results to justify retraining. '
                  'Slow first delivery also needs separate startup treatment; increasing the prefetch '
                  'cap cannot download the first segment sooner.', '',
                  'Independent browser runs can follow different request timing and measurements, '
                  'even when policies would agree on identical input. The DQN/rule difference on the '
                  'hard control is not proof of a neural advantage from one trial per case.', '',
                  'Keep default cap and frozen weights unchanged. No memory acceptance claim is made '
                  'without a target-device budget. These are four more sessions on the same old route '
                  'and one silent video; there is no new-route/video generalization claim.', '',
                  'Reproduce simulator: `.venv/bin/python -m scripts.evaluate_prefetch --prepare` then '
                  '`.venv/bin/python -m scripts.evaluate_prefetch`. Native commands and source/config '
                  'identity are documented in docs/prefetch-validation.md.', ''])
    (OUT / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('simulator', 'native', 'source_sha256')}))
    if not summary['all_native_captures_valid']:
        raise SystemExit('Native audit failed; inspect capture_checks before interpreting outcomes')


if __name__ == '__main__':
    main()
