"""Audit native recovery attempts, actual decoding, and startup feasibility."""
import hashlib
import json
from pathlib import Path

from abr.data import Dataset
from scripts.audit_player_captures import event_stalls

OUT = Path('runs/native-recovery')


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    data = Dataset.load(protocol['dataset'])
    cases = json.loads((OUT / 'cases.json').read_text())
    rows = []
    for case in cases:
        capture = json.loads((OUT / 'captures' / (case['name'] + '.json')).read_text())
        memory = json.loads((OUT / 'captures' / (case['name'] + '-memory.json')).read_text())
        context, metrics = capture['context'], capture['metrics']
        attempts, segments = capture['download_attempts'], capture['segments']
        requests = {str(r['request_id']): r for r in capture['transport_audit']['requests']}
        decisions = capture['transport_audit']['decisions']
        stalls, _ = event_stalls(capture['events'], metrics['wall_time_s'])
        first_playing = next((e['t_s'] for e in capture['events'] if e['type'] == 'playing'), None)
        expected_recovery = case['port'] == 8851
        retries = sum(a['attempt'] > 0 for a in attempts)
        wasted = sum(a['bytes_received'] for a in attempts if a['outcome'] != 'complete')
        received = sum(a['bytes_received'] for a in attempts)
        grouped = {}
        for attempt in attempts:
            grouped.setdefault(attempt['segment_index'], []).append(attempt)
        checks = {
            'native_capture': capture['origin'] == 'native_mse_policy_capture',
            'frozen_checkpoint': context['checkpoint_sha256'] == protocol['checkpoint_sha256'],
            'dataset': context['dataset_sha256'] == data.fingerprint,
            'trace': context['trace_sha256'] == data.splits['test'][case['trace']].fingerprint,
            'experimental_cap': context['config']['max_buffer_s'] == 60. and context['training_config']['max_buffer_s'] == 30.,
            'recovery_context': context['request_recovery'] == (protocol['recovery'] if expected_recovery else False),
            'visible': bool(capture['samples']) and all(s['visibility'] == 'visible' for s in capture['samples']),
            'startup_matches_events': first_playing is None and metrics['startup_s'] is None or
                                      first_playing is not None and abs(first_playing - metrics['startup_s']) < 1e-6,
            'stalls_match_events': abs(stalls - metrics['rebuffer_s']) < .02,
            'received_byte_accounting': received == metrics['received_media_bytes'],
            'discarded_byte_accounting': wasted == metrics['discarded_media_bytes'],
            'retry_accounting': retries == metrics['request_retries'],
            'retry_limits': all(len(group) <= (5 if expected_recovery else 1) and
                               [a['attempt'] for a in group] == list(range(len(group))) for group in grouped.values()),
            'retries_use_lowest': all(a['action'] == 0 for a in attempts if a['attempt'] > 0),
            'client_bytes_within_server_sent': all(a['bytes_received'] <= requests[a['request_id']]['bytes_sent']
                                                 for a in attempts if a['request_id'] is not None),
            'one_append_per_ordered_segment': [s['segment_index'] for s in segments] == list(range(len(segments))),
            'only_complete_bodies_appended': all(s['attempts'][-1]['outcome'] == 'complete' and
                s['action'] == s['attempts'][-1]['action'] and s['bytes'] == s['attempts'][-1]['bytes_received'] and
                s['bytes'] == data.video.sizes_bytes[s['segment_index'], s['action']] for s in segments),
            'policy_choices_match_server': all(s['policy_action'] == decisions[i]['action'] for i, s in enumerate(segments)),
            'policy_filter_respected': all(d['filter_applied'] and d['action'] in d['allowed_actions'] for d in decisions),
            'real_decode_and_full_media_if_complete': not capture['completed'] or
                metrics['total_frames'] > 0 and abs(metrics['played_s'] - data.video.durations_s.sum()) < .2 and len(segments) == 49,
            'transport_quiescent': not capture['transport_audit']['in_flight'],
            'memory_observed': bool(memory['samples']) and all(s['chrome_process_tree_rss_bytes'] is not None for s in memory['samples']),
        }
        checks = {k: bool(v) for k, v in checks.items()}
        rows.append({'case': case['name'], 'recovery': expected_recovery, 'trace': context['trace'],
                     'policy': context['policy'], 'completed': capture['completed'], 'error': capture['error'],
                     'metrics_scope': 'full_playback' if capture['completed'] else 'partial_playback',
                     'metrics': metrics, 'checks': checks, 'capture_valid': all(checks.values()),
                     'early_abandons': sum(a['outcome'] in ('no_progress', 'buffer_risk') for a in attempts),
                     'peak_browser_rss_bytes': memory['peak_rss_bytes'],
                     'targets_met': bool(capture['completed'] and metrics['startup_s'] <= 3 and metrics['rebuffer_s'] <= 8)})
    report = {'protocol': protocol, 'cases': rows, 'startup_analysis': json.loads((OUT / 'startup-analysis.json').read_text()),
              'source_at_capture': json.loads((OUT / 'source-at-capture.json').read_text()),
              'all_capture_audits_pass': all(r['capture_valid'] for r in rows), 'model_retrained': False,
              'defaults_changed': False, 'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Native VOD recovery validation', '',
             'Frozen weights and explicit 60-second cap. Previously observed development cases; '
             'startup/resume thresholds remain 1/1 s. Six recovery-on trials and two matched recovery-off '
             'hard controls were fixed before replay. No model training or default cap change.', '',
             '## Implemented behavior', '',
             'The browser reads streamed body chunks and checks progress while a read is blocked. '
             'It can abandon a slow higher-quality request and restart the same segment at lowest quality. '
             'At most four retries per segment; initialization and media requests each have a 30 s deadline. '
             'Timeout/network/no-progress failures back off 2 s; buffer-risk abandonment does not. '
             'Initialization failures consume an attempt too. Whole failed media bodies are discarded. '
             'Only a complete body is appended to MediaSource, so retries cannot mix partial encodings.', '',
             'A cancellation endpoint interrupts the HTTP pacer and acknowledges release of the session '
             'request lock before the browser retries. Attempt IDs connect client-received bytes with '
             'server audit records. History and previous bitrate reflect the effective representation, '
             'with received bytes from failed attempts included in goodput. Recovery is opt-in using '
             '`--enable-request-recovery`; existing default mode and checkpoint remain unchanged.', '',
             '## Native results', '',
             'Timing uses the common event clock. Partial runs remain labelled partial; their metrics '
             'do not describe full-video performance. RSS is the sampled whole Chrome process-tree sum, '
             'including runtime/codecs and potentially duplicate shared pages, not isolated buffer RAM.', '',
             '| Case | Complete | Startup s | Stalls s | Retries | Discarded media KiB | Sampled RSS MiB | Targets met |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        m = r['metrics']
        startup = 'none' if m['startup_s'] is None else f"{m['startup_s']:.3f}"
        lines.append(f"| {r['case']} | {r['completed']} | {startup} | {m['rebuffer_s']:.3f} | "
                     f"{m['request_retries']} | {m['discarded_media_bytes']/1024:.3f} | "
                     f"{r['peak_browser_rss_bytes']/2**20:.1f} | {r['targets_met']} |")
    lines.extend(['', 'Client-received media bytes and discarded failed-body bytes are measured from '
                  'ReadableStream chunks. Initialization payloads are recorded separately. This differs '
                  'from the earlier server-sent estimate, which cannot prove failed-byte receipt. '
                  'Independent native sessions can have different request timing and observations; '
                  'a single DQN/rule outcome difference does not prove a neural-policy advantage.', '',
                  '## Slow-start finding', '',
                  'Before first playback, the policy already uses the lowest representation. The '
                  'following is an offline diagnostic with exact smallest segment sizes, no initialization '
                  'traffic and zero RTT. It is a favorable lower bound, never an input to the policy.', '',
                  '| Session | Smallest first segment bytes | Service available by 3 s, bytes | Minimum startup s | 3 s target impossible for these assets |',
                  '|---|---:|---:|---:|---|'])
    for r in report['startup_analysis']:
        lines.append(f"| {r['trace']} | {r['smallest_first_segment_bytes']:.0f} | {r['service_in_first3_s_bytes']:.0f} | "
                     f"{r['minimum_startup_s_even_zero_overhead']:.3f} | {r['startup_target_impossible_for_current_assets']} |")
    lines.extend(['', 'The two slow-start cases cannot reach first playback by 3 s with the current '
                  'whole-segment media contract, even without request overhead. Recovery can preserve '
                  'completion but cannot create missing bytes. The UI now shows first-download byte '
                  'progress and explicitly records when the startup target is exceeded; it does not '
                  'restart the clock or hide waiting.', '',
                  'Meeting the startup limit needs a genuinely smaller independently playable startup '
                  'unit/representation, or a different service condition. Re-encoding/packaging must '
                  'preserve decoder validity and be measured against the ~8–9 KB service budgets; '
                  'simply labelling a rung lower or changing the reward is insufficient.', '',
                  '## Decision', '',
                  'Keep the model frozen. Recovery and startup feasibility are separate findings. '
                  'Do not retrain or claim all playback targets are met while startup remains infeasible '
                  'for these assets. Next evaluate actual smaller startup media, with exact sizes and '
                  'real decoding, rather than further tuning on these same traces.', '',
                  'Reproduce using commands in docs/native-recovery.md. Raw attempts, events, '
                  'transport audits, memory samples and screenshots are retained under captures/.', ''])
    (OUT / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'captures_valid': report['all_capture_audits_pass'],
                      'completed': sum(r['completed'] for r in rows), 'cases': len(rows),
                      'on_completed': sum(r['completed'] for r in rows if r['recovery']),
                      'on_cases': sum(r['recovery'] for r in rows),
                      'retries': sum(r['metrics']['request_retries'] for r in rows)}))
    if not report['all_capture_audits_pass']:
        raise SystemExit('Native recovery capture audit failed; inspect checks')


if __name__ == '__main__':
    main()
