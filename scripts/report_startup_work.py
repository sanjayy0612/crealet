"""Summarize the media fix, native playback, matched training and transfer evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

from abr.data import Dataset
from abr.diagnostics import optimistic_arrival_bound, service_bytes
from scripts.audit_player_captures import event_stalls

OUT = Path('runs/startup-ladder')


def main():
    gate = json.loads((OUT / 'frozen-native-gate.json').read_text())
    transfer = json.loads((OUT / 'transfer.json').read_text())
    provenance = json.loads(Path('data/startup-ladder/provenance.json').read_text())
    sintel = Dataset.load('data/sintel-ladder/development.json')
    native = []
    for case in json.loads((OUT / 'sintel-cases.json').read_text()):
        capture = json.loads((OUT / 'sintel-native' / (case['name'] + '.json')).read_text())
        m, c = capture['metrics'], capture['context']
        stalls, _ = event_stalls(capture['events'], m['wall_time_s'])
        checks = {'completed': capture['completed'], 'real_decode': m['total_frames'] == 1305,
                  'full_duration': abs(m['played_s'] - sum(sintel.video.durations_s)) < .1,
                  'same_frozen_checkpoint': c['checkpoint_sha256'] == json.loads((OUT / 'protocol.json').read_text())['checkpoint_sha256'],
                  'dataset_matches': c['dataset_sha256'] == sintel.fingerprint,
                  'segment_bytes_match': all(s['bytes'] == sintel.video.sizes_bytes[s['segment_index'], s['action']] for s in capture['segments']),
                  'all_segments': len(capture['segments']) == len(sintel.video.durations_s),
                  'event_stalls_match': abs(stalls - m['rebuffer_s']) < .02,
                  'startup_and_stalls': m['startup_s'] <= 3 and m['rebuffer_s'] <= 8}
        native.append({'case': case['name'], 'metrics': m, 'checks': {k: bool(v) for k, v in checks.items()},
                       'all_checks_pass': all(checks.values()),
                       'lowest_rung_fraction': sum(sintel.video.durations_s[s['segment_index']] for s in capture['segments'] if s['action'] == 0) / sum(sintel.video.durations_s)})
    new = Dataset.load('data/transfer-sessions/envivio.json')
    feasibility = []
    for trace in new.splits['test']:
        bound = optimistic_arrival_bound(new.video, trace, max_buffer_s=60.)
        feasibility.append({'trace': trace.name, 'service_by3_s_bytes': service_bytes(trace, 0, 3),
                            'first_minimum_bytes': float(new.video.sizes_bytes[0].min()),
                            'minimum_startup_s': bound['minimum_startup_s']})
    resource_rows = []
    for directory, cases in [('native', gate['cases']), ('sintel-native', native)]:
        for r in cases:
            memory = json.loads((OUT / directory / (r['case'] + '-memory.json')).read_text())
            capture = json.loads((OUT / directory / (r['case'] + '.json')).read_text())
            times = [s['decision_ms'] for s in capture['segments']]
            resource_rows.append({'case': r['case'], 'sampled_peak_chrome_rss_bytes': memory['peak_rss_bytes'],
                                  'server_decision_p95_ms': float(np.percentile(times, 95))})
    payload = {'engineering_gate': gate, 'sintel_native': native, 'transfer_comparisons': transfer['comparisons'],
               'promotion_recommended': transfer['promotion_recommended'], 'fresh_startup_feasibility': feasibility,
               'media': provenance, 'model_weights_replaced': False, 'default_cap_changed': False,
               'desktop_only': True, 'target_device_memory_budget': None, 'resources': resource_rows}
    (OUT / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# Remaining-work execution: startup media, training and broader tests', '',
             '## Outcome', '',
             'A real low-bandwidth ladder fixed startup on the known slow cases in native Chrome. '
             'Three matched DQN runs were completed, but none beat the same rule on the unused '
             'sessions. No new neural checkpoint or default buffer/media setting is promoted.', '',
             '## Real media and quality tradeoff', '',
             'The existing highest-quality Envivio source was decoded and re-encoded into seven '
             'aligned representations: 8, 300, 750, 1200, 1850, 2850 and 4300 kbps. '
             'The 8 kbps rung is only 96x54 pixels. All media sizes are actual encoded bytes; '
             'sample durations come from MP4 boxes and every generated stream passed strict FFmpeg decode. '
             'Decoded-frame PTS were normalized, avoiding the old source timestamp irregularities. '
             'The original inputs and checkpoints remain intact.', '',
             f"The new first fragment is **{provenance['recipes'][0]['first_media_bytes']} bytes** with "
             f"**{provenance['recipes'][0]['initialization_bytes']} bytes** of initialization. "
             'The old smallest first fragment was 181,801 bytes. This is an encoding/ladder improvement, '
             'not evidence that DQN learned a better startup decision.', '',
             '## Native desktop playback', '',
             '| Envivio case | Startup s | Stalls s | Completed |', '|---|---:|---:|---|']
    for r in gate['cases']:
        lines.append(f"| {r['case']} | {r['metrics']['startup_s']:.3f} | {r['metrics']['rebuffer_s']:.3f} | {r['completed']} |")
    lines.extend(['', 'All six cases passed the predeclared completion, startup <=3 s and stalls <=8 s gates. '
                  'They use a frozen model trained on the old media, an explicit 60 s cap, and recovery. '
                  'Every complete Envivio run decoded 4,842 frames. These are known development cases.', '',
                  'The second real content is the [Sintel trailer](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4), '
                  'copyright Blender Foundation / Durian Open Movie Team, '
                  '[CC BY 3.0](https://durian.blender.org/about/). It was re-encoded video-only, '
                  'normalized from 24 to 25 fps, into the same nominal ladder. Its duration is 52.2 s. '
                  'It was never used for training or checkpoint selection.', '',
                  '| Sintel case | Startup s | Stalls s | Decoded frames | Lowest-rung fraction |',
                  '|---|---:|---:|---:|---:|'])
    for r in native:
        m = r['metrics']
        lines.append(f"| {r['case']} | {m['startup_s']:.3f} | {m['rebuffer_s']:.3f} | {m['total_frames']} | {100*r['lowest_rung_fraction']:.1f}% |")
    lines.extend(['', 'The slow-session Sintel runs stayed at 8 kbps for the entire clip. Smooth playback '
                  'does not establish acceptable visual quality. Bitrate is only the reward proxy; '
                  'no VMAF/MOS or human-quality acceptance is claimed.', '',
                  'Sampled browser process-tree RSS and server decision timings are shown below. '
                  'RSS includes runtime/codecs and may count shared pages more than once. Decision '
                  'time includes the shared inference lock under parallel trials; it excludes HTTP '
                  'transport. Neither metric establishes a mobile-device budget.', '',
                  '| Case | Sampled Chrome RSS MiB | Server decision p95 ms |', '|---|---:|---:|'])
    for r in resource_rows:
        lines.append(f"| {r['case']} | {r['sampled_peak_chrome_rss_bytes']/2**20:.1f} | {r['server_decision_p95_ms']:.3f} |")
    lines.extend(['',
                  '## Matched training and fixed selection', '',
                  'Seeds 7, 11 and 13 each ran 120 episodes on the new real ladder, cap60 and recovery. '
                  'Data, safety mask, reward, replay/update budgets and validation selection were fixed '
                  'before scoring unused sessions. Checkpoints had to complete validation, start within '
                  '3 s, and cause no more stalls per trace than the rule (also <=8 s). Best original '
                  'validation reward chose the checkpoint within each seed. This is 360 training episodes, '
                  '17,640 environment steps and 4,038 optimizer updates. Final weights were not silently '
                  'substituted for the selected snapshots.', '',
                  '| Seed | Selected episode | Validation reward | Rule reward | Validation win |',
                  '|---:|---:|---:|---:|---|'])
    for r in transfer['training']:
        selected = None
        if r['selected_checkpoint']:
            with np.load(r['selected_checkpoint'], allow_pickle=False) as archive:
                selected = json.loads(archive['__metadata__'].tobytes())
        lines.append(f"| {r['seed']} | {selected['training']['episode'] if selected else 'none'} | "
                     f"{r['selected_validation_reward']:.3f} | {r['rule_validation_reward']:.3f} | {r['beats_rule_on_validation']} |")
    lines.extend(['', '## Four unused network sessions', '',
                  'All fixed checkpoints were frozen with hashes before scoring these sessions. '
                  'They are different recordings from the same historical ferry route. All policies '
                  'completed all videos with zero stalls. One session exceeded the startup gate for '
                  'every policy on both videos; it remains in the results.', '',
                  '| Content | Policy | Complete | Targets passed | Mean played kbps | Mean stalls s | Mean reward |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    for r in transfer['results']:
        s = r['summary']
        lines.append(f"| {r['content']} | {r['policy']} | {s['complete']}/4 | {s['targets_met']}/4 | "
                     f"{s['mean_bitrate_full_playbacks']:.1f} | {s['mean_stalls_full_playbacks']:.3f} | {s['mean_reward_all_rollouts']:.3f} |")
    lines.extend(['', 'Every newly trained seed had lower transfer reward than the rule on both contents. '
                  'Two validation wins did not generalize. Reward values are comparable only within '
                  'the same new ladder/content; the minimum-bitrate reference changed from the old ladder.', '',
                  '## Remaining startup bound', '',
                  '| New session | Service by 3 s, bytes | Smallest first fragment bytes | Optimistic startup s |',
                  '|---|---:|---:|---:|'])
    for r in feasibility:
        lines.append(f"| {r['trace']} | {r['service_by3_s_bytes']:.0f} | {r['first_minimum_bytes']:.0f} | {r['minimum_startup_s']:.3f} |")
    lines.extend(['', 'The remaining failing session supplies only about 1.1 KB by 3 s; the new first '
                  'fragment is still about 5.5 KB. Its roughly 10.35 s startup is an asset/service '
                  'limit under the tested contract, not a DQN-specific failure. No weights can make '
                  'missing bytes arrive. Further packaging work would need a smaller independently '
                  'playable unit and a new confirmation set; these outcomes must not be tuned away.', '',
                  '## Delivery and remaining limits', '',
                  'The desktop research player can now load explicitly changed media with measured '
                  'sizes/hashes. It retains strict default media checks, records original training '
                  'identity separately, and refuses to drop recovery required by a newly trained checkpoint. '
                  'The encoding scripts, fixed protocols, selected checkpoints and captures are retained.', '',
                  'The overall project is not production-approved. No target device or memory budget '
                  'was supplied, no mobile/browser matrix was executed, and low-rung visual quality '
                  'has not been accepted. The neural advantage remains unproven. The practical tested '
                  'reference remains the safety rule with bounded recovery and explicit media settings.', '',
                  'Commands, asset preparation and source credit are documented in docs/startup-ladder.md. '
                  'Large media payloads stay in artifacts/; tracked provenance and exact recipes rebuild them.', ''])
    (OUT / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'native_gate': gate['training_gate_passed'], 'second_video_native_pass': all(r['all_checks_pass'] for r in native),
                      'training_runs': len(transfer['training']), 'promote_new_model': transfer['promotion_recommended']}))
    if not all(r['all_checks_pass'] for r in native):
        raise SystemExit('Second-video audit failed')


if __name__ == '__main__':
    main()
