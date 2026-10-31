"""Separate verified production-target facts from proposed acceptance limits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import plistlib
import subprocess

OUT = Path('runs/production-readiness')
PROFILE = Path('experiments/configs/production_desktop_v1.json')


def inspect_hardware():
    values = subprocess.check_output(['sysctl', '-n', 'hw.model', 'hw.memsize', 'machdep.cpu.brand_string'], text=True).splitlines()
    browser = plistlib.loads(Path('/Applications/Google Chrome.app/Contents/Info.plist').read_bytes())
    hardware = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'model_identifier': values[0],
                'system_ram_bytes': int(values[1]), 'processor': values[2], 'architecture': platform.machine(),
                'os_version': subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip(),
                'browser_version': browser['CFBundleShortVersionString'],
                'method': 'Current sysctl hardware and installed browser plist; archived captures independently record their browser/platform'}
    (OUT / 'hardware.json').write_text(json.dumps(hardware, indent=2) + '\n')


def quality_exposure(capture, floor_kbps):
    indices = [row['segment_index'] for row in capture['segments']]
    if indices != list(range(len(capture['context']['durations_s']))):
        raise ValueError('Exposure audit requires each media segment exactly once, in playback order')
    duration = float(sum(capture['context']['durations_s']))
    emergency = streak = longest = 0.
    dimensions = set()
    for row in capture['segments']:
        seconds = float(capture['context']['durations_s'][row['segment_index']])
        rep = capture['context']['representations'][row['action']]
        if row['bitrate_kbps'] < floor_kbps:
            emergency += seconds
            streak += seconds
            longest = max(longest, streak)
            dimensions.add((rep.get('width'), rep.get('height')))
        else:
            streak = 0.
    return {'duration_s': duration, 'emergency_media_s': emergency,
            'emergency_fraction': emergency / duration, 'longest_continuous_emergency_s': longest,
            'emergency_dimensions': [list(x) for x in sorted(dimensions)],
            'scope': 'Complete captured playback; encoded media duration, not wall time or visual opinion'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-hardware', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.inspect_hardware:
        inspect_hardware()
    profile = json.loads(PROFILE.read_text())
    hardware = json.loads((OUT / 'hardware.json').read_text())
    server = json.loads((OUT / 'server-memory.json').read_text())
    study = Path('runs/startup-ladder')
    observations = []
    for directory in ('native', 'sintel-native'):
        root = study / directory
        for capture_path in sorted(root.glob('*.json')):
            if capture_path.stem in ('summary', 'browser') or capture_path.stem.endswith('-memory'):
                continue
            capture = json.loads(capture_path.read_text())
            if not capture['completed']:
                raise ValueError('Exposure audit must not silently treat partial playback as a full video')
            memory = json.loads(capture_path.with_name(capture_path.stem + '-memory.json').read_text())
            q = quality_exposure(capture, profile['quality_budget']['normal_minimum']['nominal_bitrate_kbps'])
            quality_pass = (q['longest_continuous_emergency_s'] <= profile['quality_budget']['maximum_continuous_emergency_media_s'] + 1e-9 and
                            q['emergency_fraction'] <= profile['quality_budget']['maximum_total_emergency_fraction'] + 1e-9)
            observations.append({'case': capture_path.stem, 'capture': str(capture_path),
                                 'capture_sha256': hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                                 'browser_sampled_peak_rss_bytes': memory['peak_rss_bytes'],
                                 'javascript_sampled_peak_heap_bytes': memory['peak_javascript_heap_bytes'],
                                 'quality_exposure': q, 'meets_proposed_quality_limits': quality_pass})
    browser_peak = max(x['browser_sampled_peak_rss_bytes'] for x in observations)
    js_peak = max(x['javascript_sampled_peak_heap_bytes'] for x in observations)
    worker_peak = server['sampled_process_lifetime_peak_rss_bytes']
    memory = {'browser_peak_rss_bytes': browser_peak, 'javascript_peak_heap_bytes': js_peak,
              'worker_probe_peak_rss_bytes': worker_peak,
              'sum_of_separately_observed_component_peaks_bytes': browser_peak + worker_peak,
              'fits_proposed_snapshot_budgets': bool(browser_peak <= profile['memory_budget']['browser_tree_rss_limit_bytes'] and
                  worker_peak <= profile['memory_budget']['inference_worker_rss_limit_bytes'] and
                  browser_peak + worker_peak <= profile['memory_budget']['combined_rss_limit_bytes'] and
                  js_peak <= profile['memory_budget']['javascript_heap_limit_bytes']),
              'combined_full_playback_measured': False, 'soak_completed': False,
              'status': 'Snapshot compatibility only; not memory approval or proof of no leak'}
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'profile_status': profile['status'],
              'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(), 'hardware': hardware,
              'memory_assessment': memory, 'worker_probe': server, 'quality_cases': observations,
              'proposed_quality_limits_passed': sum(x['meets_proposed_quality_limits'] for x in observations),
              'quality_cases_count': len(observations), 'human_quality_accepted': False,
              'production_approved': False,
              'decision': 'Reference target identified; draft budgets defined. See quality-case outcomes and memory snapshots; '
                          'full-stack/soak and human quality acceptance remain unverified.',
              'runtime_or_checkpoint_changes': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'target': hardware, 'memory': memory,
                      'quality_passed': result['proposed_quality_limits_passed'],
                      'quality_total': len(observations), 'production_approved': False}))


if __name__ == '__main__':
    main()
