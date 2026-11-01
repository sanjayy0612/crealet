"""Score frozen transfer decisions with frame-aligned, per-rung VMAF."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/configs/single_task_vod_v1.json'
TRANSFER = ROOT / 'runs/startup-ladder/transfer.json'
OUTPUT = ROOT / 'runs/single-task-vod-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-4000:]}")
    return result


def ffmpeg_info(binary):
    version = run([binary, '-version']).stdout.splitlines()[0]
    filters = run([binary, '-filters']).stdout
    if 'libvmaf' not in filters:
        raise RuntimeError(f'{binary} does not expose the libvmaf filter')
    return version, sha256(binary)


def video_dimensions(source):
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('ffprobe is required to record the reference dimensions')
    payload = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height', '-of', 'json', str(source)]).stdout)
    streams = payload.get('streams', [])
    if len(streams) != 1 or streams[0].get('width', 0) <= 0 or streams[0].get('height', 0) <= 0:
        raise RuntimeError(f'Could not read video dimensions from {source}')
    return int(streams[0]['width']), int(streams[0]['height'])


def vmaf_frames(binary, media_name, media_dir, source, source_fps, reps, durations, output_dir, needed_rungs):
    frames_by_rung = {}
    media_dir = ROOT / media_dir
    source_width, source_height = video_dimensions(source)
    for index, rep in enumerate(reps):
        if index not in needed_rungs:
            continue
        bitrate = int(rep['bitrate_kbps'])
        log_path = output_dir / f'{media_name}-r{bitrate}-vmaf.json'
        if not log_path.exists():
            reference = f'setpts=N/({source_fps}*TB),fps=25,settb=AVTB,setpts=PTS-STARTPTS'
            # VMAF must include resolution loss. Upscale the encoded rung to the
            # source resolution; downscaling the reference would hide that loss.
            distorted = (f'fps=25,scale={source_width}:{source_height}:flags=bicubic,'
                         'settb=AVTB,setpts=PTS-STARTPTS')
            graph = (f'[0:v]{distorted}[dist];[1:v]{reference}[ref];'
                     f'[dist][ref]libvmaf=log_fmt=json:log_path={log_path}:pool=mean:n_threads=4')
            encoded = media_dir / f'encoded-{bitrate}.mp4'
            run([binary, '-nostdin', '-hide_banner', '-loglevel', 'error',
                 '-i', str(encoded), '-i', str(source), '-filter_complex', graph,
                 '-an', '-f', 'null', '-'])
        payload = json.loads(log_path.read_text())
        frame_rows = payload.get('frames')
        if not isinstance(frame_rows, list) or not frame_rows:
            raise RuntimeError(f'VMAF log has no frame scores: {log_path}')
        values = []
        for row in frame_rows:
            value = row.get('metrics', {}).get('vmaf')
            if value is None:
                raise RuntimeError(f'Missing default VMAF score in {log_path}')
            values.append(float(value))
        expected_frames = round(sum(durations) * 25)
        if len(values) != expected_frames:
            raise RuntimeError(f'{media_name} r{bitrate}: got {len(values)} VMAF frames, expected {expected_frames}')
        frames_by_rung[index] = values
    return frames_by_rung


def segments(durations, fps=25):
    result, cursor = [], 0
    for duration in durations:
        count = round(duration * fps)
        result.append((cursor, cursor + count, float(duration)))
        cursor += count
    return result


def episode_quality(actions, frame_scores, segment_ranges):
    if len(actions) > len(segment_ranges):
        raise RuntimeError('Decision count exceeds media segment count')
    weighted, weight = 0., 0.
    for index, action in enumerate(actions):
        start, end, _ = segment_ranges[index]
        scores = frame_scores[int(action)][start:end]
        if len(scores) != end - start:
            raise RuntimeError(f'VMAF frame range is incomplete at segment {index}')
        weighted += sum(scores) / len(scores) * (end - start) / 25
        weight += (end - start) / 25
    return {'playback_duration_weighted_vmaf': weighted / weight if weight else None,
            'scored_media_duration_s': weight,
            'scored_segments': len(actions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--transfer-report', type=Path, default=TRANSFER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cache-dir', type=Path,
                        help='Reuse VMAF frame logs from an existing cache directory')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    transfer = json.loads(args.transfer_report.read_text())
    transfer_reports = [(args.transfer_report, transfer.get('results', []))]
    for relative_path in protocol.get('additional_transfer_reports', []):
        report_path = ROOT / relative_path
        if not report_path.is_file():
            raise RuntimeError(f'Missing frozen baseline results: {report_path}')
        payload = json.loads(report_path.read_text())
        transfer_reports.append((report_path, payload.get('reports', payload.get('results', payload.get('policies', [])))))
    for policy in protocol['policies']:
        if 'checkpoint' not in policy:
            continue
        checkpoint = ROOT / policy['checkpoint']
        if not checkpoint.is_file() or sha256(checkpoint) != policy['checkpoint_sha256']:
            raise RuntimeError(f"Frozen checkpoint identity mismatch for {policy['name']}: {checkpoint}")
    binary = shutil.which('ffmpeg')
    if not binary:
        raise RuntimeError('ffmpeg is required')
    version, binary_hash = ffmpeg_info(binary)
    args.output.mkdir(parents=True, exist_ok=True)
    media_cache = args.cache_dir or (args.output / 'vmaf-cache-used-rungs-threads4-v1')
    media_cache.mkdir(exist_ok=True)

    policy_names = ({item['name'] for item in protocol['policies']}
                    | {item['name'] for item in protocol.get('baselines', [])})
    policy_names |= {
        'dqn_120_seed_7', 'dqn_120_seed_11', 'dqn_120_seed_13',
        'dqn_240_seed_7', 'dqn_240_seed_11', 'dqn_240_seed_13',
        'highest_allowed_ghent', 'robust_mpc',
        'controlled_dqn_seed_7', 'controlled_dqn_seed_11',
        'controlled_dqn_seed_13', 'controlled_highest_allowed',
        'controlled_robust_mpc',
        'controlled_v2_original_seed_7', 'controlled_v2_original_seed_11',
        'controlled_v2_original_seed_13', 'controlled_v2_trained_seed_7',
        'controlled_v2_trained_seed_11', 'controlled_v2_trained_seed_13',
        'controlled_v2_highest_allowed', 'controlled_v2_robust_mpc',
    }
    scored = {}
    media_hashes = {}
    for media in protocol['media']:
        name = media['name']
        needed_rungs = set()
        for _, records in transfer_reports:
            for report in records:
                if report['content'] != name or report['policy'] not in policy_names:
                    continue
                for episode in report['episodes']:
                    actions = episode.get('effective_actions') or episode.get('policy_actions') or []
                    needed_rungs.update(int(action) for action in actions)
        dataset_path = ROOT / media['dataset']
        dataset = json.loads(dataset_path.read_text())
        durations = [float(value) for value in dataset['video']['durations_s']]
        bitrates = [float(value) for value in dataset['video']['bitrates_kbps']]
        index_path = ROOT / media['media_dir'] / 'index.json'
        media_index = json.loads(index_path.read_text())
        reps = sorted(media_index['representations'], key=lambda rep: rep['bitrate_kbps'])
        if [float(rep['bitrate_kbps']) for rep in reps] != bitrates:
            raise RuntimeError(f'{name}: media ladder and dataset bitrates differ')
        source = (ROOT / 'artifacts/startup-ladder-v1/source.mp4' if name == 'envivio'
                  else ROOT / 'artifacts/sintel-trailer-480p.mp4')
        source_fps = 25 if name == 'envivio' else 24
        scores = vmaf_frames(binary, name, media['media_dir'], source, source_fps,
                             reps, durations, media_cache, needed_rungs)
        scored[name] = {'frames': scores, 'segments': segments(durations)}
        media_hashes[name] = {'dataset': sha256(dataset_path), 'source': sha256(source),
                              'index': sha256(index_path),
                              'encodes': {str(int(rep['bitrate_kbps'])): sha256(ROOT / media['media_dir'] / f"encoded-{int(rep['bitrate_kbps'])}.mp4") for rep in reps}}
        print(json.dumps({'media': name, 'rungs_scored': len(needed_rungs),
                          'frames_per_rung': len(next(iter(scores.values())))}), flush=True)

    rows = []
    for _, records in transfer_reports:
        for report in records:
            if report['policy'] not in policy_names:
                continue
            name = report['content']
            frame_scores = scored[name]['frames']
            for episode in report['episodes']:
                actions = episode.get('effective_actions') or episode.get('policy_actions') or []
                quality = episode_quality(actions, frame_scores, scored[name]['segments'])
                rows.append({'content': name, 'policy': report['policy'], 'trace': episode['trace'],
                             'completed': episode['completed'], 'termination_reason': episode['termination_reason'],
                             'reward_proxy': episode['reward'],
                             'mean_selected_bitrate_kbps': episode['metrics'].get('mean_played_bitrate_kbps'),
                             'startup_s': episode['metrics'].get('startup_s'),
                             'rebuffer_s': episode['metrics'].get('rebuffer_s'), **quality})

    summaries = []
    for content in sorted({row['content'] for row in rows}):
        for policy in sorted({row['policy'] for row in rows if row['content'] == content}):
            cases = [row for row in rows if row['content'] == content and row['policy'] == policy]
            vmaf_values = [row['playback_duration_weighted_vmaf'] for row in cases if row['playback_duration_weighted_vmaf'] is not None]
            summaries.append({'content': content, 'policy': policy, 'sessions': len(cases),
                              'completed': sum(row['completed'] for row in cases),
                              'mean_case_vmaf': sum(vmaf_values) / len(vmaf_values) if vmaf_values else None,
                              'mean_case_reward_proxy': sum(row['reward_proxy'] for row in cases) / len(cases),
                              'mean_case_bitrate_kbps': sum(row['mean_selected_bitrate_kbps'] for row in cases) / len(cases)})

    result = {'protocol_id': protocol['protocol_id'], 'ffmpeg_version': version,
              'ffmpeg_sha256': binary_hash, 'vmaf_model': 'FFmpeg libvmaf default vmaf_v0.6.1',
              'vmaf_threads': 4,
              'vmaf_pool': 'upscale distorted rung to source resolution; mean frames within each aligned segment; then weight by segment media duration',
              'transfer_report_sha256': {str(path.relative_to(ROOT)): sha256(path) for path, _ in transfer_reports},
              'protocol_sha256': sha256(args.protocol),
              'evaluator_sha256': sha256(Path(__file__)), 'media_sha256': media_hashes,
              'case_results': rows, 'summaries': summaries,
              'limitations': protocol.get('limitations', [
                  'Historical transfer traces from a single ferry route.',
                  'VMAF scores measure encoded-frame quality, not viewer acceptance.',
                  'Existing simulator action sequences are scored; no playback or policy was rerun.'])}
    (args.output / 'vmaf_transfer.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Single-task VOD transfer quality screen', '',
             f"FFmpeg: `{version}`", '',
             'This report scores the frozen simulator action sequences from the existing transfer report. '
             'It does not claim browser equivalence or broad network generalization.', '',
             '| Content | Policy | Completed | Mean case VMAF | Mean bitrate kbps | Mean log-bitrate reward |',
             '|---|---|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['content']} | {row['policy']} | {row['completed']}/{row['sessions']} | "
                     f"{row['mean_case_vmaf']:.3f} | {row['mean_case_bitrate_kbps']:.1f} | {row['mean_case_reward_proxy']:.3f} |")
    lines.extend(['', 'VMAF is a quality metric for these aligned encoded frames. It is not a human acceptance test.',
                  'Per-session values and source hashes are in `vmaf_transfer.json`.'])
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'case_rows': len(rows), 'output': str(args.output / 'report.md')}))


if __name__ == '__main__':
    main()
