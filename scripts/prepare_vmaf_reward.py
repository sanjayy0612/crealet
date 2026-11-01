"""Build segment/rung utility matrices from frozen frame-aligned VMAF logs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from abr.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
MEDIA = {
    'envivio': {
        'dataset': 'data/startup-ladder/training.json',
        'media_dir': 'artifacts/startup-ladder-v1',
        'frame_cache': 'runs/single-task-vod-v1/vmaf-cache-used-rungs-threads4-v1',
    },
    'sintel': {
        'dataset': 'data/sintel-ladder/training.json',
        'media_dir': 'artifacts/sintel-ladder-v1',
        'frame_cache': 'runs/single-task-vod-v1/vmaf-cache-used-rungs-threads4-v1',
    },
}
OUTPUT = ROOT / 'experiments/rewards'
FPS = 25


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--content', choices=tuple(MEDIA), required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec = MEDIA[args.content]
    dataset_path = ROOT / spec['dataset']
    dataset = Dataset.load(dataset_path)
    cache = ROOT / spec['frame_cache']
    bitrates = dataset.video.bitrates_kbps
    durations = dataset.video.durations_s
    expected_frames = round(float(durations.sum()) * FPS)
    rung_frames = []
    rung_sources = {}
    for bitrate in bitrates:
        rung = int(bitrate)
        path = cache / f'{args.content}-r{rung}-vmaf.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing frame-aligned VMAF log: {path}')
        payload = json.loads(path.read_text())
        rows = payload.get('frames')
        if not isinstance(rows, list) or len(rows) != expected_frames:
            raise ValueError(f'{args.content} r{rung}: unexpected VMAF frame count')
        values = np.asarray([row['metrics']['vmaf'] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'{args.content} r{rung}: non-finite VMAF score')
        rung_frames.append(values)
        rung_sources[str(rung)] = sha256(path)

    frame_scores = np.stack(rung_frames, axis=1)
    segment_scores = np.empty((len(durations), len(bitrates)), dtype=np.float64)
    cursor = 0
    for index, duration in enumerate(durations):
        count = round(float(duration) * FPS)
        segment_scores[index] = frame_scores[cursor:cursor + count].mean(axis=0)
        cursor += count
    if cursor != expected_frames:
        raise ValueError('Segment frame boundaries do not cover the VMAF logs')

    min_vmaf, max_vmaf = float(segment_scores.min()), float(segment_scores.max())
    max_log_utility = float(np.log(bitrates[-1] / bitrates[0]))
    if max_vmaf <= min_vmaf:
        raise ValueError('VMAF ladder has no measurable utility range')
    utilities = ((segment_scores - min_vmaf) / (max_vmaf - min_vmaf)) * max_log_utility
    nonmonotonic = np.diff(segment_scores, axis=1) < 0
    output = args.output or (OUTPUT / f'{args.content}_vmaf_utility_v1.json')
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / spec['media_dir'] / 'index.json'
    source = (ROOT / ('artifacts/startup-ladder-v1/source.mp4' if args.content == 'envivio'
                      else 'artifacts/sintel-trailer-480p.mp4'))
    result = {
        'schema_version': 1,
        'content': args.content,
        'utility_name': 'segment_vmaf_linear_log_range_v1',
        'bitrates_kbps': bitrates.tolist(),
        'durations_s': durations.tolist(),
        'source_dataset': spec['dataset'],
        'source_dataset_sha256': sha256(dataset_path),
        'source_video_sha256': sha256(source),
        'media_index_sha256': sha256(index_path),
        'vmaf_model': 'FFmpeg libvmaf default vmaf_v0.6.1',
        'frame_rate': FPS,
        'frame_cache': spec['frame_cache'],
        'vmaf_frame_log_sha256': rung_sources,
        'scaling': {
            'minimum_segment_rung_vmaf': min_vmaf,
            'maximum_segment_rung_vmaf': max_vmaf,
            'minimum_utility': 0.0,
            'maximum_utility': max_log_utility,
            'rule': 'Map measured segment/rung VMAF linearly to [0, log(max_bitrate/min_bitrate)] so the per-second reward scale matches the existing log-bitrate objective.',
        },
        'nonmonotonic_segment_count': int(nonmonotonic.any(axis=1).sum()),
        'nonmonotonic_adjacent_rung_pairs': int(nonmonotonic.sum()),
        'mean_vmaf_by_rung': segment_scores.mean(axis=0).tolist(),
        'utility_by_segment_and_action': utilities.tolist(),
        'limitations': [
            'VMAF utility is calibrated on this exact encoded content and ladder.',
            'This maps encoded-frame quality to reward; it does not model human opinion or cross-content preference.',
            'The VMAF values are measured from full aligned encodes and reused for each segment during training.',
        ],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'content': args.content,
                      'segments': len(durations), 'actions': len(bitrates),
                      'nonmonotonic_segments': result['nonmonotonic_segment_count'],
                      'mean_vmaf_by_rung': np.round(segment_scores.mean(axis=0), 3).tolist()}), flush=True)


if __name__ == '__main__':
    main()
