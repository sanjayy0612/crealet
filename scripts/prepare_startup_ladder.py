"""Encode an aligned real ladder, including an explicitly very-low-quality rung."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np

from abr.data import Dataset, Video
from abr.mp4 import split_generated_mp4

OUT = Path('artifacts/startup-ladder-v1')
LADDER = [(8, 96, 54), (300, 320, 180), (750, 480, 270), (1200, 640, 360),
          (1850, 854, 480), (2850, 1280, 720), (4300, 1920, 1080)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Optional second video; no training on its content is implied')
    parser.add_argument('--input-fps', type=int, default=25)
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--data-dir', default='data/startup-ladder')
    parser.add_argument('--source-url')
    parser.add_argument('--label', default='Envivio')
    args = parser.parse_args()
    if args.input_fps <= 0:
        parser.error('Positive input frame rate required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    binary = imageio_ffmpeg.get_ffmpeg_exe()
    version = subprocess.check_output([binary, '-version'], text=True).splitlines()[0]
    original_index = json.loads(Path('artifacts/player-media/index.json').read_text())
    source = Path(args.source) if args.source else output / 'source.mp4'
    if not args.source and not source.exists():
        with source.open('wb') as destination:
            for name in ['Header.m4s'] + [f'{i}.m4s' for i in range(1, 50)]:
                destination.write((Path('artifacts/player-media/video1') / name).read_bytes())
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_commit = original_index['source_commit'] if not args.source else 'external-sha256:' + source_sha
    source_url = args.source_url or original_index['source_url']
    normalize = f'setpts=N/({args.input_fps}*TB)' + (',fps=25' if args.input_fps != 25 else '')
    manifest = {'source_commit': source_commit, 'source_url': source_url,
                'derived_media': {'ffmpeg': version, 'source_sha256': source_sha,
                                  'content': args.label, 'timing': normalize + '; output25fps, closed100-frame GOPs',
                                  'quality_warning': '8 kbps rung is 96x54, deliberately very low visual quality'},
                'representations': [], 'files': {}}
    # Preserve the first experiment's exact identity when rebuilding its default
    # cache; adding the optional second-video path must not relabel old data.
    if not args.source:
        manifest['derived_media'].pop('content')
        manifest['derived_media']['timing'] = 'Decode frame order, normalize PTS=N/(25*TB); 25 fps, closed 100-frame GOPs'
    sizes, durations, recipes = [], None, []
    for kbps, width, height in LADDER:
        name = f'encoded-{kbps}.mp4'
        encoded = output / name
        command = [binary, '-nostdin', '-hide_banner', '-y', '-i', str(source), '-an', '-vf',
                   f'{normalize},scale={width}:{height}', '-c:v', 'libx264', '-threads', '2',
                   '-preset', 'slow' if kbps == 8 else 'veryfast', '-profile:v', 'baseline', '-pix_fmt', 'yuv420p',
                   '-b:v', f'{kbps}k', '-maxrate', f'{kbps}k', '-bufsize', f'{kbps if kbps == 8 else 2*kbps}k',
                   '-g', '100', '-keyint_min', '100', '-sc_threshold', '0', '-r', '25',
                   '-video_track_timescale', '12800', '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(encoded)]
        if not encoded.exists():
            result = subprocess.run(command, capture_output=True, text=True)
            (output / f'encode-{kbps}.log').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'Encoding failed; inspect encode-{kbps}.log')
        init, fragments, times, codec = split_generated_mp4(encoded.read_bytes())
        if not args.source and len(fragments) != 49:
            raise ValueError('Expected 49 aligned fragments')
        if durations is not None and not np.allclose(times, durations, atol=1e-9, rtol=0):
            raise ValueError('Representation timings differ')
        durations = times
        rep = f'r{kbps}'
        root = output / rep
        root.mkdir(exist_ok=True)
        for filename, content in [('Header.m4s', init)] + [(f'{i+1}.m4s', b) for i, b in enumerate(fragments)]:
            path = root / filename
            if path.exists() and path.read_bytes() != content:
                raise ValueError('Existing derived media differs; use a new versioned output')
            path.write_bytes(content)
            manifest['files'][f'{rep}/{filename}'] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        # Decode the assembled stream with strict error handling. Output PTS are normalized.
        decoded = subprocess.run([binary, '-nostdin', '-hide_banner', '-v', 'error', '-xerror', '-i', str(encoded),
                                  '-f', 'null', '-'], capture_output=True, text=True)
        if decoded.returncode or decoded.stderr.strip():
            raise ValueError(f'Derived representation failed strict decode: {decoded.stderr[:300]}')
        manifest['representations'].append({'id': rep, 'bitrate_kbps': kbps, 'codec': codec, 'width': width, 'height': height})
        sizes.append([len(b) for b in fragments])
        recipes.append({'bitrate_kbps': kbps, 'command': command[1:], 'encoded_sha256': hashlib.sha256(encoded.read_bytes()).hexdigest(),
                        'initialization_bytes': len(init), 'first_media_bytes': len(fragments[0]), 'strict_decode_passed': True})
        print(json.dumps({'kbps': kbps, 'codec': codec, 'first_bytes': len(fragments[0]), 'init_bytes': len(init)}), flush=True)
    (output / 'index.json').write_text(json.dumps(manifest, indent=2) + '\n')
    video = Video([r[0] for r in LADDER], durations, np.asarray(sizes).T)
    for filename, original in [('training.json', 'data/hard_envivio/dataset.json'),
                               ('development.json', 'data/prefetch-sessions/dataset.json')]:
        old = Dataset.load(original)
        description = ('Derived aligned real media; same source video content, not a new video; '
                       if not args.source else f'Derived aligned real media: {args.label}; ')
        Dataset(video, old.splits, description +
                'all rungs re-encoded with normalized decoded-frame PTS, exact MP4 sample durations and real byte sizes; '
                'same trace splits; 8 kbps survival rung has very low visual quality').save(data_dir / filename)
    provenance = {'ffmpeg': version, 'ffmpeg_binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                  'source_sha256': source_sha, 'source_commit': manifest['source_commit'], 'source_url': source_url,
                  'content': args.label, 'recipes': recipes,
                  'duration_s': sum(durations), 'durations_s': durations, 'original_assets_unchanged': True}
    (data_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (data_dir / 'README.md').write_text(f'Real, aligned H.264 fragments: {args.label}. Source: {source_url}. '
        'An 8 kbps 96x54 rung trades substantial visual quality for startup/download feasibility. '
        'Sizes and sample durations are measured from MP4 boxes. '
        'Recipes, encoder/version hashes and strict decode results are in provenance.json. '
        f'Large media stays in ignored {output}; run scripts.prepare_startup_ladder to rebuild.\n')


if __name__ == '__main__':
    main()
