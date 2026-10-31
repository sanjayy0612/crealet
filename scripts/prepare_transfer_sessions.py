"""Predeclared remaining ferry sessions; outcomes never select a training run."""
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from abr.data import Dataset, Trace


def main():
    dates = ['2011-02-01_0840CET', '2011-02-01_1000CET', '2011-02-01_1539CET', '2011-02-01_1639CET']
    out = Path('runs/startup-ladder')
    protocol = {'sessions': dates, 'selection': 'Next four unused archive filenames, no filtering by observed outcomes/duration',
                'videos': ['derived Envivio', 'Sintel trailer, Blender Foundation / Durian Open Movie Team'],
                'second_video_used_for_training': False, 'context': '60 s cap, startup/resume1 s, same recovery/filter',
                'checkpoint_selection': 'Validation only; each fixed seed evaluated after checkpoints are frozen',
                'finite_traces': 'Keep too-short traces and failures; no repetition or padding',
                'Sintel_source': 'https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4',
                'Sintel_credit_license': 'Blender Foundation / Durian Open Movie Team; video-only re-encode, CC BY 3.0; https://durian.blender.org/about/'}
    path = out / 'transfer-protocol.json'
    serialized = json.dumps(protocol, indent=2) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise ValueError('Existing transfer protocol differs')
    path.write_text(serialized)
    raw_dir = Path('data/transfer-sessions')
    raw_dir.mkdir(parents=True, exist_ok=True)
    known = [Dataset.load(p) for p in ('data/hard_envivio/dataset.json', 'data/learning-holdout/dataset.json',
             'data/buffer-sessions/dataset.json', 'data/prefetch-sessions/dataset.json')]
    hashes = {t.fingerprint for d in known for ts in d.splits.values() for t in ts}
    traces, metadata = [], []
    for date in dates:
        path = raw_dir / f'report.{date}.log'
        url = 'https://datasets.simula.no/downloads/hsdpa-tcp-logs/ferry.nesoddtangen-oslo/' + path.name
        if not path.exists():
            path.write_bytes(subprocess.check_output(['curl', '--fail', '--silent', '--show-error', '--location',
                '--retry', '2', '--max-time', '40', url]))
        raw = path.read_bytes(); values = np.loadtxt(io.BytesIO(raw), ndmin=2)
        durations = np.diff(values[:, 1]) / 1000
        if values.shape[1] != 6 or not np.isfinite(values).all() or (durations <= 0).any() or (values[1:, 4] < 0).any():
            raise ValueError('Invalid source log; no imputation')
        sha = hashlib.sha256(raw).hexdigest()
        trace = Trace('transfer-ferry-' + date, durations, values[1:, 4] * 8 / (durations * 1000), .08,
                      source_group='transfer-ferry-route', source=f'{url}; raw_sha256={sha}; first row discarded; '
                      'bytes/monotonic timestamp deltas; RTT assumed80ms')
        if trace.fingerprint in hashes:
            raise ValueError('Already-used trace')
        traces.append(trace)
        metadata.append({'file': str(path), 'url': url, 'raw_sha256': sha, 'trace_sha256': trace.fingerprint})
    for label, directory in [('envivio', 'data/startup-ladder'), ('sintel', 'data/sintel-ladder')]:
        old = Dataset.load(directory + '/development.json')
        Dataset(old.video, {**old.splits, 'test': tuple(traces)}, 'Fixed new-session transfer evaluation; '
                'same route; never used for training or checkpoint selection').save(raw_dir / (label + '.json'))
    (raw_dir / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (raw_dir / 'README.md').write_text('Four remaining original ferry sessions fixed before scoring. '
        'Raw provenance retained, no scaling/repetition or duration-based exclusions. Envivio and Sintel '
        'datasets share these traces but have separate measured video arrays. Only test is scored.\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
