"""Two predeclared additional sessions, never used for arm selection."""
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from abr.data import Dataset, Trace


def main():
    original = Dataset.load('data/hard_envivio/dataset.json')
    output = Path('data/learning-holdout')
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    for session in ('2010-09-21_1622CEST', '2010-09-21_1735CEST'):
        filename = f'report.{session}.log'
        url = 'https://datasets.simula.no/downloads/hsdpa-tcp-logs/ferry.nesoddtangen-oslo/' + filename
        path = output / filename
        if not path.exists():
            path.write_bytes(subprocess.check_output(['curl', '--fail', '--silent', '--show-error', '--location',
                                                     '--retry', '2', '--max-time', '40', url]))
        raw = path.read_bytes()
        x = np.loadtxt(io.BytesIO(raw), ndmin=2)
        durations = np.diff(x[:, 1]) / 1000
        if x.shape[1] != 6 or not np.isfinite(x).all() or (durations <= 0).any() or (x[1:, 4] < 0).any():
            raise ValueError('Invalid session; no imputation allowed')
        traces.append(Trace(f'fresh-ferry-{session}', durations, x[1:, 4] * 8 / (durations * 1000), .08,
                            source_group='fresh-ferry-route',
                            source=f'{url}; raw_sha256={hashlib.sha256(raw).hexdigest()}; bytes/monotonic timestamp differences, first row discarded; RTT assumed80ms'))
    dataset = Dataset(original.video, {**original.splits, 'test': tuple(traces)},
                      'Two additional original Norway sessions, same ferry route as historical tests; not a new-route or new-video claim. Neither used for training or validation. Dates predeclared before experiment selection.')
    dataset.save(output / 'dataset.json')
    (output / 'README.md').write_text('Two additional measured sessions from the original Norway HSDPA host, predeclared before model selection. Same ferry route as previous tests; video and 80ms assumed RTT unchanged. Raw bytes and URLs/checksums retained. Only the test split is used from this file; existing train/validation sessions are included solely for Dataset schema compatibility.\n')
    print(json.dumps({'dataset_sha256': dataset.fingerprint, 'sessions': [t.name for t in traces]}))


if __name__ == '__main__':
    main()
