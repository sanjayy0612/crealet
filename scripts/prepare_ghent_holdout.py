"""Prepare previously unused Ghent 4G sessions as an Envivio transfer set.

Selection is filename-based: include every archive log except the eight already
included in ``data/realistic_envivio``. No throughput values affect selection.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from abr.data import Dataset, Trace


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/ghent-holdout/raw/logs_all.zip'
OUTPUT = ROOT / 'data/ghent-holdout/dataset.json'
PROVENANCE = ROOT / 'data/ghent-holdout/provenance.json'
ARCHIVE_URL = 'https://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip'
LOG_BASE_URL = 'https://users.ugent.be/~jvdrhoof/dataset-4g/logs/'


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def existing_dataset_fingerprints():
    fingerprints = set()
    for path in (ROOT / 'data').glob('**/dataset.json'):
        if path.resolve() == OUTPUT.resolve():
            continue
        try:
            dataset = Dataset.load(path)
        except (KeyError, ValueError, TypeError):
            continue
        fingerprints.update(trace.fingerprint for traces in dataset.splits.values() for trace in traces)
    return fingerprints


def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f'Missing public trace archive: {ARCHIVE}')
    archive_bytes = ARCHIVE.read_bytes()
    archive_sha256 = sha256_bytes(archive_bytes)
    original = Dataset.load(ROOT / 'data/startup-ladder/training.json')
    used_files = {f'{trace.name}.log' for traces in Dataset.load(
        ROOT / 'data/realistic_envivio/dataset.json').splits.values() for trace in traces}
    known_fingerprints = existing_dataset_fingerprints()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        all_logs = sorted(name for name in archive.namelist()
                          if Path(name).name.startswith('report_') and name.endswith('.log'))
        new_logs = [name for name in all_logs if Path(name).name not in used_files]
        if len(all_logs) != 40 or len(new_logs) != 32:
            raise ValueError(f'Expected 40 archive logs / 32 new sessions; found {len(all_logs)} / {len(new_logs)}')

        traces, session_records = [], []
        for member in new_logs:
            raw = archive.read(member)
            values = np.loadtxt(io.BytesIO(raw), ndmin=2)
            if values.shape[1] != 6 or len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f'Invalid six-column Ghent trace: {member}')
            durations_s = np.diff(values[:, 1]) / 1000.0
            received_bytes = values[1:, 4]
            if (durations_s <= 0).any() or (received_bytes < 0).any():
                raise ValueError(f'Invalid timestamps or byte counts in {member}')
            filename = Path(member).name
            stem = filename.removesuffix('.log')
            mode = stem.removeprefix('report_').rsplit('_', 1)[0]
            trace = Trace(
                name='ghent-' + stem,
                durations_s=durations_s,
                bandwidth_kbps=received_bytes * 8.0 / np.diff(values[:, 1]),
                rtt_s=0.08,
                source_group='ghent-' + mode,
                source=(f'{LOG_BASE_URL}{filename}; archive_sha256={archive_sha256}; '
                        f'member_sha256={sha256_bytes(raw)}; bytes divided by monotonic elapsed ms; '
                        'RTT assumed 0.08 s'),
                semantics='observed_goodput',
            )
            if trace.fingerprint in known_fingerprints:
                raise ValueError(f'Archive trace duplicates an existing dataset: {filename}')
            traces.append(trace)
            session_records.append({
                'filename': filename,
                'mode_group': mode,
                'archive_member': member,
                'raw_sha256': sha256_bytes(raw),
                'trace_sha256': trace.fingerprint,
                'intervals': len(trace.durations_s),
                'duration_s': float(trace.durations_s.sum()),
            })

    dataset = Dataset(
        original.video,
        {'train': original.splits['train'],
         'validation': original.splits['validation'],
         'test': tuple(traces)},
        'Envivio aligned seven-rung video from data/startup-ladder/training.json. '
        'Test split contains 32 previously unused Ghent 4G/LTE sessions from the public 2015-2016 collection. '
        'Train/validation are copied only for Dataset schema compatibility and are not used by this evaluation. '
        'RTT is assumed to be 80 ms; throughput is observed HTTP goodput, not controlled link capacity.'
    )
    dataset.save(OUTPUT)
    PROVENANCE.write_text(json.dumps({
        'archive_url': ARCHIVE_URL,
        'archive_sha256': archive_sha256,
        'selection_rule': 'All report_*.log archive members except the eight filenames already in data/realistic_envivio; selected lexically before parsing throughput values.',
        'archive_log_count': len(session_records) + len(used_files),
        'excluded_existing_filenames': sorted(used_files),
        'test_session_count': len(session_records),
        'dataset_sha256': dataset.fingerprint,
        'video_dataset': 'data/startup-ladder/training.json',
        'video_ladder_kbps': dataset.video.bitrates_kbps.tolist(),
        'measurement_interval_conversion': 'duration_s = diff(column 2 ms) / 1000; bandwidth_kbps = column 5 bytes * 8 / diff(column 2 ms)',
        'rtt_assumption_s': 0.08,
        'train_validation_usage': 'Copied from startup-ladder training data only to satisfy Dataset schema; evaluator uses test split only.',
        'sessions': session_records,
    }, indent=2) + '\n')
    (ROOT / 'data/ghent-holdout/README.md').write_text(
        '# Ghent held-out 4G/LTE sessions\n\n'
        'The test split contains every session in the official archive that was not one of the eight logs already included in `data/realistic_envivio`. '
        'The selection rule uses filenames only and is recorded in `provenance.json`; all 32 unused sessions are retained. '
        'The video uses the same seven-rung aligned Envivio ladder as model training. '
        'The Norway train and validation traces exist only because the dataset schema requires all three splits; the evaluator uses only `test`.\n\n'
        'These are new sessions from an independent historical Ghent 4G collection. The source publishes transport-mode groups but does not provide enough route metadata to claim that every path is geographically distinct. '
        'The traces are measured HTTP goodput, and RTT is assumed to be 80 ms. This is not a current-network or causal capacity benchmark.\n\n'
        f'Archive: {ARCHIVE_URL}\n\n'
        'Rebuild with `.venv/bin/python -m scripts.prepare_ghent_holdout`.\n'
    )
    print(json.dumps({'archive_sha256': archive_sha256, 'dataset_sha256': dataset.fingerprint,
                      'test_sessions': len(traces), 'test_mode_groups': sorted({t.source_group for t in traces}),
                      'dataset': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
