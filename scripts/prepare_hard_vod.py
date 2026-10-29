"""Prepare unscaled Norway HSDPA logs with whole routes in separate splits."""
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from abr.data import Dataset, Trace

BASE = 'https://datasets.simula.no/downloads/hsdpa-tcp-logs/'
SESSIONS = {
    'train': {
        'bus.ljansbakken-oslo': ['2010-09-28_1407CEST', '2010-09-29_0852CEST', '2010-09-29_1622CEST', '2010-09-29_1823CEST'],
        'car.snaroya-smestad': ['2011-02-14_2032CET', '2011-02-14_2051CET'],
        'tram.jernbanetorget-ljabru': ['2010-11-23_1515CET', '2010-12-09_1222CET', '2010-12-09_1310CET', '2010-12-16_1100CET'],
    },
    'validation': {'train.oslo-vestby': ['2011-02-11_1618CET', '2011-02-14_0644CET']},
    'test': {'ferry.nesoddtangen-oslo': ['2010-09-20_1542CEST', '2010-09-21_1001CEST']},
}
CONTROLLED = {
    'train': [([30, 45, 4, 121, 300], [3500, 650, 0, 1500, 2500]),
              ([50, 50, 6, 94, 300], [2200, 450, 0, 1800, 900])],
    'validation': [([35, 35, 4, 126, 300], [3000, 450, 0, 2600, 1200])],
    'test': [([55, 45, 6, 94, 300], [2500, 550, 0, 1800, 1400])],
}


def main():
    output = Path('data/hard_envivio')
    original = Dataset.load('data/realistic_envivio/dataset.json')
    splits, summary, excluded = {}, {}, []
    for split, routes in SESSIONS.items():
        traces = []
        summary[split] = []
        for route, sessions in routes.items():
            for session in sessions:
                filename = f'report.{session}.log'
                url = BASE + route + '/' + filename
                path = output / 'raw' / route / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    raw = subprocess.check_output(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2', '--max-time', '45', url])
                    path.write_bytes(raw)
                raw = path.read_bytes()
                x = np.loadtxt(io.BytesIO(raw), ndmin=2)
                if x.shape[1] != 6 or not np.isfinite(x).all():
                    raise ValueError(f'Invalid log: {path}')
                # Source defines bytes as received since the previous measurement.
                # Use measured elapsed timestamps, including logging overhead. Row 0
                # has no preceding timestamp and is discarded. Do not invent a tail.
                ms, byte_count, dt = x[:, 1], x[1:, 4], np.diff(x[:, 1])
                mismatches = int((np.abs(dt - x[1:, 5]) > 2).sum())
                if (dt <= 0).any() or (byte_count < 0).any():
                    excluded.append({'path': str(path), 'reason': 'nonmonotonic timestamps or invalid bytes'})
                    continue
                trace = Trace(f'{route}-{session}', dt / 1000, byte_count * 8 / dt, .08,
                              source_group=route,
                              source=f'{url}; raw_sha256={hashlib.sha256(raw).hexdigest()}; HSDPA bytes since previous row / monotonic timestamp differences; first row discarded; reported duration discrepancies >2ms={mismatches}; RTT assumed 0.08s')
                traces.append(trace)
                summary[split].append({'trace': trace.name, 'duration_s': float(trace.durations_s.sum()),
                                       'mean_kbps': float(np.average(trace.bandwidth_kbps, weights=trace.durations_s)),
                                       'reported_interval_discrepancies': mismatches,
                                       'zero_intervals': int((trace.bandwidth_kbps == 0).sum())})
        for i, (durations, rates) in enumerate(CONTROLLED[split]):
            trace = Trace(f'controlled-{split}-{i}', durations, rates, .08,
                          source_group=f'controlled-{split}-{i}', semantics='controlled_capacity',
                          source='Explicit synthetic step-capacity stress case, not measured: slow period, abrupt drop, zero-capacity outage, recovery; no trace repetition')
            traces.append(trace)
            summary[split].append({'trace': trace.name, 'duration_s': sum(durations), 'capacity_kbps': rates,
                                   'intervals_s': durations, 'kind': 'synthetic controlled stress case'})
        splits[split] = tuple(traces)
    dataset = Dataset(original.video, splits,
                      'Envivio video sizes unchanged from realistic_envivio; measured Norway HSDPA 2010–2011; unscaled bytes / monotonic timestamp differences (first row discarded, reported interval discrepancies audited); full routes isolated by split; plus explicitly synthetic controlled outage profiles; RTT assumed 80ms; observed HTTP goodput is not causal capacity. Sources: https://datasets.simula.no/hsdpa/ and ' + BASE)
    dataset.save(output / 'dataset.json')
    (output / 'summary.json').write_text(json.dumps({'dataset_sha256': dataset.fingerprint, 'splits': summary, 'excluded': excluded}, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
