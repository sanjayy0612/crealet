"""Auditable imports: explicit units, source session, checksum and replay semantics.

No downloads. HTTP goodput is an observation, not intrinsic link capacity.
"""
import csv
import hashlib
from pathlib import Path
import numpy as np
from .data import Trace


def _source(path, source_url, detail):
    if not source_url:
        raise ValueError("A source URL or collection description is required")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"{source_url}; raw_sha256={digest}; {detail}"


def load_ghent4g(path, *, source_group, source_url, rtt_s=0.08):
    """Ghent's six-column log: interval bytes / interval milliseconds * 8 = kbps.

    RTT is not in these logs. The caller's RTT is an explicit assumed request cost.
    """
    if not source_group:
        raise ValueError("A collection session/group is required before splitting")
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError("Expected six finite Ghent log columns")
    elapsed_ms, byte_count, interval_ms = rows[:, 1], rows[:, 4], rows[:, 5]
    if (interval_ms <= 0).any() or (byte_count < 0).any() or (np.diff(elapsed_ms) <= 0).any():
        raise ValueError("Invalid intervals, bytes, or timestamp order")
    if not np.allclose(np.diff(elapsed_ms), interval_ms[1:], rtol=0, atol=2):
        raise ValueError("Missing/overlapping intervals: do not silently fill log gaps")
    return Trace(Path(path).stem, interval_ms/1000, byte_count*8/interval_ms, rtt_s,
                 source_group=source_group, source=_source(path, source_url,
                 f"Ghent 4G interval-byte log; assumed RTT={rtt_s}s"))


def load_interval_csv(path, *, source_group, source_url, semantics="observed_goodput"):
    """Canonical CSV header: duration_s,goodput_kbps,rtt_s (one row per interval)."""
    if not source_group:
        raise ValueError("A collection session/group is required before splitting")
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['duration_s', 'goodput_kbps', 'rtt_s']:
            raise ValueError("CSV header must be duration_s,goodput_kbps,rtt_s")
        rows = np.array([[float(row[k]) for k in reader.fieldnames] for row in reader], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("Nonempty interval table required")
    return Trace(Path(path).stem, rows[:, 0], rows[:, 1], source_group=source_group,
                 source=_source(path, source_url, "canonical interval CSV"), semantics=semantics,
                 rtt_series_s=rows[:, 2])


def load_timestamp_mbps(path, *, source_group, source_url, alignment, rtt_s=0.08):
    """Two columns: timestamp seconds, observed Mbps (e.g. Lumos5G).

    Alignment MUST be specified: 'start' holds row i until timestamp i+1;
    'end' applies row i+1 to the preceding interval. Never invent a last interval.
    Keep all windows from the same collection group in the same dataset split.
    """
    if alignment not in {'start', 'end'} or not source_group:
        raise ValueError("Explicit start/end alignment and collection group required")
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 2 or len(rows) < 2 or not np.isfinite(rows).all() or (rows[:, 1] < 0).any():
        raise ValueError("Expected finite timestamp-seconds/Mbps pairs")
    ds = np.diff(rows[:, 0])
    if (ds <= 0).any():
        raise ValueError("Timestamps must strictly increase")
    rates = rows[:-1, 1] if alignment == 'start' else rows[1:, 1]
    return Trace(Path(path).stem, ds, rates*1000, rtt_s, source_group=source_group,
                 source=_source(path, source_url, f"timestamp/Mbps; {alignment}-aligned; assumed RTT={rtt_s}s"))
