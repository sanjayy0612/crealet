"""Validated, content-aligned quality utilities for reward experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_quality_utility(path, dataset, *, expected_sha256=None):
    """Load a [segment, representation] utility matrix for one exact video ladder."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'Quality utility hash mismatch: {path}')
    payload = json.loads(raw)
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported quality utility schema')
    if not np.array_equal(np.asarray(payload['bitrates_kbps'], dtype=np.float64),
                          dataset.video.bitrates_kbps):
        raise ValueError('Quality utility bitrate ladder does not match dataset')
    if not np.array_equal(np.asarray(payload['durations_s'], dtype=np.float64),
                          dataset.video.durations_s):
        raise ValueError('Quality utility segment durations do not match dataset')
    matrix = np.asarray(payload['utility_by_segment_and_action'], dtype=np.float64)
    expected_shape = (len(dataset.video.durations_s), len(dataset.video.bitrates_kbps))
    if (matrix.shape != expected_shape or not np.isfinite(matrix).all()
            or (matrix < 0).any()):
        raise ValueError('Quality utility must be a finite nonnegative [segment, action] matrix')
    return matrix
