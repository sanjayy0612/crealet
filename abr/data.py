"""Explicit units and disjoint network-trace splits. No network downloads at import."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def vector(values, *, positive=False):
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite vector")
    invalid = (result <= 0).any() if positive else (result < 0).any()
    if invalid:
        raise ValueError("Values must be positive" if positive else "Values must be nonnegative")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Trace:
    name: str
    durations_s: np.ndarray
    bandwidth_kbps: np.ndarray
    rtt_s: float = 0.08
    source_group: str | None = None
    source: str = "unspecified"
    semantics: str = "observed_goodput"
    rtt_series_s: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "durations_s", vector(self.durations_s, positive=True))
        object.__setattr__(self, "bandwidth_kbps", vector(self.bandwidth_kbps))
        if self.durations_s.shape != self.bandwidth_kbps.shape:
            raise ValueError("One bandwidth value is required per interval")
        if not self.name or not np.isfinite(self.rtt_s) or self.rtt_s < 0:
            raise ValueError("Trace name and nonnegative finite RTT required")

        if self.semantics not in {"observed_goodput", "controlled_capacity"}:
            raise ValueError("Unknown trace semantics")
        if self.rtt_series_s is not None:
            rtts = vector(self.rtt_series_s)
            if rtts.shape != self.durations_s.shape:
                raise ValueError("One RTT per trace interval is required")
            object.__setattr__(self, "rtt_series_s", rtts)

    @property
    def fingerprint(self):
        # Deliberately excludes the name: renaming a leaked trace must not evade detection.
        return hashlib.sha256(self.durations_s.tobytes() + self.bandwidth_kbps.tobytes()).hexdigest()


@dataclass(frozen=True)
class Video:
    bitrates_kbps: np.ndarray
    durations_s: np.ndarray
    sizes_bytes: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "bitrates_kbps", vector(self.bitrates_kbps, positive=True))
        object.__setattr__(self, "durations_s", vector(self.durations_s, positive=True))
        sizes = np.array(self.sizes_bytes, dtype=np.float64, copy=True)
        if (sizes.shape != (len(self.durations_s), len(self.bitrates_kbps))
                or not np.isfinite(sizes).all() or (sizes <= 0).any()
                or (np.diff(self.bitrates_kbps) <= 0).any()):
            raise ValueError("Video needs increasing bitrates and positive [chunk, quality] byte sizes")
        sizes.setflags(write=False)
        object.__setattr__(self, "sizes_bytes", sizes)


@dataclass(frozen=True)
class Dataset:
    video: Video
    splits: dict[str, tuple[Trace, ...]]
    provenance: str

    def __post_init__(self):
        if set(self.splits) != {"train", "validation", "test"}:
            raise ValueError("Exactly train, validation and test splits are required")
        seen_names, seen_hashes, groups = set(), set(), {}
        for split, traces in self.splits.items():
            if not traces:
                raise ValueError("Each split must have at least one trace")
            for trace in traces:
                if trace.name in seen_names or trace.fingerprint in seen_hashes:
                    raise ValueError("Duplicate trace name or content across dataset")
                if trace.source_group is not None:
                    if trace.source_group in groups and groups[trace.source_group] != split:
                        raise ValueError("Source session/group appears in multiple splits")
                    groups[trace.source_group] = split
                seen_names.add(trace.name)
                seen_hashes.add(trace.fingerprint)

    def to_dict(self):
        return {"schema_version": 2, "provenance": self.provenance,
                "video": {"bitrates_kbps": self.video.bitrates_kbps.tolist(),
                          "durations_s": self.video.durations_s.tolist(),
                          "sizes_bytes": self.video.sizes_bytes.tolist()},
                "splits": {key: [{"name": t.name, "durations_s": t.durations_s.tolist(),
                                  "bandwidth_kbps": t.bandwidth_kbps.tolist(), "rtt_s": t.rtt_s,
                                  "source_group": t.source_group, "source": t.source,
                                  "semantics": t.semantics,
                                  "rtt_series_s": None if t.rtt_series_s is None else t.rtt_series_s.tolist()}
                                 for t in traces] for key, traces in self.splits.items()}}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["schema_version"] not in (1, 2):
            raise ValueError("Unsupported dataset schema")
        return cls(Video(**data["video"]),
                   {key: tuple(Trace(**item) for item in items)
                    for key, items in data["splits"].items()}, data["provenance"])


def synthetic_dataset(seed=17, chunks=48):
    """Engineering fixtures, not measured network/video data or a research benchmark."""
    rng = np.random.default_rng(seed)
    ladder = np.array([500., 1000., 2000.])
    durations = np.full(chunks, 4.)
    complexity = np.clip(rng.lognormal(-0.03, 0.25, chunks), 0.5, 1.8)
    sizes = durations[:, None] * ladder[None, :] * 1000 / 8 * complexity[:, None]
    splits = {}
    for split, count in (("train", 24), ("validation", 8), ("test", 8)):
        traces = []
        for index in range(count):
            base = rng.uniform(550, 4200)
            # Correlated intervals, sudden drops and finite zero-capacity outages.
            knots = rng.uniform(0.45, 1.5, 41)
            capacity = base * np.interp(np.arange(1200), np.arange(41) * 30, knots)
            capacity *= rng.lognormal(-0.005, 0.1, 1200)
            if index % 2 == 0:
                start = int(rng.integers(30, 100))
                capacity[start:start + 30] *= 0.18
            if index % 3 == 0:
                start = int(rng.integers(15, 150))
                capacity[start:start + int(rng.integers(2, 9))] = 0
            traces.append(Trace(f"synthetic-{split}-{index}", np.ones(1200), capacity,
                                float(rng.uniform(0.02, 0.15))))
        splits[split] = tuple(traces)
    return Dataset(Video(ladder, durations, sizes), splits,
                   f"synthetic engineering fixtures, generator seed={seed}; shared synthetic video")
