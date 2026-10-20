"""Media timing is explicit; part byte sizes are never inferred from a segment."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .data import Video, vector


@dataclass(frozen=True)
class Media:
    video: Video
    part_durations_s: tuple[np.ndarray, ...] | None = None
    part_sizes_bytes: tuple[np.ndarray, ...] | None = None
    provenance: str = "unspecified segment sizes"

    def __post_init__(self):
        if (self.part_durations_s is None) != (self.part_sizes_bytes is None):
            raise ValueError("Part durations and part byte sizes must be supplied together")
        if self.part_durations_s is None:
            return
        if len(self.part_durations_s) != len(self.video.durations_s) or len(self.part_sizes_bytes) != len(self.video.durations_s):
            raise ValueError("Part tables must cover every segment")
        durations, sizes = [], []
        for i, (ds, bs) in enumerate(zip(self.part_durations_s, self.part_sizes_bytes)):
            ds = vector(ds, positive=True)
            bs = np.array(bs, dtype=float, copy=True)
            if (bs.shape != (len(ds), len(self.video.bitrates_kbps)) or
                    not np.isfinite(bs).all() or (bs <= 0).any()):
                raise ValueError("Positive [part, representation] byte sizes required")
            if not np.isclose(ds.sum(), self.video.durations_s[i], rtol=0, atol=1e-6):
                raise ValueError("Part durations must sum to segment duration")
            if not np.allclose(bs.sum(axis=0), self.video.sizes_bytes[i], rtol=1e-9, atol=1e-6):
                raise ValueError("Part sizes must sum to segment byte sizes")
            bs.setflags(write=False)
            durations.append(ds)
            sizes.append(bs)
        object.__setattr__(self, "part_durations_s", tuple(durations))
        object.__setattr__(self, "part_sizes_bytes", tuple(sizes))

    def units(self, segment, live):
        if live:
            if self.part_durations_s is None:
                raise ValueError("Live mode requires explicit part durations and byte sizes")
            return self.part_durations_s[segment], self.part_sizes_bytes[segment]
        return np.array([self.video.durations_s[segment]]), self.video.sizes_bytes[segment:segment + 1]

    def save(self, path):
        v = self.video
        Path(path).write_text(json.dumps({"schema_version": 1, "provenance": self.provenance,
            "video": {"bitrates_kbps": v.bitrates_kbps.tolist(), "durations_s": v.durations_s.tolist(),
                      "sizes_bytes": v.sizes_bytes.tolist()},
            "part_durations_s": None if self.part_durations_s is None else [x.tolist() for x in self.part_durations_s],
            "part_sizes_bytes": None if self.part_sizes_bytes is None else [x.tolist() for x in self.part_sizes_bytes]}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop("schema_version") != 1:
            raise ValueError("Unsupported media schema")
        return cls(video=Video(**data.pop("video")), **data)
