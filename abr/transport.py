"""Finite application-goodput replay. No TCP, radio scheduler, or packet-loss model."""
from dataclasses import dataclass
import numpy as np
from .data import Trace


@dataclass(frozen=True)
class Transfer:
    elapsed_s: float
    bytes_received: float
    complete: bool


class Replay:
    def __init__(self, trace: Trace, offset_s=0.):
        self.trace = trace
        self.ends = np.cumsum(trace.durations_s)
        if not np.isfinite(offset_s) or not 0 <= offset_s < self.ends[-1]:
            raise ValueError("Trace offset must be within the trace")
        self.time_s = float(offset_s)

    @property
    def rtt_s(self):
        i = min(int(np.searchsorted(self.ends, self.time_s, side="right")), len(self.ends) - 1)
        return float(self.trace.rtt_s if self.trace.rtt_series_s is None else self.trace.rtt_series_s[i])

    def wait(self, duration_s, deadline_s=float("inf")):
        elapsed = max(0., min(duration_s, self.ends[-1] - self.time_s, deadline_s - self.time_s))
        self.time_s += elapsed
        return float(elapsed), bool(elapsed + 1e-9 >= duration_s)

    def transfer(self, size_bytes, deadline_s=float("inf")):
        if not np.isfinite(size_bytes) or size_bytes <= 0:
            raise ValueError("Positive finite byte count required")
        start, remaining = self.time_s, float(size_bytes)
        end = min(self.ends[-1], deadline_s)
        while remaining > 1e-7 and self.time_s < end:
            i = int(np.searchsorted(self.ends, self.time_s, side="right"))
            rate = self.trace.bandwidth_kbps[i] * 1000 / 8
            available = min(self.ends[i], end) - self.time_s
            dt = available if rate == 0 else min(available, remaining / rate)
            remaining = max(0., remaining - rate * dt)
            self.time_s += dt
        return Transfer(float(self.time_s - start), float(size_bytes - remaining), bool(remaining <= 1e-7))
