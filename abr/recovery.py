"""Bounded request recovery using received progress, not future trace capacity."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RecoveryConfig:
    max_retries: int = 4
    poll_s: float = 1.
    min_observation_s: float = 2.
    no_progress_s: float = 3.
    buffer_reserve_s: float = 2.
    backoff_s: float = 2.
    restart_fraction: float = .8
    early_abandon: bool = True

    def __post_init__(self):
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError('Nonnegative integer retry limit required')
        for name in ('poll_s', 'min_observation_s', 'no_progress_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('Positive finite progress intervals required')
        for name in ('buffer_reserve_s', 'backoff_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError('Nonnegative finite reserve and backoff required')
        if not math.isfinite(self.restart_fraction) or not 0 < self.restart_fraction <= 1:
            raise ValueError('Restart fraction must be in (0,1]')
        if type(self.early_abandon) is not bool:
            raise ValueError('early_abandon must be boolean')

    def abandon_reason(self, *, elapsed_s, received_bytes, content_bytes, idle_s,
                       buffer_s, lowest_estimate_bytes, at_lowest):
        if not self.early_abandon or at_lowest or elapsed_s < self.min_observation_s:
            return None
        remaining = max(0., content_bytes - received_bytes)
        # A nearly complete request is cheaper to finish than restart.
        if lowest_estimate_bytes >= self.restart_fraction * remaining:
            return None
        if idle_s >= self.no_progress_s:
            return 'no_progress'
        if received_bytes > 0:
            remaining_s = remaining / (received_bytes / elapsed_s)
            if remaining_s > max(self.min_observation_s, buffer_s - self.buffer_reserve_s):
                return 'buffer_risk'
        return None

    def to_dict(self):
        return asdict(self)
