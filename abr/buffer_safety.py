"""Opt-in buffer-aware safety adapter; neural checkpoint sources stay unchanged."""
from dataclasses import dataclass
import numpy as np

from .models.safety import ActionSafety, allowed_actions as baseline_allowed_actions
from .models.policy import DecisionPolicy


@dataclass(frozen=True)
class BufferAwareSafety(ActionSafety):
    """Experimental normal-floor recovery; preserves the original filter too.

    All rates and sizes are observed or public nominal estimates. This is a
    bounded heuristic, not a guarantee against a future outage or size error.
    """
    recovery_samples: int = 3
    recovery_floor_kbps: float = 300.
    recovery_min_buffer_s: float = 12.
    recovery_reserve_s: float = 4.
    recovery_max_download_s: float = 8.
    recovery_size_factor: float = 1.25

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.recovery_samples, bool) or not isinstance(self.recovery_samples, int) or self.recovery_samples < 2:
            raise ValueError('At least two sustained recovery samples required')
        values = (self.recovery_floor_kbps, self.recovery_min_buffer_s,
                  self.recovery_reserve_s, self.recovery_max_download_s, self.recovery_size_factor)
        if any(not np.isfinite(v) or v <= 0 for v in values) or self.recovery_size_factor < 1:
            raise ValueError('Positive finite recovery limits and size factor >= 1 required')


def recovery_floor_action(observation, safety):
    """Return one permitted recovery action, or None; do not modify observations."""
    history = observation['history'][observation['history'][:, -1] > 0]
    candidates = observation['candidates']
    floor = np.flatnonzero(np.isclose(candidates[:, 0] * 1000, safety.recovery_floor_kbps, rtol=1e-5))
    buffer = float(observation['state'][0])
    if (len(floor) != 1 or len(history) < safety.recovery_samples or
            buffer < max(safety.min_buffer_s, safety.recovery_min_buffer_s)):
        return None
    action = int(floor[0])
    recent = history[-safety.recovery_samples:]
    rate = safety.throughput_factor * float(np.min(recent[:, 0]))
    # Each recent sample must support the floor; one fast body is insufficient.
    if not np.isfinite(rate) or rate < float(candidates[action, 0]) or rate <= 0:
        return None
    # MB * 8 / Mbps = seconds. Budget a whole segment for possible cap wait,
    # preserve a playback reserve, and include observed header/RTT overhead.
    _, size_mb, duration = map(float, candidates[action])
    rtt = float(np.max(recent[:, 3])) * 10
    usable = min(safety.recovery_max_download_s, buffer - duration - safety.recovery_reserve_s)
    estimate = safety.recovery_size_factor * size_mb * 8 / rate + rtt
    return action if np.isfinite(estimate) and 0 < estimate <= usable else None


def allowed_actions(observation, safety=None):
    allowed = baseline_allowed_actions(observation, safety)
    if isinstance(safety, BufferAwareSafety):
        recovery = recovery_floor_action(observation, safety)
        if recovery is not None:
            allowed[recovery] = True
    return allowed


class BufferAwarePolicy(DecisionPolicy):
    def __call__(self, observation, stream_config):
        scores = self.scores(observation, stream_config)
        return int(np.where(allowed_actions(observation, self.safety), scores, -np.inf).argmax())
