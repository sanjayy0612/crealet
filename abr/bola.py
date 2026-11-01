"""BOLA-Basic style VOD baseline using only current buffer and public candidates."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BolaBasicConfig:
    segment_duration_s: float = 4.0
    gamma_p: float = 5.0

    def __post_init__(self):
        if (not np.isfinite(self.segment_duration_s) or self.segment_duration_s <= 0
                or not np.isfinite(self.gamma_p) or self.gamma_p < 0):
            raise ValueError('BOLA segment duration must be positive and gamma_p nonnegative')


class BolaBasicPolicy:
    """Buffer-only bitrate selection with logarithmic bitrate utility.

    This follows the BOLA-Basic score ``(V*(utility + gamma*p) - Q) / size``.
    Segment sizes use observation-v2 nominal size estimates. The source
    algorithm includes a no-download choice, but this simulator only accepts a
    bitrate action; this adapter chooses the highest-scoring bitrate instead.
    """

    def __init__(self, *, config: BolaBasicConfig | None = None):
        self.config = config or BolaBasicConfig()

    def scores(self, observation, stream_config):
        candidates = np.asarray(observation['candidates'], dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) < 2:
            raise ValueError('BOLA expects at least two bitrate candidates with size and duration')
        bitrates = candidates[:, 0]
        sizes = candidates[:, 1]
        if (not np.isfinite(candidates).all() or (bitrates <= 0).any()
                or (sizes <= 0).any()):
            raise ValueError('BOLA candidate rates and sizes must be positive and finite')
        p = self.config.segment_duration_s
        q = float(observation['state'][0]) / p
        q_max = float(stream_config.max_buffer_s) / p
        if q_max <= 1:
            raise ValueError('BOLA requires room for more than one segment in the buffer')
        utility = np.log(bitrates / bitrates.min())
        vmax = float(utility.max())
        if vmax <= 0:
            raise ValueError('BOLA requires at least two distinct candidate bitrates')
        v = (q_max - 1.0) / (vmax + self.config.gamma_p)
        return (v * (utility + self.config.gamma_p) - q) / sizes

    def __call__(self, observation, stream_config):
        scores = self.scores(observation, stream_config)
        # The no-download option is omitted because env.step requires a rung.
        return int(np.argmax(scores))


class BolaFinitePolicy(BolaBasicPolicy):
    """BOLA-FINITE scoring and pause for a known finite VOD presentation.

    The caller supplies exact elapsed and remaining media duration. The pause
    is returned separately because ``StreamingEnv.step`` accepts only a rung.
    This implements the finite-horizon buffer target from the paper; it does
    not model request abandonment or a separate no-download action.
    """

    def decision(self, observation, stream_config, *, elapsed_s, remaining_s):
        if (not np.isfinite(elapsed_s) or elapsed_s < 0
                or not np.isfinite(remaining_s) or remaining_s < 0):
            raise ValueError('BOLA-FINITE elapsed and remaining duration must be finite and nonnegative')
        p = self.config.segment_duration_s
        candidates = np.asarray(observation['candidates'], dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) < 2:
            raise ValueError('BOLA expects at least two bitrate candidates with size and duration')
        bitrates, sizes = candidates[:, 0], candidates[:, 1]
        if (not np.isfinite(candidates).all() or (bitrates <= 0).any()
                or (sizes <= 0).any()):
            raise ValueError('BOLA candidate rates and sizes must be positive and finite')
        q = float(observation['state'][0]) / p
        q_max = float(stream_config.max_buffer_s) / p
        if q_max <= 1:
            raise ValueError('BOLA requires room for more than one segment in the buffer')
        utility = np.log(bitrates / bitrates.min())
        u_max = float(utility.max())
        if u_max <= 0:
            raise ValueError('BOLA requires at least two distinct candidate bitrates')

        # BOLA-FINITE shrinks its target buffer near startup and end of video.
        t = min(elapsed_s, remaining_s)
        t_prime = max(t / 2.0, 3.0 * p)
        q_max_dynamic = min(q_max, t_prime / p)
        v = (q_max_dynamic - 1.0) / (u_max + self.config.gamma_p)
        scores = (v * (utility + self.config.gamma_p) - q) / sizes
        wait_s = max(p * (q - q_max_dynamic + 1.0), 0.0)
        return int(np.argmax(scores)), float(wait_s), {
            'q_segments': q,
            'q_max_dynamic_segments': q_max_dynamic,
            'v': v,
            'scores': scores.tolist(),
        }
