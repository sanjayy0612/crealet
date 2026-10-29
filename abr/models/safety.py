"""Observed-throughput action filter; a heuristic, not a stall guarantee."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ActionSafety:
    min_buffer_s: float = 12.
    throughput_factor: float = .65

    def __post_init__(self):
        if not np.isfinite(self.min_buffer_s) or self.min_buffer_s < 0:
            raise ValueError('Finite nonnegative buffer reserve required')
        if not np.isfinite(self.throughput_factor) or not 0 < self.throughput_factor <= 1:
            raise ValueError('Throughput factor must be in (0, 1]')


def allowed_actions(observation, safety=None):
    candidates = observation['candidates']
    allowed = np.ones(len(candidates), dtype=bool)
    if safety is None:
        return allowed
    lowest = int(np.argmin(candidates[:, 0]))
    samples = observation['history'][observation['history'][:, -1] > 0, 0]
    allowed[:] = False
    if len(samples) and observation['state'][0] >= safety.min_buffer_s:
        estimate = safety.throughput_factor * len(samples) / np.sum(1 / np.maximum(samples, 1e-12))
        allowed = candidates[:, 0] <= estimate
    allowed[lowest] = True
    return allowed


def constrained_argmax(q_values, observations, safety=None):
    """Filter real actions and padding before Double DQN chooses a next action."""
    masks = torch.zeros_like(q_values, dtype=torch.bool)
    for i, observation in enumerate(observations):
        mask = allowed_actions(observation, safety)
        masks[i, :len(mask)] = torch.as_tensor(mask, device=q_values.device)
    return q_values.masked_fill(~masks, -torch.inf).argmax(dim=1)
