"""Strict adapter from observation v2 and public StreamConfig to model features."""
import numpy as np
import torch
from ..env import StreamConfig
from .network import ModelConfig


def context(config: StreamConfig):
    live = config.mode == 'live'
    return np.array([float(live), config.target_latency_s if live else 0., config.max_buffer_s,
        config.startup_s, config.resume_s, float(live and config.catchup),
        config.min_rate if live and config.catchup else 1.,
        config.max_rate if live and config.catchup else 1.,
        config.catchup_gain if live and config.catchup else 0.,
        config.latency_deadband_s if live and config.catchup else 0.], dtype=np.float32)


def make_batch(observations, stream_configs, model_config: ModelConfig, *, device='cpu'):
    """Batch variable candidate counts by padding. No traces, video sizes or env internals.

    Player vector = raw state[5] plus public context[10]. Candidate and history
    units match observation v2 exactly. Transformations live inside the network.
    """
    if not observations or len(observations) != len(stream_configs):
        raise ValueError('Provide equally sized nonempty observation/config lists')
    records = []
    for obs, cfg in zip(observations, stream_configs):
        if not isinstance(cfg, StreamConfig) or set(obs) != {'history', 'state', 'candidates'}:
            raise ValueError('Expected observation v2 and explicit StreamConfig')
        arrays = {k: np.array(obs[k], dtype=np.float32, copy=True) for k in obs}
        h, s, a = (arrays[k] for k in ('history', 'state', 'candidates'))
        if h.shape != (model_config.history_length, 5) or s.shape != (5,) or a.ndim != 2 or a.shape[1] != 3 or not len(a):
            raise ValueError('Observation shape does not match the model contract')
        if any(not np.isfinite(x).all() or (x < 0).any() for x in arrays.values()):
            raise ValueError('Observation values must be finite and nonnegative')
        if not np.isin(h[:, 4], [0., 1.]).all() or s[3] > 1 or (a <= 0).any():
            raise ValueError('History mask, remaining fraction, or candidate metadata is invalid (terminal observation?)')
        records.append((h, np.r_[s, context(cfg)], a))
    maximum = max(len(a) for _, _, a in records)
    candidates = np.zeros((len(records), maximum, 3), dtype=np.float32)
    mask = np.zeros((len(records), maximum), dtype=bool)
    for i, (_, _, a) in enumerate(records):
        candidates[i, :len(a)] = a
        mask[i, :len(a)] = True
    return {'history': torch.tensor(np.stack([r[0] for r in records]), device=device),
            'player': torch.tensor(np.stack([r[1] for r in records]), device=device),
            'candidates': torch.tensor(candidates, device=device),
            'candidate_mask': torch.tensor(mask, device=device)}
