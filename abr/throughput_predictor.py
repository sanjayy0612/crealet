"""Supervised conservative forecast of near-future controlled application goodput."""
import numpy as np
import torch
from torch import nn


HISTORY_LENGTH = 8
FEATURE_COUNT = HISTORY_LENGTH * 5 + 5 + 1


def features(observation):
    history = np.asarray(observation['history'], dtype=np.float32).copy()
    if history.shape != (HISTORY_LENGTH, 5):
        raise ValueError('Predictor requires eight history rows with five features each')
    # Goodput is strongly right-skewed. Log compression keeps the small network
    # numerically stable while preserving a separate valid-sample indicator.
    history[:, 0] = np.log1p(np.maximum(history[:, 0], 0.0))
    history[:, 1:4] = np.clip(history[:, 1:4], 0.0, 30.0)
    state = np.asarray(observation['state'], dtype=np.float32).copy()
    state[0] = np.clip(state[0] / 60.0, 0.0, 2.0)
    state[1] = np.clip(state[1] / 30.0, 0.0, 2.0)
    state[2] = np.clip(state[2], 0.0, 2.0)
    state[3] = np.clip(state[3], 0.0, 1.0)
    state[4] = np.log1p(max(state[4], 0.0))
    duration = float(observation['candidates'][0, 2]) / 10.0
    return np.concatenate((history.reshape(-1), state, [duration])).astype(np.float32)


class QuantileNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURE_COUNT, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, value):
        # Output is log1p(Mbps), so capacity predictions cannot be negative.
        return self.layers(value).squeeze(-1)


class LearnedThroughputPredictor:
    def __init__(self, model):
        self.model = model.eval()

    @classmethod
    def load(cls, path, *, expected_dataset_fingerprint=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('schema_version') != 1:
            raise ValueError('Unsupported throughput predictor checkpoint')
        if (expected_dataset_fingerprint is not None
                and checkpoint['training_dataset_fingerprint'] != expected_dataset_fingerprint):
            raise ValueError('Predictor training dataset fingerprint mismatch')
        model = QuantileNetwork()
        model.load_state_dict(checkpoint['state_dict'])
        return cls(model)

    def predict(self, observation):
        value = torch.from_numpy(features(observation)).unsqueeze(0)
        with torch.no_grad():
            predicted_log_mbps = float(self.model(value).item())
        predicted_mbps = float(np.expm1(np.clip(predicted_log_mbps, -20.0, 20.0)))
        return max(0.0, predicted_mbps), {'quantile': 0.2}


def future_mean_capacity(trace, start_s, horizon_s):
    """Integrate trace capacity over a future window; return None past trace end."""
    ends = np.cumsum(trace.durations_s)
    end_s = start_s + horizon_s
    if start_s < 0 or end_s > ends[-1] + 1e-9:
        return None
    cursor = start_s
    total_kbps_s = 0.0
    while cursor < end_s - 1e-9:
        index = min(int(np.searchsorted(ends, cursor, side='right')), len(ends) - 1)
        interval_start = 0.0 if index == 0 else float(ends[index - 1])
        slice_end = min(end_s, float(ends[index]))
        total_kbps_s += float(trace.bandwidth_kbps[index]) * (slice_end - cursor)
        cursor = slice_end
    return total_kbps_s / horizon_s / 1000.0


def harmonic_robust_prediction(observation, config):
    history = np.asarray(observation['history'], dtype=np.float64)
    rates = history[history[:, -1] > 0, 0]
    if not len(rates):
        return 0.0
    recent = rates[-config.throughput_window:]
    estimate = len(recent) / np.sum(1.0 / np.maximum(recent, 1e-12))
    first = max(1, len(rates) - config.error_window)
    errors = []
    for index in range(first, len(rates)):
        prior = rates[max(0, index - config.throughput_window):index]
        if len(prior):
            hm = len(prior) / np.sum(1.0 / np.maximum(prior, 1e-12))
            errors.append(abs(float(rates[index]) - hm) / max(float(rates[index]), 1e-12))
    return float(estimate / (1.0 + min(1e6, max(errors, default=0.0))))
