"""Small, trace-driven RobustMPC reference for finite VOD.

This is a receding-horizon controller. It enumerates candidate bitrate sequences,
simulates buffer/QoE over a short horizon, and returns the first action from the
best sequence. Future chunk sizes use nominal bitrate-times-duration estimates.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class RobustMPCConfig:
    horizon_chunks: int = 5
    throughput_window: int = 5
    error_window: int = 5
    rtt_s: float = 0.08

    def __post_init__(self):
        for name in ('horizon_chunks', 'throughput_window', 'error_window'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(self.rtt_s) or self.rtt_s < 0:
            raise ValueError('rtt_s must be finite and nonnegative')


class RobustMPCPolicy:
    """Five-chunk robustMPC using log-bitrate QoE and a conservative HM estimate.

    ``durations_s`` are public presentation metadata. The policy does not inspect
    future trace values or future realized segment sizes.
    """

    def __init__(self, durations_s, *, config=None, safety=None, throughput_predictor=None):
        self.durations_s = np.asarray(durations_s, dtype=np.float64)
        if (self.durations_s.ndim != 1 or not len(self.durations_s)
                or not np.isfinite(self.durations_s).all() or (self.durations_s <= 0).any()):
            raise ValueError('durations_s must be a nonempty vector of positive finite values')
        self.config = config or RobustMPCConfig()
        self.safety = safety
        self.throughput_predictor = throughput_predictor

    @staticmethod
    def _harmonic_mean(values):
        values = np.asarray(values, dtype=np.float64)
        positive = values[values > 0]
        return 0.0 if not len(positive) else float(len(positive) / np.sum(1.0 / positive))

    def predict_throughput(self, observation):
        if self.throughput_predictor is not None:
            estimate, details = self.throughput_predictor.predict(observation)
            estimate = float(estimate)
            if not np.isfinite(estimate) or estimate < 0:
                raise ValueError('Throughput predictor must return a finite nonnegative estimate')
            return estimate, {'predictor': 'injected', **details,
                              'predicted_mbps': estimate}
        history = np.asarray(observation['history'], dtype=np.float64)
        rates = history[history[:, -1] > 0, 0]
        if not len(rates):
            return 0.0, {'harmonic_mean_mbps': 0.0, 'max_relative_error': 0.0,
                         'samples': 0}
        recent = rates[-self.config.throughput_window:]
        estimate = self._harmonic_mean(recent)

        # Reconstruct past one-step harmonic-mean prediction errors from the
        # observed sequence. Each target is predicted only from earlier samples.
        first_error_index = max(1, len(rates) - self.config.error_window)
        errors = []
        for index in range(first_error_index, len(rates)):
            prior = rates[max(0, index - self.config.throughput_window):index]
            prediction = self._harmonic_mean(prior)
            actual = float(rates[index])
            errors.append(abs(actual - prediction) / max(actual, 1e-12))
        max_error = min(1e6, max(errors, default=0.0))
        robust = estimate / (1.0 + max_error)
        return robust, {'harmonic_mean_mbps': estimate,
                        'max_relative_error': max_error,
                        'samples': int(len(rates)),
                        'robust_mbps': robust}

    def _allowed(self, observation, bitrates_mbps):
        if self.safety is None:
            return np.ones(len(bitrates_mbps), dtype=bool)
        safety = self.safety
        samples = np.asarray(observation['history'])
        samples = samples[samples[:, -1] > 0, 0]
        allowed = np.zeros(len(bitrates_mbps), dtype=bool)
        if len(samples) and observation['state'][0] >= safety['min_buffer_s']:
            estimate = safety['throughput_factor'] * len(samples) / np.sum(1.0 / np.maximum(samples, 1e-12))
            allowed = bitrates_mbps <= estimate
        allowed[0] = True
        return allowed

    def plan(self, observation, stream_config, weights, *, segment_index=None):
        candidates = np.asarray(observation['candidates'], dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) < 2:
            raise ValueError('RobustMPC requires at least two candidates with rate, size and duration')
        bitrates_mbps = candidates[:, 0]
        if (not np.isfinite(candidates).all() or (bitrates_mbps <= 0).any()
                or (candidates[:, 1] <= 0).any() or (candidates[:, 2] <= 0).any()):
            raise ValueError('Candidate rates, sizes and durations must be positive and finite')
        if segment_index is None:
            segment_index = int(round((1.0 - float(observation['state'][3])) * len(self.durations_s)))
        if isinstance(segment_index, bool) or not isinstance(segment_index, (int, np.integer)):
            raise ValueError('segment_index must be an integer')
        if not 0 <= segment_index < len(self.durations_s):
            raise ValueError('segment_index is outside the presentation')
        remaining = len(self.durations_s) - int(segment_index)
        horizon = min(self.config.horizon_chunks, remaining)
        durations = self.durations_s[segment_index:segment_index + horizon]
        throughput_mbps, predictor = self.predict_throughput(observation)
        if throughput_mbps <= 0:
            # No measurements exist at startup. Stay at the floor until a
            # completed request provides an observed goodput sample.
            return 0, {'predicted_throughput': predictor, 'horizon_chunks': horizon,
                       'best_score': None, 'startup_fallback': True}

        allowed = self._allowed(observation, bitrates_mbps)
        choices = np.flatnonzero(allowed)
        if not len(choices):
            raise RuntimeError('RobustMPC action mask removed every candidate')
        buffer_start = float(observation['state'][0])
        previous_mbps = float(observation['state'][4])
        previous_kbps = previous_mbps * 1000.0 if previous_mbps > 0 else None
        min_kbps = float(bitrates_mbps.min() * 1000.0)
        max_buffer_s = float(stream_config.max_buffer_s)
        first_chunk = int(segment_index) == 0
        best_action, best_score = int(choices[0]), -np.inf

        for sequence in product(range(len(bitrates_mbps)), repeat=horizon):
            if sequence[0] not in choices:
                continue
            buffer_s = buffer_start
            previous = previous_kbps
            score = 0.0
            for horizon_index, action in enumerate(sequence):
                duration_s = float(durations[horizon_index])
                bitrate_kbps = float(bitrates_mbps[action] * 1000.0)
                # Candidate sizes are nominal bitrate-duration estimates. Their
                # download duration is therefore bitrate / predicted goodput.
                body_s = duration_s * bitrates_mbps[action] / throughput_mbps
                download_s = body_s + self.config.rtt_s

                # Model the environment's finite-buffer request backpressure.
                buffer_wait_s = max(0.0, buffer_s + duration_s - max_buffer_s)
                buffer_s = max(0.0, buffer_s - buffer_wait_s)
                if first_chunk and horizon_index == 0:
                    score -= weights.startup * download_s
                else:
                    rebuffer_s = max(0.0, download_s - buffer_s)
                    score -= weights.rebuffer * rebuffer_s
                    buffer_s = max(0.0, buffer_s - download_s)
                score += duration_s * np.log(bitrate_kbps / min_kbps)
                if previous is not None:
                    score -= weights.switch * abs(np.log(bitrate_kbps / previous))
                buffer_s = min(max_buffer_s, buffer_s + duration_s)
                previous = bitrate_kbps

            if score > best_score:
                best_score, best_action = float(score), int(sequence[0])

        return best_action, {
            'predicted_throughput': predictor,
            'horizon_chunks': int(horizon),
            'best_score': best_score,
            'candidate_count': int(len(bitrates_mbps)),
            'allowed_actions': choices.tolist(),
            'startup_fallback': False,
        }

    def __call__(self, observation, stream_config, weights, *, segment_index=None):
        return self.plan(observation, stream_config, weights, segment_index=segment_index)
