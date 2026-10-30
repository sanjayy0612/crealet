"""Version 2 streaming environment: finite VOD or finite low-latency live event.

An action selects a representation for one segment. Live parts share one request.
This is application-level replay, not a packet emulator or dash.js implementation.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .data import Dataset
from .media import Media
from .playback import Player
from .transport import Replay
from .recovery import RecoveryConfig


@dataclass(frozen=True)
class StreamConfig:
    mode: str = "vod"
    max_buffer_s: float = 30.
    startup_s: float = 1.
    resume_s: float = 1.
    request_timeout_s: float = 30.
    source_age_s: float = 3.
    encoder_delay_s: float = 0.
    target_latency_s: float = 3.
    catchup: bool = True
    min_rate: float = 0.95
    max_rate: float = 1.05
    catchup_gain: float = 0.03
    latency_deadband_s: float = 0.1
    control_tick_s: float = 0.1
    known_vod_sizes: bool = False

    def __post_init__(self):
        if self.mode not in {"vod", "live"}:
            raise ValueError("mode must be vod or live")
        positive = (self.max_buffer_s, self.request_timeout_s, self.target_latency_s, self.control_tick_s)
        nonnegative = (self.startup_s, self.resume_s, self.source_age_s, self.encoder_delay_s,
                       self.catchup_gain, self.latency_deadband_s)
        if any(not np.isfinite(v) or v <= 0 for v in positive) or any(not np.isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError("Invalid timing/control configuration")
        if not (np.isfinite(self.min_rate) and np.isfinite(self.max_rate) and 0 < self.min_rate <= 1 <= self.max_rate):
            raise ValueError("Playback rate bounds must include 1")
        if self.known_vod_sizes and self.mode == "live":
            raise ValueError("Future live part sizes cannot be supplied to the policy")


@dataclass(frozen=True)
class QoEWeights:
    # Explicit experimental objective, not a universal human quality score.
    rebuffer: float = 4.3
    startup: float = 1.
    switch: float = 1.
    latency: float = 0.1
    rate_deviation: float = 1.

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Reward weights must be finite and nonnegative")


class StreamingEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_version = 2

    def __init__(self, dataset: Dataset, *, media: Media | None = None, split="train",
                 config: StreamConfig | None = None, weights: QoEWeights | None = None, history=8,
                 recovery: RecoveryConfig | None = None):
        super().__init__()
        self.dataset, self.config, self.weights = dataset, config or StreamConfig(), weights or QoEWeights()
        self.media = media or Media(dataset.video)
        self.video = dataset.video
        if split not in dataset.splits:
            raise ValueError("Unknown split")
        if not isinstance(history, int) or isinstance(history, bool) or history < 1:
            raise ValueError("Positive history length required")
        # Avoid accidentally associating another video's part sizes with this ladder.
        for key in ("bitrates_kbps", "durations_s", "sizes_bytes"):
            if not np.array_equal(getattr(self.media.video, key), getattr(self.video, key)):
                raise ValueError("Media and dataset videos differ")
        self.live = self.config.mode == "live"
        if recovery is not None and (not isinstance(recovery, RecoveryConfig) or self.live):
            raise ValueError('Request recovery currently requires VOD and RecoveryConfig')
        self.recovery = recovery
        self.units = [self.media.units(i, self.live) for i in range(len(self.video.durations_s))]
        largest = max(float(ds.max()) for ds, _ in self.units)
        # Enough room to build a threshold and receive the next indivisible unit.
        if self.config.max_buffer_s < largest + max(self.config.startup_s, self.config.resume_s):
            raise ValueError("Buffer must hold a playable unit plus the startup/resume threshold")
        self.split, self.history_length = split, history
        self.segment_starts = np.r_[0., np.cumsum(self.video.durations_s)[:-1]]
        self.total_s = float(self.video.durations_s.sum())
        self.action_space = spaces.Discrete(len(self.video.bitrates_kbps))
        self.observation_space = spaces.Dict({
            "history": spaces.Box(0., np.inf, (history, 5), dtype=np.float32),
            "state": spaces.Box(0., np.inf, (5,), dtype=np.float32),
            "candidates": spaces.Box(0., np.inf, (self.action_space.n, 3), dtype=np.float32),
        })
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        traces = self.dataset.splits[self.split]
        index = opts.get("trace_index", int(self.np_random.integers(len(traces))))
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(traces):
            raise ValueError("Invalid trace_index")
        self.network = Replay(traces[index], opts.get("trace_offset_s", 0.))
        self.player = Player(self.config.startup_s, self.config.resume_s, self.video.bitrates_kbps[0])
        self.wall_s = 0.
        self.segment = 0
        self.previous_action = None
        self.history = np.zeros((self.history_length, 5), dtype=np.float32)
        self.downloaded_bytes = self.downloaded_media_s = 0.
        self.request_retries = self.abandoned_requests = 0
        self.wasted_bytes = 0.
        self._done = False
        return self._observation(), {"trace": traces[index].name, "observation_version": 2}

    @property
    def live_edge_s(self):
        return min(self.total_s, self.config.source_age_s + self.wall_s)

    @property
    def live_latency_s(self):
        return max(0., self.live_edge_s - self.player.playhead_s) if self.live else 0.

    def _rate(self):
        c, p = self.config, self.player
        if not self.live or not c.catchup or not p.playing:
            return 1.
        # Simple bounded proportional controller. It is not LoL+ or dash.js catch-up.
        drift = self.live_latency_s - c.target_latency_s
        if abs(drift) <= c.latency_deadband_s:
            return 1.
        if p.buffer_s < c.resume_s and drift > 0:
            return 1.
        return float(np.clip(1 + c.catchup_gain * drift, c.min_rate, c.max_rate))

    def _advance(self, duration):
        remaining = duration
        while remaining > 1e-9:
            dt = min(remaining, self.config.control_tick_s)
            # Split at source end so the live edge's slope is well defined.
            to_end = self.total_s - self.config.source_age_s - self.wall_s
            if self.live and 1e-9 < to_end < dt:
                dt = to_end
            self.player.advance(dt, self._rate(), edge_start=self.live_edge_s if self.live else None,
                                edge_speed=float(to_end > 1e-9), target_latency=self.config.target_latency_s)
            self.wall_s += dt
            remaining -= dt

    def _wait(self, seconds, deadline=float("inf")):
        elapsed, complete = self.network.wait(seconds, deadline)
        self._advance(elapsed)
        return elapsed, complete

    def _observation(self):
        candidates = np.zeros((self.action_space.n, 3), dtype=np.float32)
        if self.segment < len(self.video.durations_s):
            duration = self.video.durations_s[self.segment]
            estimates = self.video.bitrates_kbps * 1000 / 8 * duration
            if self.config.known_vod_sizes:
                estimates = self.video.sizes_bytes[self.segment]
            candidates[:] = np.column_stack((self.video.bitrates_kbps / 1000, estimates / 1e6,
                                             np.full(self.action_space.n, duration)))
        previous = 0. if self.previous_action is None else self.video.bitrates_kbps[self.previous_action] / 1000
        return {"history": self.history.copy(), "candidates": candidates,
                "state": np.array([self.player.buffer_s, self.live_latency_s, self._rate(),
                                   1 - self.segment / len(self.video.durations_s), previous], dtype=np.float32)}

    def metrics(self):
        s = dict(self.player.stats)
        s.update(buffer_s=self.player.buffer_s, peak_buffer_s=self.player.peak_buffer_s,
                 wall_time_s=self.wall_s, downloaded_bytes=self.downloaded_bytes,
                 downloaded_media_s=self.downloaded_media_s,
                 completion_fraction=self.player.playhead_s / self.total_s,
                 mean_played_bitrate_kbps=s["bitrate_integral"] / max(s["played_s"], 1e-12),
                 mean_live_latency_s=s["latency_integral"] / max(s["latency_wall_s"], 1e-12),
                 stall_ratio=s["rebuffer_s"] / max(s["rebuffer_s"] + s["playback_wall_s"], 1e-12))
        if self.recovery is not None:
            s.update(request_retries=self.request_retries, abandoned_requests=self.abandoned_requests,
                     wasted_bytes=self.wasted_bytes)
        return s

    def _recover_vod_transfer(self, action, duration, sizes, deadline):
        """Restart cancelled bodies at lowest quality; every byte and RTT counts.

        Actual size is known only for the active request (Content-Length). A
        proposed lower representation uses the same nominal size estimate as
        the normal observation, never its unrequested true future size.
        """
        cfg = self.recovery
        lowest = 0  # Video contract requires a strictly increasing ladder.
        estimate = self.video.bitrates_kbps[lowest] * 1000 / 8 * duration
        attempts = []
        total_bytes = total_body_s = extra_rtt_s = backoff_s = 0.
        complete = False
        exhausted = False
        for attempt in range(cfg.max_retries + 1):
            body_start = self.wall_s
            last_progress = self.wall_s
            received = 0.
            size = float(sizes[action])
            reason = None
            probe = None
            while (received < size - 1e-7 and self.network.time_s < deadline - 1e-9
                   and self.network.time_s < self.network.ends[-1] - 1e-9):
                transfer = self.network.transfer(size - received,
                    min(deadline, self.network.time_s + cfg.poll_s))
                self._advance(transfer.elapsed_s)
                total_body_s += transfer.elapsed_s
                total_bytes += transfer.bytes_received
                received += transfer.bytes_received
                if transfer.bytes_received > 1e-7:
                    last_progress = self.wall_s
                if received >= size - 1e-7:
                    complete = True
                    break
                # A deadline/trace boundary is a failed transfer, not an early
                # controller cancellation (and must retain timeout backoff).
                if self.network.time_s >= min(deadline, self.network.ends[-1]) - 1e-9:
                    break
                probe = dict(elapsed_s=self.wall_s - body_start, received_bytes=received,
                             content_bytes=size, idle_s=self.wall_s - last_progress,
                             buffer_s=self.player.buffer_s, lowest_estimate_bytes=float(estimate),
                             at_lowest=action == lowest)
                reason = cfg.abandon_reason(**probe) if attempt < cfg.max_retries else None
                if reason or transfer.elapsed_s <= 1e-12:
                    break
            trace_end = self.network.time_s >= self.network.ends[-1] - 1e-9
            outcome = 'complete' if complete else reason or ('trace_end' if trace_end else 'request_timeout')
            attempts.append({'attempt': attempt, 'representation': int(action),
                             'bitrate_kbps': float(self.video.bitrates_kbps[action]),
                             'body_start_s': body_start, 'body_end_s': self.wall_s,
                             'content_bytes': size, 'bytes_received': received,
                             'outcome': outcome, 'last_probe': probe})
            if complete:
                break
            self.wasted_bytes += received
            self.abandoned_requests += int(reason is not None)
            if trace_end:
                break
            if attempt == cfg.max_retries:
                exhausted = True
                break
            # Back off after a hard timeout or no progress, not after an active
            # but overly large download. Backoff drains playback normally.
            if outcome in ('request_timeout', 'no_progress'):
                waited, ok = self._wait(cfg.backoff_s)
                backoff_s += waited
                if not ok or self.network.time_s >= self.network.ends[-1] - 1e-9:
                    break
            deadline = self.network.time_s + self.config.request_timeout_s
            action = lowest
            self.request_retries += 1
            waited, ok = self._wait(self.network.rtt_s, deadline)
            extra_rtt_s += waited
            if not ok:
                attempts.append({'attempt': attempt + 1, 'representation': int(action),
                                 'bitrate_kbps': float(self.video.bitrates_kbps[action]),
                                 'body_start_s': self.wall_s, 'body_end_s': self.wall_s,
                                 'content_bytes': None, 'bytes_received': 0.,
                                 'outcome': ('trace_end' if self.network.time_s >= self.network.ends[-1] - 1e-9
                                             else 'request_timeout'), 'last_probe': None})
                break
        return {'complete': complete, 'action': action, 'bytes': total_bytes,
                'body_s': total_body_s, 'rtt_s': extra_rtt_s, 'backoff_s': backoff_s,
                'attempts': attempts, 'exhausted': exhausted}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before step")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("Action must be an integer representation index")
        action = int(action)
        policy_action = action
        segment_index = self.segment
        before = dict(self.player.stats)
        start = self.wall_s
        ds, bs = self.units[self.segment]
        request_start = self.network.time_s
        deadline = request_start + self.config.request_timeout_s
        rtt, complete = self._wait(self.network.rtt_s, deadline)
        source_wait = buffer_wait = body_time = received = 0.
        events = []
        recovery_result = None
        media_end = self.segment_starts[self.segment]
        for part, duration in enumerate(ds):
            if not complete:
                break
            media_end += duration
            if self.live:
                available_at = media_end + self.config.encoder_delay_s - self.config.source_age_s
                waited, complete = self._wait(max(0., available_at - self.wall_s), deadline)
                source_wait += waited
            if not complete:
                break
            # Application backpressure approximation; not TCP receive-window simulation.
            while self.player.buffer_s + duration > self.config.max_buffer_s + 1e-9 and complete:
                waited, complete = self._wait(min(self.config.control_tick_s,
                    (self.player.buffer_s + duration - self.config.max_buffer_s) / self._rate()), deadline)
                buffer_wait += waited
            if not complete:
                break
            if self.recovery is not None:
                recovery_result = self._recover_vod_transfer(action, duration, bs[part], deadline)
                complete, action = recovery_result['complete'], recovery_result['action']
                body_time += recovery_result['body_s']
                received += recovery_result['bytes']
                rtt += recovery_result['rtt_s']
            else:
                transfer = self.network.transfer(float(bs[part, action]), deadline)
                self._advance(transfer.elapsed_s)
                body_time += transfer.elapsed_s
                received += transfer.bytes_received
                complete = transfer.complete
            if complete:
                self.player.append(duration, self.video.bitrates_kbps[action])
                self.downloaded_media_s += duration
                events.append({"part": part, "received_at_s": self.wall_s,
                               "media_end_s": float(media_end), "buffer_s": self.player.buffer_s,
                               "playhead_s": self.player.playhead_s})
        self.downloaded_bytes += received
        request_s = self.wall_s - start
        if complete:
            self.history = np.roll(self.history, -1, axis=0)
            # Body time excludes RTT, producer idle, and player backpressure; outages remain.
            self.history[-1] = [received * 8 / max(body_time, 1e-12) / 1e6, request_s / 10,
                               source_wait / 10, rtt / 10, 1.]
            self.previous_action = action
            self.segment += 1
        terminated = complete and self.segment == len(self.units)
        if terminated:
            self.player.end()
            # Playback tail needs no further network service, even if trace has ended.
            while self.player.buffer_s > 1e-9:
                self._advance(min(self.config.control_tick_s, self.player.buffer_s / self._rate()))
        truncated = not complete
        self._done = terminated or truncated
        delta = {k: self.player.stats[k] - before[k] for k in before}
        w = self.weights
        terms = {"quality": delta["quality_integral"], "startup": -w.startup * delta["startup_s"],
                 "rebuffer": -w.rebuffer * delta["rebuffer_s"], "switch": -w.switch * delta["switch_log"],
                 "latency": -w.latency * delta["latency_excess_integral"],
                 "rate_deviation": -w.rate_deviation * delta["rate_deviation_s"]}
        reason = ("playback_end" if terminated else
                  "trace_end" if truncated and self.network.time_s >= self.network.ends[-1] - 1e-9 else
                  "retry_limit" if truncated and recovery_result is not None and recovery_result['exhausted'] else
                  "request_timeout" if truncated else None)
        info = {"trace": self.network.trace.name, "segment_complete": complete,
                "segment_index": segment_index, "elapsed_s": self.wall_s - start,
                "playback_tail_s": max(0., self.wall_s - start - request_s),
                "request_s": request_s, "rtt_s": rtt, "body_transfer_s": body_time,
                "source_wait_s": source_wait, "buffer_wait_s": buffer_wait,
                "bytes_received": received, "parts": events, "reward_terms": terms,
                "termination_reason": reason, "metrics": self.metrics()}
        if self.recovery is not None:
            info.update(policy_action=policy_action, effective_action=action,
                        request_attempts=[] if recovery_result is None else recovery_result['attempts'],
                        recovery_backoff_s=0. if recovery_result is None else recovery_result['backoff_s'])
        return self._observation(), float(sum(terms.values())), bool(terminated), bool(truncated), info

    def provenance(self):
        media = {"provenance": self.media.provenance,
                 "part_durations_s": None if self.media.part_durations_s is None else [x.tolist() for x in self.media.part_durations_s],
                 "part_sizes_bytes": None if self.media.part_sizes_bytes is None else [x.tolist() for x in self.media.part_sizes_bytes]}
        return {"environment_version": 2, "observation_version": 2,
                "request_recovery": None if self.recovery is None else self.recovery.to_dict(),
                "dataset_sha256": self.dataset.fingerprint,
                "media_sha256": hashlib.sha256(json.dumps(media, sort_keys=True).encode()).hexdigest(),
                "config": asdict(self.config), "weights": asdict(self.weights),
                "network_model": "finite application-goodput replay; no TCP/QUIC/radio/packet model",
                "quality_model": "playback-duration-weighted log bitrate proxy; not VMAF/MOS",
                "latency_metric": "live edge minus playhead; wall-time weighted after startup, including stalls"}
