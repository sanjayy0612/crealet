"""Playable-unit queue, startup/resume thresholds, and playback-weighted metrics."""
from collections import deque
import math

EPS = 1e-9


def positive_linear_area(a, b, dt):
    if a >= 0 and b >= 0:
        return (a + b) * dt / 2
    if a <= 0 and b <= 0:
        return 0.
    return max(a, b) ** 2 * dt / (2 * abs(b - a))


class Player:
    def __init__(self, startup_s, resume_s, min_bitrate):
        self.startup_threshold, self.resume_threshold = startup_s, resume_s
        self.min_bitrate = min_bitrate
        self.queue = deque()
        self.buffer_s = self.playhead_s = self.peak_buffer_s = 0.
        self.started = self.playing = self.ended = self.stalled = False
        self.previous_bitrate = None
        self.stats = dict(startup_s=0., rebuffer_s=0., stall_events=0, played_s=0.,
                          playback_wall_s=0., quality_integral=0., bitrate_integral=0.,
                          switch_log=0., rate_deviation_s=0., latency_integral=0.,
                          latency_excess_integral=0., latency_wall_s=0., max_live_latency_s=0.)

    def _resume(self):
        threshold = self.resume_threshold if self.started else self.startup_threshold
        if self.buffer_s > EPS and (self.buffer_s + EPS >= threshold or self.ended):
            self.started = self.playing = True
            self.stalled = False

    def append(self, duration, bitrate, quality_utility=None):
        if quality_utility is not None:
            quality_utility = float(quality_utility)
            if not math.isfinite(quality_utility) or quality_utility < 0:
                raise ValueError('Quality utility must be finite and nonnegative')
        self.queue.append([float(duration), float(bitrate), quality_utility])
        self.buffer_s += duration
        self.peak_buffer_s = max(self.peak_buffer_s, self.buffer_s)
        self._resume()

    def end(self):
        self.ended = True
        self._resume()

    def _latency(self, a, b, dt, target):
        self.stats["latency_integral"] += (a + b) * dt / 2
        self.stats["latency_wall_s"] += dt
        self.stats["latency_excess_integral"] += positive_linear_area(a - target, b - target, dt)
        self.stats["max_live_latency_s"] = max(self.stats["max_live_latency_s"], a, b)

    def advance(self, duration, rate, *, edge_start=None, edge_speed=1., target_latency=0.):
        elapsed = 0.
        while elapsed < duration - EPS:
            left = duration - elapsed
            if not self.playing or not self.queue:
                if self.ended and not self.queue:
                    break
                if self.started:
                    if not self.stalled:
                        self.stats["stall_events"] += 1
                        self.stalled = True
                    self.stats["rebuffer_s"] += left
                    if edge_start is not None:
                        a = max(0., edge_start + edge_speed * elapsed - self.playhead_s)
                        self._latency(a, a + edge_speed * left, left, target_latency)
                else:
                    self.stats["startup_s"] += left
                self.playing = False
                break
            unit = self.queue[0]
            dt = min(left, unit[0] / rate)
            media_s = min(unit[0], dt * rate)
            if self.previous_bitrate != unit[1]:
                if self.previous_bitrate is not None:
                    self.stats["switch_log"] += abs(math.log(unit[1] / self.previous_bitrate))
                self.previous_bitrate = unit[1]
            quality_utility = unit[2]
            if quality_utility is None:
                quality_utility = math.log(unit[1] / self.min_bitrate)
            self.stats["quality_integral"] += media_s * quality_utility
            self.stats["bitrate_integral"] += media_s * unit[1]
            self.stats["rate_deviation_s"] += dt * abs(rate - 1.)
            self.stats["playback_wall_s"] += dt
            if edge_start is not None:
                a = max(0., edge_start + edge_speed * elapsed - self.playhead_s)
                b = max(0., a + (edge_speed - rate) * dt)
                self._latency(a, b, dt, target_latency)
            self.playhead_s += media_s
            self.stats["played_s"] += media_s
            self.buffer_s = max(0., self.buffer_s - media_s)
            unit[0] -= media_s
            elapsed += dt
            if unit[0] <= EPS:
                self.queue.popleft()
            if not self.queue:
                self.playing = False
