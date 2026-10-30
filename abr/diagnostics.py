"""Offline service/demand diagnostics. Never supply these values to a policy.

These bounds apply to the simulator's serial, whole-segment VOD contract with
unit playback rate, exact trace start, and backpressure before body transfer.
They do not establish a physical bound for a different player/network model.
"""
import numpy as np

from .transport import Replay


def service_bytes(trace, start_s, end_s):
    """Integrate the recorded service over an absolute trace-time interval."""
    if not np.isfinite([start_s, end_s]).all() or not 0 <= start_s <= end_s:
        raise ValueError('Ordered finite nonnegative service interval required')
    ends = np.cumsum(trace.durations_s)
    starts = np.r_[0., ends[:-1]]
    overlaps = np.maximum(0., np.minimum(ends, end_s) - np.maximum(starts, start_s))
    return float(np.dot(overlaps, trace.bandwidth_kbps * 1000 / 8))


def optimistic_arrival_bound(video, trace, *, max_buffer_s, startup_budget_s=3., stall_budget_s=8.):
    """Necessary waiting under more favorable conditions than any real policy.

    At the start of segment i's body the VOD buffer must have room for that
    whole segment. Since played media cannot exceed wall time, body release is
    no earlier than max(0, cumulative_media_end[i] - buffer_cap). This uses zero
    startup/stalls to release space as early as possible. Bodies are serial.

    Transfer the smallest actual representation per segment at those releases,
    with zero RTT, no retries, no timeouts, and no wasted bytes. The resulting
    arrival times are lower bounds, not an implementable policy. If segment i
    arrives at a_i, total startup+stalls must be >= a_i - media_start[i]. Future
    trace values and exact candidate sizes are used only for this diagnostic.
    """
    if (not np.isfinite([max_buffer_s, startup_budget_s, stall_budget_s]).all() or
            max_buffer_s < float(max(video.durations_s)) or
            startup_budget_s < 0 or stall_budget_s < 0):
        raise ValueError('Buffer must hold a segment; finite nonnegative budgets required')
    network = Replay(trace)
    media_ends = np.cumsum(video.durations_s)
    rows = []
    required_waiting = 0.
    bottleneck = None
    for index, media_end in enumerate(media_ends):
        media_start = float(media_end - video.durations_s[index])
        release = max(0., float(media_end) - max_buffer_s)
        _, released = network.wait(max(0., release - network.time_s))
        representation = int(np.argmin(video.sizes_bytes[index]))
        size = float(video.sizes_bytes[index, representation])
        body_start = network.time_s
        result = network.transfer(size) if released else None
        complete = released and result.complete
        row = {'segment_index': index, 'representation': representation,
               'minimum_size_bytes': size, 'media_start_s': media_start,
               'optimistic_release_s': release, 'body_start_s': body_start,
               'earliest_arrival_s': network.time_s if complete else None,
               'complete': bool(complete)}
        rows.append(row)
        if not complete:
            break
        waiting = max(0., network.time_s - media_start)
        if waiting > required_waiting:
            required_waiting, bottleneck = waiting, index
    completed = len(rows) == len(media_ends) and all(r['complete'] for r in rows)
    first_startup = rows[0]['earliest_arrival_s']
    min_stalls = max(0., required_waiting - startup_budget_s) if completed else None
    return {'assumptions': 'offline lower bound: serial whole-segment VOD, unit playback rate, '
            'trace starts at zero, same pre-body buffer cap, smallest actual sizes, '
            'zero RTT, zero wasted bytes, no timeouts, earliest possible buffer releases',
            'max_buffer_s': max_buffer_s, 'startup_budget_s': startup_budget_s,
            'stall_budget_s': stall_budget_s, 'all_segments_arrive': bool(completed),
            'minimum_startup_s': first_startup,
            'startup_plus_stalls_lower_bound_s': required_waiting if completed else None,
            'stalls_lower_bound_with_startup_budget_s': min_stalls,
            'bottleneck_segment_index': bottleneck,
            'targets_impossible_under_relaxation': bool(not completed or
                first_startup > startup_budget_s + 1e-9 or min_stalls > stall_budget_s + 1e-9),
            'segments': rows}
