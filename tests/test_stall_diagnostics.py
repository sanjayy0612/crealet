import numpy as np
import pytest

from abr.data import Trace, Video
from abr.diagnostics import optimistic_arrival_bound, service_bytes
from abr.env import StreamingEnv, StreamConfig
from tests.test_env import fixture


def test_service_integral_clips_intervals_and_preserves_outages():
    trace = Trace('service', [1, 2, 1], [8, 0, 16])
    assert service_bytes(trace, .5, 3.5) == pytest.approx(1500.)
    assert service_bytes(trace, 1, 3) == 0
    assert service_bytes(trace, 0, 10) == pytest.approx(3000.)
    with pytest.raises(ValueError):
        service_bytes(trace, 2, 1)


def test_buffer_release_bound_exposes_service_that_cannot_be_prefetched():
    video = Video([8., 16.], [2., 2.], [[2000., 4000.], [2000., 4000.]])
    trace = Trace('gap', [1., 20., 100.], [8000., 0., 8000.])
    small = optimistic_arrival_bound(video, trace, max_buffer_s=2.)
    large = optimistic_arrival_bound(video, trace, max_buffer_s=4.)
    assert small['segments'][1]['optimistic_release_s'] == 2.
    assert small['segments'][1]['earliest_arrival_s'] == pytest.approx(21.002)
    assert small['startup_plus_stalls_lower_bound_s'] == pytest.approx(19.002)
    assert small['stalls_lower_bound_with_startup_budget_s'] == pytest.approx(16.002)
    assert small['targets_impossible_under_relaxation']
    assert large['startup_plus_stalls_lower_bound_s'] == pytest.approx(.002)
    assert not large['targets_impossible_under_relaxation']


def test_bound_uses_smallest_actual_file_and_checks_startup_separately():
    video = Video([8., 16.], [2.], [[2000., 1000.]])
    trace = Trace('constant', [100.], [8.])
    result = optimistic_arrival_bound(video, trace, max_buffer_s=4., startup_budget_s=.5)
    assert result['segments'][0]['representation'] == 1
    assert result['minimum_startup_s'] == 1.
    assert result['targets_impossible_under_relaxation']


def test_finite_trace_exhaustion_is_not_converted_to_infinite_or_repeated_service():
    video = Video([8.], [2., 2.], [[2000.], [2000.]])
    result = optimistic_arrival_bound(video, Trace('short', [1.], [8.]), max_buffer_s=4.)
    assert not result['all_segments_arrive']
    assert result['segments'][0]['earliest_arrival_s'] is None
    assert result['stalls_lower_bound_with_startup_budget_s'] is None
    assert result['targets_impossible_under_relaxation']


@pytest.mark.parametrize('seed', range(5))
def test_bound_never_exceeds_waiting_in_executed_serial_vod(seed):
    rng = np.random.default_rng(seed)
    data, _ = fixture(segments=10, rates=(80., 0., 16., 800.), intervals=(2., 5., 10., 500.),
                      sizes=rng.uniform(1000., 12000., (10, 2)), rtt=.1)
    env = StreamingEnv(data, config=StreamConfig(max_buffer_s=6., request_timeout_s=1000.))
    env.reset(options={'trace_index': 0})
    while True:
        _, _, done, truncated, _ = env.step(int(rng.integers(2)))
        if done or truncated:
            break
    assert done
    bound = optimistic_arrival_bound(data.video, data.splits['train'][0], max_buffer_s=6.)
    assert bound['all_segments_arrive']
    metrics = env.metrics()
    assert bound['startup_plus_stalls_lower_bound_s'] <= metrics['startup_s'] + metrics['rebuffer_s'] + 1e-7


def test_bound_rejects_invalid_buffer():
    video = Video([8.], [2.], [[1000.]])
    with pytest.raises(ValueError):
        optimistic_arrival_bound(video, Trace('constant', [10.], [8.]), max_buffer_s=1.)
