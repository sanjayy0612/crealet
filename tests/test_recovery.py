import numpy as np
import pytest

from abr.env import StreamingEnv, StreamConfig
from abr.recovery import RecoveryConfig
from tests.test_env import fixture


def make_env(*, recovery=None, **kwargs):
    config = kwargs.pop('config', StreamConfig())
    data, media = fixture(**kwargs)
    return StreamingEnv(data, media=media, config=config, recovery=recovery)


def step(env, action=0):
    env.reset(seed=1)
    return env.step(action)


def test_controller_uses_progress_and_avoids_nearly_finished_or_lowest_requests():
    cfg = RecoveryConfig()
    probe = dict(elapsed_s=3., received_bytes=100., content_bytes=10000.,
                 idle_s=0., buffer_s=1., lowest_estimate_bytes=1000., at_lowest=False)
    assert cfg.abandon_reason(**probe) == 'buffer_risk'
    assert cfg.abandon_reason(**(probe | {'idle_s': 3.})) == 'no_progress'
    for change in ({'at_lowest': True}, {'elapsed_s': 1.}, {'received_bytes': 9500.},
                   {'buffer_s': 400.}, {'received_bytes': 0.}):
        assert cfg.abandon_reason(**(probe | change)) is None
    assert RecoveryConfig(early_abandon=False).abandon_reason(**probe) is None


@pytest.mark.parametrize('kwargs', [{'max_retries': -1}, {'max_retries': True},
    {'poll_s': 0}, {'no_progress_s': float('nan')}, {'min_observation_s': -1},
    {'backoff_s': -1}, {'buffer_reserve_s': float('inf')}, {'restart_fraction': 0},
    {'restart_fraction': 1.1}, {'early_abandon': 1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RecoveryConfig(**kwargs)


def test_timeout_retry_completes_after_outage_and_charges_rtt_and_backoff():
    args = dict(segments=1, rates=(0., 8.), intervals=(3., 20.), rtt=.1,
                config=StreamConfig(request_timeout_s=2.))
    _, _, term, trunc, original = step(make_env(**args))
    assert trunc and not term and original['termination_reason'] == 'request_timeout'
    _, _, term, trunc, row = step(make_env(recovery=RecoveryConfig(backoff_s=1.), **args))
    assert term and not trunc
    assert row['metrics']['request_retries'] == 1
    assert row['rtt_s'] == pytest.approx(.2)
    assert row['recovery_backoff_s'] == pytest.approx(1.)
    assert row['request_s'] == pytest.approx(4.1)
    assert row['metrics']['wall_time_s'] == pytest.approx(6.1)
    assert row['metrics']['wasted_bytes'] == 0


def test_abandon_downgrades_same_segment_and_counts_discarded_bytes():
    # Lowest nominal estimate is 125000 bytes; high request has 1 MB left.
    env = make_env(segments=1, sizes=[[1000., 1000000.]], rates=(8.,),
                   intervals=(100.,), recovery=RecoveryConfig())
    obs, _, term, trunc, row = step(env, 1)
    assert term and not trunc
    assert row['policy_action'] == 1 and row['effective_action'] == 0
    assert row['segment_index'] == 0 and len(row['parts']) == 1
    assert [x['outcome'] for x in row['request_attempts']] == ['buffer_risk', 'complete']
    assert row['metrics']['wasted_bytes'] == pytest.approx(2000.)
    assert row['metrics']['downloaded_bytes'] == pytest.approx(3000.)
    assert row['metrics']['mean_played_bitrate_kbps'] == pytest.approx(500.)
    assert row['metrics']['startup_s'] == pytest.approx(3.)
    assert row['metrics']['wall_time_s'] == pytest.approx(5.)
    assert obs['state'][-1] == pytest.approx(.5)
    assert env.history[-1, 0] == pytest.approx(.008)


def test_failed_partial_bodies_are_discarded_in_full():
    env = make_env(segments=1, sizes=[[3000., 6000.]], rates=(8.,), intervals=(30.,),
                   config=StreamConfig(request_timeout_s=2.),
                   recovery=RecoveryConfig(max_retries=2, backoff_s=1.))
    _, _, term, trunc, row = step(env)
    assert trunc and not term and row['termination_reason'] == 'retry_limit'
    assert row['request_s'] == pytest.approx(8.)
    assert row['metrics']['wasted_bytes'] == pytest.approx(6000.)
    assert row['metrics']['downloaded_bytes'] == pytest.approx(6000.)
    assert row['metrics']['request_retries'] == 2
    assert row['metrics']['downloaded_media_s'] == 0


def test_permanent_outage_is_bounded_and_finite_trace_has_priority():
    args = dict(segments=1, rates=(0.,), config=StreamConfig(request_timeout_s=2.),
                recovery=RecoveryConfig(max_retries=2, backoff_s=1.))
    _, _, _, trunc, row = step(make_env(**args, intervals=(30.,)))
    assert trunc and row['termination_reason'] == 'retry_limit'
    assert row['request_s'] == pytest.approx(8.)
    _, _, _, trunc, row = step(make_env(**args, intervals=(2.5,)))
    assert trunc and row['termination_reason'] == 'trace_end'
    assert row['metrics']['request_retries'] == 0  # Trace ends during backoff.
    assert row['request_s'] == pytest.approx(2.5)


def test_retry_cut_short_during_headers_is_recorded():
    env = make_env(segments=1, rates=(0.,), intervals=(3.,), rtt=.8,
                   config=StreamConfig(request_timeout_s=2.),
                   recovery=RecoveryConfig(backoff_s=.5))
    _, _, _, trunc, row = step(env)
    assert trunc and row['termination_reason'] == 'trace_end'
    assert row['metrics']['request_retries'] == 1
    assert row['rtt_s'] == pytest.approx(1.3)
    assert row['request_attempts'][-1]['content_bytes'] is None
    assert row['request_attempts'][-1]['outcome'] == 'trace_end'


@pytest.mark.parametrize('trace_s,expected', [(2., 'trace_end'), (100., 'request_timeout')])
def test_boundary_failure_is_not_counted_as_early_abandon(trace_s, expected):
    env = make_env(segments=1, sizes=[[1000., 1000000.]], rates=(8.,), intervals=(trace_s,),
                   config=StreamConfig(request_timeout_s=2.), recovery=RecoveryConfig())
    _, _, _, _, row = step(env, 1)
    assert row['request_attempts'][0]['outcome'] == expected
    assert row['metrics']['abandoned_requests'] == 0
    if expected == 'request_timeout':
        assert row['recovery_backoff_s'] == pytest.approx(2.)


@pytest.mark.parametrize('rates,intervals', [((8.,), (100.,)), ((0.,), (100.,)),
                                          ((8., 0., 16.), (1., 2., 100.))])
def test_zero_retries_preserves_original_transfer_and_playback(rates, intervals):
    args = dict(segments=1, sizes=[[3000., 6000.]], rates=rates, intervals=intervals,
                rtt=.1, config=StreamConfig(request_timeout_s=4.))
    original = step(make_env(**args), 1)
    recovered = step(make_env(recovery=RecoveryConfig(max_retries=0), **args), 1)
    for key in original[0]:
        np.testing.assert_allclose(original[0][key], recovered[0][key], atol=1e-6)
    assert recovered[1:4] == pytest.approx(original[1:4])
    for key, value in original[4]['metrics'].items():
        assert recovered[4]['metrics'][key] == pytest.approx(value, abs=1e-7)
    assert recovered[4]['bytes_received'] == pytest.approx(original[4]['bytes_received'])


def test_recovery_rejects_live_mode():
    with pytest.raises(ValueError, match='requires VOD'):
        make_env(config=StreamConfig(mode='live'), recovery=RecoveryConfig())
