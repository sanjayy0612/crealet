import numpy as np
import pytest

pytest.importorskip('torch')
from abr.env import StreamingEnv
from abr.recovery import RecoveryConfig
from scripts.diagnose_quality_gate import probe_upgrade
from tests.test_env import fixture


def test_counterfactual_does_not_advance_actual_player_network_or_history():
    data, media = fixture(segments=2, rates=(8000.,), intervals=(100.,),
                          sizes=np.array([[1000., 2000.], [1000., 2000.]]), rtt=0.)
    env = StreamingEnv(data, media=media, recovery=RecoveryConfig())
    obs, _ = env.reset(seed=0)
    probe = probe_upgrade(env, 1)
    assert probe['locally_safe']
    assert probe['actual_segment_bytes'] == 2000
    assert env.segment == 0 and env.network.time_s == 0 and env.wall_s == 0
    assert not env.player.started
    np.testing.assert_array_equal(env.history, obs['history'])
    # The original request must still start from reset after the offline branch.
    _, _, _, _, info = env.step(0)
    assert info['segment_index'] == 0
    assert info['request_s'] == pytest.approx(.001)


def test_counterfactual_rejects_recovery_to_lowest_even_without_stalls():
    data, media = fixture(segments=2, rates=(8.,), intervals=(200.,),
                          sizes=np.array([[1000., 100000.], [1000., 100000.]]), rtt=0.)
    env = StreamingEnv(data, media=media, recovery=RecoveryConfig(max_retries=1, early_abandon=False))
    env.reset(seed=0)
    probe = probe_upgrade(env, 1)
    assert probe['effective_action'] == 0
    assert not probe['delivered_requested_quality']
    assert probe['added_stall_s'] == 0
    assert not probe['locally_safe']
