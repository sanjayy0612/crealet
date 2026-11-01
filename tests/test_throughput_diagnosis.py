import numpy as np
import pytest

pytest.importorskip('torch')
from abr.models.safety import ActionSafety, allowed_actions
from scripts.diagnose_throughput import estimate_kbps, window_observation


def test_minimal_slow_sample_locks_floor_despite_three_fast_downloads():
    # Minimal 8-vs-3 pattern: remove any fast row and the short window includes
    # the slow row too. A captured pattern, not a claim the shorter rule is safe.
    history = np.zeros((8, 5), dtype=np.float32)
    history[-4:, 0] = [.029, 2., 2., 2.]
    history[-4:, -1] = 1
    obs = {'history': history, 'state': np.array([60., 0, 1, 1, 0]),
           'candidates': np.array([[.008, .004, 4.], [.3, .15, 4.]])}
    safety = ActionSafety(min_buffer_s=4.)
    assert not allowed_actions(obs, safety)[1]
    assert allowed_actions(window_observation(obs, 3), safety)[1]
    assert estimate_kbps(obs, safety, 8) < 300 < estimate_kbps(obs, safety, 3)
    np.testing.assert_array_equal(obs['history'], history)
    # Buffer protection still applies even if the latest rate is high.
    low_buffer = {**obs, 'state': np.array([2., 0, 1, 1, 0])}
    assert not allowed_actions(window_observation(low_buffer, 3), safety)[1]


def test_shorter_history_does_not_create_measurements_or_remove_valid_newest():
    history = np.zeros((8, 5), dtype=np.float32)
    obs = {'history': history}
    assert estimate_kbps(obs, ActionSafety(), 3) is None
    history[-1] = [2, 0, 0, 0, 1]
    copy = window_observation(obs, 1)
    copy['history'][-1, 0] = 99
    assert history[-1, 0] == 2
