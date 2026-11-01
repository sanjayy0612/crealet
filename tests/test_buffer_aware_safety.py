from dataclasses import asdict

import numpy as np
import pytest

from abr.models.safety import ActionSafety
from abr.buffer_safety import BufferAwareSafety, allowed_actions
from abr.player_server import experimental_safety


def observation():
    history = np.zeros((8, 5), dtype=np.float32)
    history[:, 0] = [.029] * 5 + [2.] * 3
    history[:, 3] = .01
    history[:, -1] = 1
    return {'history': history, 'state': np.array([60., 0, 1, 1, .008]),
            'candidates': np.array([[.008, .004, 4.], [.3, .15, 4.], [.75, .375, 4.]])}


def test_recovery_unlocks_only_normal_floor_after_sustained_fast_samples():
    obs = observation()
    assert allowed_actions(obs, ActionSafety(min_buffer_s=4.)).tolist() == [True, False, False]
    assert allowed_actions(obs, BufferAwareSafety(min_buffer_s=4.)).tolist() == [True, True, False]
    reverse = {**obs, 'candidates': obs['candidates'][::-1].copy()}
    np.testing.assert_array_equal(allowed_actions(reverse, BufferAwareSafety(min_buffer_s=4.)), [False, True, True])


@pytest.mark.parametrize('change', ['one_fast', 'falling', 'low_buffer', 'huge_size', 'high_rtt', 'missing_floor', 'empty'])
def test_recovery_refuses_unconfirmed_or_expensive_upgrades(change):
    obs = observation()
    if change == 'one_fast': obs['history'][-3:-1, 0] = .029
    if change == 'falling': obs['history'][-1, 0] = .029
    if change == 'low_buffer': obs['state'][0] = 8
    if change == 'huge_size': obs['candidates'][1, 1] = 4
    if change == 'high_rtt': obs['history'][-3:, 3] = 2
    if change == 'missing_floor': obs['candidates'][1, 0] = .4
    if change == 'empty': obs['history'][:] = 0
    baseline = allowed_actions(obs, ActionSafety(min_buffer_s=4.))
    np.testing.assert_array_equal(allowed_actions(obs, BufferAwareSafety(min_buffer_s=4.)), baseline)


def test_experimental_override_preserves_baseline_and_never_removes_actions():
    original = ActionSafety()
    changed = experimental_safety(original, 4., buffer_aware=True)
    assert isinstance(changed, BufferAwareSafety)
    assert original == ActionSafety() and changed.throughput_factor == .65
    assert asdict(changed)['min_buffer_s'] == 4
    with pytest.raises(ValueError): experimental_safety(None, buffer_aware=True)
    obs = observation()
    for rate in (.001, .5, 2., 10.):
        obs['history'][-3:, 0] = rate
        assert np.all(~allowed_actions(obs, ActionSafety(min_buffer_s=4.)) | allowed_actions(obs, changed))


@pytest.mark.parametrize('kwargs', [{'recovery_samples': True}, {'recovery_samples': 1},
    {'recovery_size_factor': .5}, {'recovery_reserve_s': 0}, {'recovery_max_download_s': float('nan')}])
def test_invalid_recovery_configuration(kwargs):
    with pytest.raises(ValueError): BufferAwareSafety(**kwargs)
