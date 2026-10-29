import numpy as np
import pytest
import torch

from abr.models.safety import ActionSafety, allowed_actions, constrained_argmax


def observation(buffer=20., rate=2.):
    return {'state': np.array([buffer, 0, 1, 1, 0]),
            'history': np.array([[rate, .2, 0, .008, 1.]]),
            'candidates': np.array([[.3, .15, 4], [.75, .375, 4], [1.2, .6, 4], [1.85, .925, 4]])}


def test_empty_history_low_buffer_and_outage_keep_lowest_available():
    safety = ActionSafety()
    for obs in [observation(buffer=11.9), observation(rate=0),
                {**observation(), 'history': np.zeros((8, 5))}]:
        assert allowed_actions(obs, safety).tolist() == [True, False, False, False]


def test_filter_opens_higher_quality_and_is_candidate_order_independent():
    obs, safety = observation(), ActionSafety()
    assert allowed_actions(obs, safety).tolist() == [True, True, True, False]
    reverse = {**obs, 'candidates': obs['candidates'][::-1].copy()}
    np.testing.assert_array_equal(allowed_actions(reverse, safety), allowed_actions(obs, safety)[::-1])
    assert allowed_actions(observation(rate=.8), safety).tolist() == [True, False, False, False]


def test_bootstrap_argmax_ignores_unsafe_action_and_batch_padding():
    q = torch.tensor([[1., 2., 3., 100., 1000.], [1., 100., 200., 300., 1000.]])
    actions = constrained_argmax(q, [observation(), observation(buffer=0)], ActionSafety())
    assert actions.tolist() == [2, 0]
    assert constrained_argmax(q[:1], [observation()]).tolist() == [3]


@pytest.mark.parametrize('kwargs', [{'min_buffer_s': -1}, {'throughput_factor': 0},
                                   {'throughput_factor': 1.1}, {'throughput_factor': float('nan')}])
def test_invalid_filter_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ActionSafety(**kwargs)
