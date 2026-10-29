import numpy as np
import pytest
import torch

from abr.env import StreamingEnv
from abr.learning_features import QueueObservation, queue_potential, credit_reward
from abr.learning_experiment import initialize_model
from abr.models.inputs import make_batch
from abr.models.checkpoint import load_model, save_trained_model
from .test_env import fixture


@pytest.mark.parametrize('intervals', [(100.,), (1.5,)])
def test_quality_credit_preserves_success_and_failure_returns(intervals):
    data, _ = fixture(segments=3, intervals=intervals)
    env = StreamingEnv(data)
    env.reset(seed=1)
    raw = shaped = 0.
    while True:
        before = queue_potential(env)
        _, reward, ended, failed, _ = env.step(1)
        raw += reward
        shaped += credit_reward(reward, before, queue_potential(env), terminal=ended or failed)
        if ended or failed:
            break
    assert shaped == pytest.approx(raw, abs=1e-10)
    if len(intervals) == 1 and intervals[0] < 2:
        assert failed and queue_potential(env) > 0  # Unplayed credit must be reversed.


def test_queue_extension_matches_initial_scores_and_can_learn(tmp_path):
    data, _ = fixture(segments=4)
    plain = StreamingEnv(data)
    rich = QueueObservation(StreamingEnv(data))
    plain.reset(seed=1); rich.reset(seed=1)
    obs, *_ = plain.step(1)
    extra, *_ = rich.step(1)
    assert rich.observation_space.contains(extra)
    assert extra['queue'][0] > 0 and extra['queue'][1] == pytest.approx(1.)
    base, queue = initialize_model(7), initialize_model(7, True)
    a = make_batch([obs], [plain.config], base.config)
    b = make_batch([extra], [rich.config], queue.config)
    assert b['player'].shape == (1, 19)
    torch.testing.assert_close(base(a), queue(b), rtol=1e-5, atol=1e-6)
    queue(b).sum().backward()
    assert queue.state_projection.weight.grad[:, 15:].abs().sum() > 0
    path = tmp_path / 'queue.npz'
    save_trained_model(queue, path, training_metadata={'observation_extension': 'queue_summary_v1'})
    restored, _ = load_model(path)
    torch.testing.assert_close(queue(b), restored(b))
    with pytest.raises(ValueError):
        make_batch([obs], [plain.config], queue.config)
