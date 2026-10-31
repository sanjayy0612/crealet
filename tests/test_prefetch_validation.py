from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from abr.env import StreamingEnv, StreamConfig
from abr.player_server import experimental_replay, select_action
from abr.models.safety import ActionSafety
from abr.recovery import RecoveryConfig
from scripts.evaluate_prefetch import MeteredEnv
from tests.test_env import fixture, finish


def test_payload_meter_reports_compressed_backlog_without_changing_playback():
    data, _ = fixture(segments=2)
    reference = finish(StreamingEnv(data, recovery=RecoveryConfig()))[-1][-1]['metrics']
    env = MeteredEnv(data, recovery=RecoveryConfig())
    actual = finish(env)[-1][-1]['metrics']
    for key, value in reference.items():
        assert actual[key] == pytest.approx(value)
    assert actual['estimated_peak_encoded_whole_segments_bytes'] == 2000.
    assert actual['estimated_peak_encoded_proportional_bytes'] == pytest.approx(1500.)
    env.reset()
    assert env.metrics()['estimated_peak_encoded_whole_segments_bytes'] == 0


def test_explicit_cap_override_keeps_original_config_and_requires_exact_video():
    data, _ = fixture()
    config = StreamConfig()
    selected, changed = experimental_replay(data, config, data, 60.)
    assert selected is data and changed.max_buffer_s == 60. and config.max_buffer_s == 30.
    wrong, _ = fixture(sizes=[[1001., 2000.], [1000., 2000.]])
    with pytest.raises(ValueError, match='Replay video differs'):
        experimental_replay(data, config, wrong, 60.)
    for cap in (float('nan'), -1., 1.):
        with pytest.raises(ValueError):
            experimental_replay(data, config, data, cap)


def test_native_highest_allowed_uses_same_checkpoint_filter():
    data, _ = fixture()
    env = StreamingEnv(data)
    obs, _ = env.reset()
    obs['state'][0] = 20.
    obs['history'][-1] = [2., .1, 0., 0., 1.]
    policy = SimpleNamespace(safety=ActionSafety())
    assert select_action('highest_allowed', obs, env.config, policy) == 1
    obs['state'][0] = 0
    assert select_action('highest_allowed', obs, env.config, policy) == 0
    with pytest.raises(ValueError, match='checkpoint action filter'):
        select_action('highest_allowed', obs, env.config)


def test_changed_media_requires_explicit_dataset_and_opt_in():
    data, _ = fixture()
    changed, _ = fixture(sizes=[[1001., 2000.], [1000., 2000.]])
    with pytest.raises(ValueError, match='explicit replay dataset'):
        experimental_replay(data, StreamConfig(), None, 60., allow_media_change=True)
    selected, config = experimental_replay(data, StreamConfig(), changed, 60., allow_media_change=True)
    assert selected is changed and config.max_buffer_s == 60.
