from dataclasses import asdict

import pytest

pytest.importorskip('torch')
from abr.env import StreamConfig
from abr.models.checkpoint import save_trained_model
from abr.models.dueling_dqn import DuelingDoubleDQN
from abr.player_server import PlayerApp
from abr.recovery import RecoveryConfig
from tests.test_env import fixture


def test_player_refuses_to_drop_recovery_required_by_trained_checkpoint(tmp_path):
    data, _ = fixture()
    checkpoint = tmp_path / 'fixture.npz'
    # Serialization fixture only, not evidence of trained policy performance.
    save_trained_model(DuelingDoubleDQN(), checkpoint, training_metadata={
        'config': asdict(StreamConfig(max_buffer_s=60)), 'recovery': asdict(RecoveryConfig())})
    with pytest.raises(ValueError, match='recorded request-recovery'):
        PlayerApp(data, checkpoint, tmp_path / 'missing-media')
    with pytest.raises(ValueError, match='recorded request-recovery'):
        PlayerApp(data, checkpoint, tmp_path / 'missing-media', recovery=RecoveryConfig(max_retries=1))
