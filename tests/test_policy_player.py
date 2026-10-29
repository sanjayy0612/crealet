"""Player-side baseline decisions must match the simulator's diagnostic rule."""
import pytest

pytest.importorskip('torch')
from abr.benchmark import evaluate
from abr.env import StreamingEnv
from abr.player_server import select_action
from .test_env import fixture


@pytest.mark.parametrize('name', ['lowest', 'highest', 'throughput'])
def test_player_baselines_match_simulator(name):
    dataset, media = fixture()
    reference = evaluate(StreamingEnv(dataset, media=media), baseline=name)
    actual = evaluate(StreamingEnv(dataset, media=media), baseline='model',
                      policy=lambda obs, config: select_action(name, obs, config),
                      policy_metadata={'kind': 'browser-baseline-equivalence'})
    for expected, observed in zip(reference['episodes'], actual['episodes']):
        assert [e['action'] for e in expected['steps']] == [e['action'] for e in observed['steps']]
        assert expected['metrics'] == observed['metrics']


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match='Unknown policy'):
        select_action('typo', {}, None)
