"""A high reward cannot override incomplete playback, stalls or startup limits."""
from abr.models.train_dueling_dqn import checkpoint_eligible


def report(completion=1., stall=True, startup=1.):
    return {'completion_rate': completion, 'stall_constraint_met': stall,
            'mean_reward': 1e6, 'episodes': [{'startup_s': startup}]}


def test_high_reward_cannot_override_playback_limits():
    assert checkpoint_eligible(report(), 3.)
    assert not checkpoint_eligible(report(completion=.5), 3.)
    assert not checkpoint_eligible(report(stall=False), 3.)
    assert not checkpoint_eligible(report(startup=3.001), 3.)
    assert checkpoint_eligible(report(startup=3.), 3.)
