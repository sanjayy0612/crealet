"""Observable queue summaries and return-preserving training reward credit."""
import math

import gymnasium as gym
import numpy as np


def queue_potential(env):
    player = env.unwrapped.player
    return sum(duration * math.log(bitrate / player.min_bitrate) for duration, bitrate in player.queue)


def queue_summary(env):
    player = env.unwrapped.player
    front = player.queue[0] if player.queue else (0., 0.)
    return np.array([queue_potential(env), front[1] / 1000,
                     (player.previous_bitrate or 0.) / 1000, front[0]], dtype=np.float32)


def credit_reward(reward, before, after, *, terminal, gamma=1.):
    # All exits, including trace exhaustion and request timeout, clear escrow.
    return float(reward + gamma * (0. if terminal else after) - before)


class QueueObservation(gym.Wrapper):
    """Opt-in summary, not a claim that the ordered playback queue is Markov."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({**env.observation_space.spaces,
            'queue': gym.spaces.Box(0., np.inf, (4,), dtype=np.float32)})

    @property
    def dataset(self):
        return self.unwrapped.dataset

    @property
    def split(self):
        return self.unwrapped.split

    @property
    def config(self):
        return self.unwrapped.config

    def metrics(self):
        return self.unwrapped.metrics()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return {**observation, 'queue': queue_summary(self)}, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return {**observation, 'queue': queue_summary(self)}, reward, terminated, truncated, info
