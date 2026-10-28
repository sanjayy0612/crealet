# Training the ABR CandidateTransformer

## Recommendation

Start with **online actor-critic training using PPO** in `StreamingEnv`. The environment can generate fresh trajectories by sampling training traces, and the model already exposes the two outputs PPO needs: candidate scores for the policy and a value estimate for the critic. PPO alternates environment interaction with several small-batch optimization passes, which makes it a practical first algorithm to implement. The original Pensieve ABR work also learns bitrate decisions through interaction with a trace-driven streaming environment.

This is a recommendation from the repository design and prior work, not a claim that PPO is proven best for this environment. Compare it against the existing ABR baselines and a simple supervised/oracle baseline before making quality claims.

## Why this fits this repository

- Each decision affects later buffer, playback, history, and live latency. That makes this a sequential decision problem, rather than independent classification.
- `StreamingEnv.step()` returns a reward for the QoE terms, next observation, and terminal/truncated status. It can create fresh episodes from training traces.
- `CandidateTransformer` returns masked candidate logits and a state value. Use logits as a categorical distribution over available bitrates; train the value head to estimate future discounted reward.
- The network, reward weights, and simulator are explicitly approximations. In particular, the reward uses a log-bitrate quality proxy, and transport uses finite goodput replay rather than packet/TCP modeling. Training will optimize those choices, not real-world QoE directly.

## Small first training loop

1. Reset episodes on randomly selected **training** traces and, when useful, random trace offsets/configurations.
2. At each observation, sample a valid candidate from `softmax(logits)`, record its log probability, value, reward, and end flags.
3. Collect a rollout batch, then calculate returns and generalized advantage estimates (GAE). Decide how to discount first: `gamma=1` matches maximizing the finite episode's total reward; elapsed-time discounting is a different objective because decisions take different amounts of simulated time.
4. Update policy and value heads with PPO's clipped policy objective, value loss, and a small entropy bonus. Mask padded candidates before sampling and computing log probabilities.
5. Repeat. Periodically evaluate deterministically on the **validation** split; choose checkpoints by validation QoE and report QoE components plus completion/truncation rates.
6. Keep the **test** split untouched until final comparison. Compare against the built-in baselines, with multiple random seeds and confidence intervals if making research claims.

For this finite stream, use the final return when playback completes. The environment uses `truncated=True` for trace exhaustion or request timeout, which ends this task with incomplete playback. Treat that as an episode end, do not bootstrap as if more trace data were available, and report the failure rate. The current environment reward has no separate completion penalty, so reward and completion rate must be read together. Gymnasium's usual bootstrap rule is for an external cutoff where the task could have continued, so it does not automatically apply to these environment failures.

## Other paradigms

- **Supervised learning / behavior cloning:** learn to copy an existing ABR policy. It is simple and useful as a warm start, but it can only reproduce the teacher's behavior and does not directly improve the repository's reward.
- **Offline RL:** learn from a fixed log of actions and outcomes. Consider it only if replaying the environment is too costly or if you have real interaction logs. It is harder because the learned policy may choose actions poorly covered by the log, which can make value estimates unreliable.
- **DQN:** possible because bitrate choices are discrete, but the model is currently an actor-critic with logits and a value head, not an action-value head. PPO fits the current outputs more directly.
- **Model predictive control (MPC):** a valuable non-neural baseline if future segment sizes and network estimates can be made available. It solves a short-horizon planning problem and helps tell whether the neural policy adds value.

## Sources

- Mao, Netravali, and Alizadeh, [Neural Adaptive Video Streaming with Pensieve](https://web.mit.edu/pensieve/) (SIGCOMM 2017 project and paper links): trace-driven ABR policy learned through reinforcement learning from observed outcomes.
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017): PPO alternates environment sampling and optimization of a clipped surrogate objective, with multiple minibatch epochs per sample batch.
- Fujimoto, Meger, and Precup, [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900) (2019): offline RL faces extrapolation error from actions outside the logged dataset's support; this is why offline training is not the default when a simulator can generate experience.

## Limits and next experiment

The new architecture is currently untrained, and the repository does not yet contain a training loop. First implement PPO with clear episode/truncation handling, then compare it to the current baselines on validation traces. Vary one factor at a time (for example, reward weights or rollout length). Do not tune on test traces. The simulator's synthetic fixtures are engineering checks, not evidence of generalization to measured networks.
