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
- Wang et al., [Dueling Network Architectures for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/wangf16.html) (ICML 2016): separates state value and per-action advantage estimates.
- van Hasselt, Guez, and Silver, [Deep Reinforcement Learning with Double Q-Learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295) (AAAI 2016): uses one network to select the next action and another to score it, reducing Q-value overestimation.
- Fujimoto, Meger, and Precup, [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900) (2019): offline RL faces extrapolation error from actions outside the logged dataset's support; this is why offline training is not the default when a simulator can generate experience.

## Recorded training results

The PPO loop is implemented in [`abr/models/train_ppo.py`](../../abr/models/train_ppo.py). It samples candidate actions from the policy, uses `gamma=1` for the finite episode reward, normalizes the critic loss by return scale, and selects checkpoints by validation completion rate and then reward. Truncated episodes from trace exhaustion or request timeout end without bootstrapping. The checkpoint records its training data hash and hyperparameters.

The available run used the built-in **synthetic VOD fixture**, not measured network data. The final sweep used five policy seeds (7–11), one fixed dataset seed (7), 50 PPO updates per run, and 16 episodes per update. It saw about 190,380 environment decisions across the five runs. Test traces were not used to select these revised checkpoints.

The validation gate was: mean reward must match or beat the throughput baseline, and completion rate must be no lower. All five revised runs passed. The throughput baseline scored 124.39 with 100% completion on these eight validation traces; the model median reward was 172.17. Individual results:

| Policy seed | Best update | Validation reward | Completed | Rebuffer time | Mean bitrate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 49 | 175.50 | 100% | 4.21 s | 1,598 kbps |
| 8 | 25 | 172.17 | 100% | 4.22 s | 1,547 kbps |
| 9 | 43 | 166.56 | 100% | 7.65 s | 1,617 kbps |
| 10 | 44 | 164.60 | 100% | 7.97 s | 1,645 kbps |
| 11 | 34 | 174.09 | 100% | 4.97 s | 1,591 kbps |

For context, the throughput baseline's validation rebuffer time was 2.90 s and mean bitrate was 1,228 kbps. PPO earned more reward by choosing higher bitrates, with more stalls. The reward encodes a bitrate proxy and configured penalties; these numbers do not establish better viewer-perceived quality.

The earlier small-rollout setup (four episodes per update, without critic-loss normalization) passed the same reward/completion gate in only two of five seeds. The revised rollout and loss scaling made this synthetic result repeatable across the five seeds.

The selected validation checkpoint is [`runs/ppo-revised-seed-7/candidate_transformer_ppo.npz`](../../runs/ppo-revised-seed-7/candidate_transformer_ppo.npz). Per-seed validation numbers and checkpoint paths are in [`runs/ppo-revised-seed-sweep.json`](../../runs/ppo-revised-seed-sweep.json). An earlier test-split report exists at [`runs/ppo-synthetic-50/test_metrics.json`](../../runs/ppo-synthetic-50/test_metrics.json); it belongs to the earlier seed-7 checkpoint, not the revised selected model, and must not be presented as final evidence for the revised model.

**Conclusion:** PPO cleared the stated first gate on the synthetic validation set. This showed that the training loop could find a repeatable policy for that fixture; it did not establish generalization. Later DQN, baseline-controller, and supervised-predictor studies tested the project on controlled profiles and historical measured traces. They did not establish a repeatable playback advantage over the simple rule. See the [current throughput-predictor study](throughput_predictor_study.md) and [controlled-capacity study](controlled_capacity_transfer.md). Further model training is paused pending new measured traces that can expose a playback trade-off.

## Dueling Double DQN comparison

The separate [`DuelingDoubleDQN`](../../abr/models/dueling_dqn.py) keeps the numeric Transformer encoder but changes its output to Q-values. A state-value head estimates `V(s)`. A candidate head estimates `A(s,a)`, the extra value of selecting each bitrate. The model combines them as `Q(s,a) = V(s) + A(s,a) - mean(valid A)`. The [trainer](../../abr/models/train_dueling_dqn.py) uses a replay buffer and target network. For the Double DQN target, the online network selects the next action and the target network evaluates it. Deployment picks the valid bitrate with the largest Q-value.

For a fair first comparison, DQN and PPO each used five policy seeds and the same synthetic data seed, reward weights, and training/validation traces. Each run used about 38,000 environment decisions. DQN trained for 800 episodes; PPO trained for 50 updates of 16 episodes. The same validation gate was used: reward at least equal to the throughput baseline and completion no lower. Both methods passed in all five seeds.

| Method | Median validation reward | Median rebuffer time | Completion gate |
| --- | ---: | ---: | --- |
| Throughput baseline | 124.39 | 2.90 s | 100% |
| PPO | 172.17 | 4.97 s | Passed 5/5 seeds |
| Dueling Double DQN | 170.10 | 5.48 s | Passed 5/5 seeds |

DQN was close to PPO on this simulator, but its median reward was slightly lower and median rebuffer time slightly higher. Both learned policies traded some stall time for higher bitrate under the current reward. The weights still encode only a simulator proxy, not measured viewer preference.

The DQN seed-by-seed metrics and checkpoint paths are in [`runs/dueling-dqn-seed-sweep.json`](../../runs/dueling-dqn-seed-sweep.json). The seed-11 checkpoint had the highest DQN validation reward and is available at [`runs/dueling-dqn-equal-seed-11/dueling_double_dqn.npz`](../../runs/dueling-dqn-equal-seed-11/dueling_double_dqn.npz). The test split was not used for this comparison or checkpoint selection. Results remain synthetic-only and need replication on audited traces.
