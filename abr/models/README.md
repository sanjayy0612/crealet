# Decision model structure

Model construction starts with random, untrained weights. The optional PPO
trainer below can write trained checkpoints. PyTorch is optional for the
simulator; install `requirements-model.txt` to use these modules.

```text
abr/models/
  candidate_transformer.py   Transformer configuration, projections, encoder, policy/value heads
  dueling_dqn.py            Candidate Q-values with dueling value/advantage streams
  gru/
    history.py               Legacy masked GRU history encoder; no persistent state
    __init__.py              GRU encoder export
  network.py                 Existing dense/GRU configuration and shared candidate scorer
  features.py                Fixed numerical scaling used by all architectures
  inputs.py                  Observation/config validation, batching and padding masks
  factory.py                 Model-kind registration and construction
  policy.py                  Deployment-style argmax selection (ignores value head)
  checkpoint.py              Versioned, source-checked NPZ serialization
  __main__.py                init / inspect / diagnostic rollout CLI
  train_dueling_dqn.py       Double-DQN replay training loop
```

## ABR-CandidateTransformer v1

```text
history [B,8,5] -- fixed log scaling -- Linear(5,64) + history positions --+
state/config [B,15] -- fixed scaling -- Linear(15,64) -- state token -----+--> concatenate
candidates [B,K,3] -- fixed scaling -- Linear(3,64) ---------------------+     [B,9+K,64]
                                                                                |
                                                    two pre-LayerNorm encoder layers
                                                  4 heads; FFN 128; GELU; dropout 0
                                                                                |
                         +------------------------------------------------------+
                         |                                                      |
                candidate token vectors                                state token vector
                         |                                                      |
       LayerNorm -> Linear(64,64) -> GELU -> Linear(64,1)             same shape, separate weights
                         |                                                      |
                  logits [B,K]                                            value [B]
```

The default configuration has 77,826 trainable parameters. Each encoder layer
is independently initialized. There are no language tokens, persistent recurrent
states, candidate positional embeddings, or extra type embeddings. Learned
positions belong only to history slots, oldest to newest. Changing the configured
history length changes the position table and parameter count.

The history validity column becomes a boolean `history_mask`; `candidate_mask`
identifies real candidates when batching different ladders. Invalid rows are
sanitized before projection and excluded as attention keys/values in every layer.
The state token is always valid, including at reset when history is entirely
padded. Candidate logits are masked to negative infinity before argmax or softmax.
Every sample must have at least one valid candidate. The low-level tensor forward
expects the validated nonnegative feature contract from `make_batch`.

Without candidate positions, a permutation of candidates permutes logits and
leaves the value output unchanged, within floating-point tolerance. The state
token attends to candidates as well as history, so its value readout can depend
on the available action set.

```python
from abr.models import CandidateTransformer, make_batch
from abr.models.policy import DecisionPolicy

model = CandidateTransformer()
batch = make_batch([observation], [stream_config], model.config)
logits, value = model(batch)
action = DecisionPolicy(model)(observation, stream_config)
```

`observation` and `stream_config` come from the existing V2 environment. The model
receives no trace or simulator object. Fixed feature scaling matches the existing
dense/GRU contract; it is not fitted on training, validation, or test data.

## Artifacts and diagnostic rollouts

```bash
python -m abr.models init --kind candidate_transformer --seed 0 --output runs/transformer-initial.npz
python -m abr.models inspect --checkpoint runs/transformer-initial.npz
python -m abr.models rollout --checkpoint runs/transformer-initial.npz --dataset data/audited.json --mode vod --allow-untrained --output runs/transformer-integration.json
```

Live rollouts additionally need packaged part metadata via `--media`, as with the
existing models. Rollouts are restricted to train/validation splits. Loading checks
model configuration, tensor keys/shapes/dtypes, finite weights, fixed scaling,
parameter count, and a source hash covering all nested model Python modules.
Older artifacts require regeneration after source changes. Checkpoints record
whether they are `untrained` or `trained` and include training metadata.

The value head supports actor-critic training; deployment policy selection uses
only logits. The value output is a learned estimate of future reward. Candidate
scores are policy scores, not calibrated correctness or safety probabilities.

## Dueling Double DQN variant

`DuelingDoubleDQN` uses the same numeric Transformer encoder. One head estimates
the value of the current state; another estimates each bitrate's advantage over
the other valid bitrates. Their combined output is a Q-value per candidate. The
trainer uses replay memory and a target network; Double DQN uses the online model
to choose the next action and the target model to score it.

```bash
python -m abr.models.train_dueling_dqn \
  --dataset data/realistic_envivio/dataset.json \
  --episodes 400 --seed 7 \
  --output runs/dqn-envivio
```

For each validation trace, checkpoint selection uses that trace's throughput
baseline stall time as a hard ceiling. By default, the DQN must complete every
validation episode and stay at or below the baseline stall time on every trace.
`--stall-tolerance-s` can add a measured tolerance. The trainer keeps an
unconstrained candidate for diagnosis, but marks it unaccepted if no checkpoint
passes the limit. It evaluates the accepted checkpoint on the held-out test
traces after selection. Test results do not choose a checkpoint. Reports show
the bitrate selected for each segment.

The harder follow-up run also sets `--max-rebuffer-s 8 --max-startup-s 3`.
The stall ceiling on each validation trace is the smaller of the absolute limit
and that trace's throughput-baseline stalls. `--rebuffer-penalty 12` charges
12 reward units per stall second, identically for DQN and all baselines. These
are declared experimental limits, not universal quality-of-experience standards.
See `runs/dqn-hard/training_report.md` for the outcome and
`experiments/validation/POLICY_PLAYER.md` for the native browser connection.

The example input is one DASH video ladder and eight historical Ghent 4G traces.
It is an initial realistic replay check, not evidence for other videos, current
networks, or real-player performance. See
`data/realistic_envivio/README.md` for sources and limitations.

## PPO training

The first training loop uses complete episodes and the finite episode reward
(`gamma=1`). With no dataset argument it trains on the built-in synthetic
engineering fixture; this is useful for plumbing and early experiments, not for
claims about real networks.

```bash
python -m abr.models.train_ppo --updates 50 --episodes-per-update 16 --seed 7 --output runs/ppo
python -m abr.models.train_ppo --dataset data/audited.json --updates 50 --seed 7 --output runs/ppo-measured
```

The trainer checks validation traces after each update, keeps the best checkpoint
by completion rate and then reward, and writes `training_metrics.json`. The test
split is not used for training or selection. Timeout and trace-exhaustion episodes
end without critic bootstrapping; their reward and completion failure remain in
the report.
