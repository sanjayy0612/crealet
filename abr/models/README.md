# Decision model structure

Model construction starts with random, untrained weights. The optional PPO
trainer below can write trained checkpoints. PyTorch is optional for the
simulator; install `requirements-model.txt` to use these modules.

```text
abr/models/
  candidate_transformer.py   Transformer configuration, projections, encoder, policy/value heads
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
