# Bounded learning ablation

## Decision

Do not promote a new neural variant. Neither earlier reward credit nor the
four queue-summary inputs showed a useful advantage over the highest-allowed
rule in this short screen. Two additional measured sessions exposed request
timeouts for every candidate and the rule. These failures are retained.

## What was executed

Three arms, seeds 7 and 11, 120 episodes each: six runs, 720 episodes total.
Each run used 5,880 environment steps and 1,346 gradient updates, including five
target-network refreshes. Transition budgets, the architecture's common weights,
data, optimizer, objective, filter and validation limits were matched.

- Baseline: current filtered Dueling Double DQN.
- Credit only: `r' = r + Phi(next) - Phi(current)`, where Phi is the quality
  value of the remaining playback queue. Phi is forced to zero on both successful
  and failed exits. Evaluation still uses original QoE. The largest training
  episode return difference was 8.53e-14, consistent with floating-point error.
- Queue only: four additional player-observable inputs: buffered log-quality
  integral, next-played bitrate, previous-played bitrate, and remaining duration
  of the front playable unit. They add 256 parameters (77,826 to 78,082).
  Shared weights match baseline initialization; extra input weights start at
  zero. This is a queue summary, not the complete ordered queue or a Markov-state
  claim. Reward is unchanged in this arm.

All arms retain the 12 second reserve and 0.65 observed-goodput filter, reward
stall penalty 12, startup ceiling 3 seconds, and validation stalls at or below
the smaller of 8 seconds and the original throughput baseline on each trace.

## Validation selection

| Arm | Seed 7 reward | Seed 11 reward | Median bitrate | Mean stalls |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 96.868 | 89.147 | 687.9 kbps | 2.164 s |
| Credit only | 89.616 | 89.616 | 660.3 kbps | 2.164 s |
| Queue only | 96.868 | 89.434 | 690.1 kbps | 2.164 s |
| Highest-allowed rule | 96.868 | 96.868 | 717.0 kbps | 2.164 s |

Every selected checkpoint passed all validation playback limits. Queue inputs
increased median bitrate by only 2.2 kbps (about 0.3%) relative to baseline and
did not beat the simpler rule. Credit shaping reduced median bitrate. These
results do not establish a useful learning improvement.

The queue arm was frozen for the additional-session check by the predeclared
median-original-validation-return criterion. Exact artifact hashes and all six
frozen checkpoints were recorded in `frozen_selection.json` before fresh scoring.
All frozen arms were then measured without retraining or reselection.

`metrics.json` for each run contains interval TD loss, absolute TD error,
Q/target magnitudes, terminal/nonterminal errors, target-sync count, selected
and final validation metrics, and predicted-Q versus realized-continuation-return
diagnostics. Lower prediction error alone is not treated as policy improvement.

## Additional measured sessions

Two original Norway sessions were predeclared before selection:
`2010-09-21_1622CEST` and `2010-09-21_1735CEST`. They are additional collections
on the same ferry route as the historical tests, not a new route or video.
Raw files, URLs, checksums and timestamp conversion are in
`../../data/learning-holdout/`.

| Controller | Session 1622 | Session 1735 |
| --- | --- | --- |
| All six frozen neural candidates | Request timeout | Request timeout |
| Highest-allowed rule | Request timeout | Request timeout |
| Original throughput rule | Request timeout | Request timeout |
| Fixed lowest | Completed, 21.5 s stalls | Request timeout |

The filtered rule and neural candidates stopped after playing about 76.3% and
66.0% of the video respectively. Their partial-playback mean stalls/bitrates
must not be treated as complete-video results. The 30 second request timeout
and existing acceptance limits were retained. No failed session was discarded
and no limit was relaxed after seeing results.

This is evidence of a reliability gap under the current environment and request
protocol. It does not prove that every possible controller must fail. The next
useful reliability experiment is outage detection and request recovery, with
full completion measured separately from quality and stalls. A larger neural
network is not justified by this screen.

## Compatibility and verification

The queue-input artifact has an explicit opt-in model configuration and input
contract. Model checkpoint loading and the offline rollout adapter support it;
the native player rejects it explicitly. No experimental queue model was
deployed because it failed the promotion checks.

The existing operational filtered policy remains the reference. The newly
reproduced baseline seed-7 checkpoint has **exactly the same tensors** as the
previous accepted `runs/dqn-safe/` checkpoint; see `control_identity.json`.
The player's default path points to this current-source baseline artifact so
the source-hash check remains valid. Historical artifacts retain their original
bytes and require their matching historical source version to reload.

The full suite passed **107 tests**, with two existing Gymnasium infinite-bound
warnings. Checks cover return preservation on completion and failure with an
unplayed queue, matched initial scores, learnable queue input weights, checkpoint
round-trip and missing-input rejection. A larger model and new native playback
runs were deferred because no candidate beat the required rule and all failed
the additional-session completion test.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_ablation_holdout.py
PYTHONPATH=. .venv/bin/python scripts/run_learning_ablation.py
PYTHONPATH=. .venv/bin/python scripts/evaluate_learning_ablation.py
PYTHONPATH=. .venv/bin/python -m pytest -q
```

These short two-seed results do not rule out benefits from reward shaping or
queue information on broader data or longer, better-calibrated training. They
do not provide evidence to promote either change now.
