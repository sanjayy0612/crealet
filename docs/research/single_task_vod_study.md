# Single-task VOD decision-model study

## Purpose

This study isolates one question: **does a learned candidate scorer improve
VOD representation choices over the existing highest-allowed rule when both
use the same safety filter?** It does not test live streaming, production
readiness, or a general-purpose decision model.

The first study reuses the frozen startup-ladder checkpoints and recorded
transfer cases. It is a diagnostic replication, not a new generalization claim.
It should establish whether the current log-bitrate reward tells the same story
as measured visual quality before we spend effort on another training run.

## Fixed task and policy

- At each segment request, select one representation from the current finite
  VOD ladder using the observation v2 history, player state, and candidate
  metadata.
- Keep the existing player configuration, request recovery, and
  `ActionSafety(min_buffer_s=12, throughput_factor=0.65)` fixed.
- Compare the frozen Dueling Double DQN checkpoint with `highest_allowed`,
  which selects the highest representation in that exact allowed-action mask.
- Also report fixed-lowest as a floor/reference. Do not tune any policy on the
  transfer cases.
- Add BOLA-Basic with log-bitrate utility, a fixed four-second segment
  assumption, `gamma*p=5`, and nominal sizes from observation v2. Run it once
  without the safety mask and once with the same mask as the DQN/rule.
- Run Envivio and Sintel on the four transfer sessions already fixed in
  `runs/startup-ladder/transfer.json`. These are unseen session recordings from
  the same historical ferry route, and Sintel is held out from training. They
  are useful transfer cases, but they are not independent routes or modern
  network data.

## Outcomes

Report per case and paired policy differences for:

1. Completion, startup, rebuffer time, and stall events.
2. Playback-duration-weighted VMAF from the actual encoded representation
   segments, with source frame alignment, dimensions, frame rate, and metric
   implementation/version recorded.
3. Mean selected bitrate and the existing reward, clearly labeled as a bitrate
   proxy rather than visual quality.
4. Batch-one decision latency, measured separately from network transfer and
   with the same hardware/runtime controls for both policies.

The old 3 s startup and 8 s rebuffer limits may be shown as engineering gates
for continuity with prior reports. They are not approved user-quality limits.
Do not define a VMAF non-inferiority margin or claim acceptable quality until
the encodes have been visually reviewed at the intended player size.

## Hypotheses

- **H1:** DQN will not outperform `highest_allowed` on these cases because prior
  paired tests found identical actions for the frozen baseline.
- **H2:** The trained startup-ladder DQN seeds may trade visual quality for
  lower bitrate on transfer cases, consistent with their lower reward than the
  rule; VMAF will show whether this proxy tracks the visible quality change.
- **H3:** The current result is too narrow to establish broad generalization,
  even if one policy wins these cases.

## Freeze and reporting rules

- Keep checkpoint hashes, media fingerprints, datasets, trace cases, recovery,
  safety settings, and evaluator revision in the output manifest.
- Preserve failures and incomplete episodes. Do not rank only completed runs.
- Report each of the four network sessions separately. Do not treat segments
  within a session as independent samples.
- Do not change weights, safety settings, quality limits, or test cases after
  viewing these outcomes.
- Separate simulator results from browser results. Existing browser captures
  are not a substitute for this visual-quality measurement.

## Initial screen result

The frozen transfer screen ran with FFmpeg 9.0.2 and its `libvmaf` filter.
The evaluator scores only rungs selected by one or more frozen policies. It
normalizes both videos to 25 fps, upscales each distorted rung to that video's
source resolution before VMAF, checks exact frame counts, and computes the mean
frame score for each selected segment. Segment means are weighted by media
duration. The VMAF cache, full 48-row JSON result, and generated table are in
`runs/single-task-vod-v1/`; regenerate with:

```sh
.venv/bin/python -m scripts.evaluate_bola_transfer
.venv/bin/python scripts/evaluate_vmaf_transfer.py
```

| Content | Policy | Mean VMAF across four sessions | Mean bitrate kbps | Mean log-bitrate reward |
|---|---|---:|---:|---:|
| Envivio | BOLA-Basic score | 52.714 | 1341.3 | 632.619 |
| Envivio | BOLA-Basic plus safety mask | 43.337 | 673.5 | 514.648 |
| Envivio | Highest-allowed rule | 47.682 | 772.1 | 557.456 |
| Envivio | Old frozen DQN | 47.682 | 772.1 | 557.456 |
| Envivio | Seed 7 | 45.275 | 627.9 | 538.700 |
| Envivio | Seed 11 | 46.116 | 651.6 | 544.700 |
| Envivio | Seed 13 | 43.252 | 573.5 | 516.283 |
| Sintel | BOLA-Basic score | 31.887 | 124.1 | 55.926 |
| Sintel | BOLA-Basic plus safety mask | 28.689 | 101.8 | 50.118 |
| Sintel | Highest-allowed rule | 39.539 | 300.4 | 84.886 |
| Sintel | Old frozen DQN | 39.539 | 300.4 | 84.886 |
| Sintel | Seed 7 | 38.810 | 247.0 | 81.023 |
| Sintel | Seed 11 | 39.539 | 299.6 | 84.839 |
| Sintel | Seed 13 | 33.337 | 207.5 | 63.892 |

All 48 policy/session cases completed in the existing simulator records. The
old frozen DQN exactly matches the rule. The three newer learned checkpoints
never improve mean VMAF over the rule: seed 11 ties it on Sintel but is lower on
Envivio; the other two are lower on both. They also select lower mean bitrate.
The reward proxy follows that trend, so it does not appear to hide a learned
quality gain here.

This is a paired screen on only four recordings from one historical ferry
route. It does not support significance estimates or broad generalization.
VMAF is not human acceptance, and these are simulator action sequences, not
browser playback quality measurements. The very-low-resolution 8 kbps rung
scores near zero on Envivio (mean 0.822), which quantifies the startup rung's
large quality cost.

Both BOLA variants completed all eight video/session episodes with zero
rebuffering. Each met the 3 s startup limit on three of four sessions per
video. The unfiltered BOLA-Basic score beat the rule on Envivio VMAF by 5.032
points, but lost on Sintel by 7.652 points. It also missed the startup limit on
the same very-low-service session where every controller took about 10.3 s.
Adding the existing safety mask reduced mean VMAF below the rule on both videos
(-4.345 Envivio, -10.850 Sintel). BOLA is a useful conventional reference, not
a uniform winner.

This screen implements the BOLA-Basic score from [Spiteri et al.](https://arxiv.org/abs/1601.06748)
with a fixed four-second segment assumption and nominal candidate sizes. It
does not implement the paper's BOLA-FINITE startup/wind-down changes. The
simulator has no no-download action, so the adapter chooses the highest-scoring
available rung; at a full buffer, the simulator's buffer-cap backpressure
supplies the wait. Treat this as a BOLA-Basic adaptation, not a reproduced
dash.js controller. Batch-one inference latency was not measured in this
screen.

I also inspected one upscaled 8 kbps frame from each video at 640×360 display
size. Both are visibly blurred, with poor fine detail; the Sintel frame loses
clear character and scene detail. This supports using 8 kbps only as an
emergency survival rung, not as the normal quality floor. This was a small
frame sample, not a full human-quality review.

**Decision at this stage:** do not promote these weights. The rule already
matched the old model, the newer models gave up quality on this transfer screen,
and BOLA-Basic gave a mixed result. The later finite-video, independent-route,
controlled-capacity, and throughput-predictor studies are recorded below and in
the [predictor report](throughput_predictor_study.md). They do not show a
repeatable playback gain over highest-allowed. Keep human review of the
minimum-quality rung as a separate requirement.

## Independent-route BOLA-FINITE diagnostic

The follow-up uses the untouched `test` split of `data/realistic_envivio`,
which holds out Ghent bus 0002 and car 0002 from the Norway HSDPA training
collection. It compares the published BOLA-FINITE dynamic buffer target and
explicit pause with fixed BOLA-Basic, each with and without the frozen safety
mask, the highest-allowed rule, and the frozen seed-7 DQN. The model checkpoint
hash, protocol, dataset fingerprint, and per-route results are stored in the
generated `runs/bola-finite-ghent-v1/report.json`; regenerate with:

```sh
.venv/bin/python -m scripts.evaluate_bola_finite_ghent
```

| Policy | Completed | Mean reward | Mean bitrate kbps | Mean startup s | Mean rebuffer s |
|---|---:|---:|---:|---:|---:|
| BOLA-FINITE | 2/2 | 491.57 | 4135.1 | 0.193 | 0.000 |
| BOLA-FINITE plus safety mask | 2/2 | 377.26 | 3269.1 | 0.193 | 0.000 |
| BOLA-Basic | 2/2 | 400.06 | 3340.2 | 0.193 | 0.000 |
| BOLA-Basic plus safety mask | 2/2 | 400.06 | 3340.2 | 0.193 | 0.000 |
| Highest-allowed rule | 2/2 | 470.30 | 3970.1 | 0.193 | 0.000 |
| Frozen seed-7 DQN | 2/2 | 467.13 | 3941.5 | 0.193 | 0.000 |

All policies complete both sessions without stalls. On this tiny split, the
unfiltered finite controller exceeds the highest-allowed bitrate-proxy reward
by 21.27 (about 4.5%); the frozen DQN is 3.18 below the rule. Applying the
safety mask to BOLA-FINITE lowers its mean reward to 377.26, below both the
rule and BOLA-Basic. That means the observed finite-BOLA gain comes from a
different risk/quality choice under an unfiltered action set. Since both
sessions are high-service routes and no video payload is present in this trace
dataset, the result does not establish robust generalization or visual-quality
gain.

The DQN was trained on a seven-rung Envivio ladder including an 8 kbps startup
rung. The Ghent transfer ladder has six rungs (300–4300 kbps). Its
candidate-conditioned model accepts variable candidate counts and has no slot
embeddings, but this is still a changed action set; keep the DQN result as a
diagnostic transfer check. BOLA-FINITE follows the dynamic target and explicit
wait from [Spiteri et al.](https://arxiv.org/abs/1601.06748). The simulator
adapter does not implement request abandonment or dash.js integration.

**Decision at this stage:** the evidence did not justify another DQN
architecture or weight promotion. The follow-up controller and predictor
studies are now documented in the [controlled-capacity
report](controlled_capacity_transfer.md) and [throughput-predictor
report](throughput_predictor_study.md). They also did not establish a repeatable
gain over highest-allowed. Keep finite BOLA as a reference. Do not treat
architecture changes as generalization improvements without new route-held-out
evidence.

## RobustMPC head-to-head

The next controller comparison uses a five-chunk exhaustive RobustMPC plan.
It predicts throughput with the harmonic mean of the last five completed
requests, then discounts it by the largest recent one-step relative prediction
error. It scores candidate sequences with this project's log-bitrate reward,
including rebuffer, switch, and startup penalties. The first action uses the
same frozen safety mask as the DQN and highest-allowed rule. This adapts the
published RobustMPC method to the project's objective; it is not a byte-for-byte
reproduction of the original paper's QoE-linear setup. The RobustMPC paper
describes the five-sample error bound and five-chunk horizon in its benchmark
description ([Yin et al.](https://xieyaxiongfly.github.io/CSE610_UB/_files/paper/mpc_abr.pdf),
[Pensieve's benchmark details](https://web.mit.edu/pensieve/content/pensieve-sigcomm17.pdf)).

Run the frozen comparison with:

```sh
.venv/bin/python -m scripts.evaluate_robust_mpc_ghent
```

| Policy | Completed | Mean reward | Mean bitrate kbps | Mean rebuffer s | Decision latency p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| RobustMPC | 2/2 | 470.30 | 3970.1 | 0.000 | 59.8 / 69.6 |
| Frozen seed-7 DQN | 2/2 | 467.13 | 3941.5 | 0.000 | 0.33 / 0.44 |
| Highest-allowed rule | 2/2 | 470.30 | 3970.1 | 0.000 | 0.01 / 0.01 |

RobustMPC selected exactly the same action as the rule on all 49 chunks of
both traces. The DQN matched 48 of 49 decisions on each route. This tiny,
high-service screen shows no QoE gain from the learned model over the rule. The
enumerative Python implementation took about 60 ms at median per decision on
this machine, versus 0.33 ms for the DQN; treat those as local prototype
measurements, not hardware-independent performance claims.

This is still a diagnostic, not a blind test: the two Ghent test routes were
already used in the BOLA screen, both are high-service, and the trained DQN's
seven-rung ladder differs from the six-rung transfer ladder. The output at
`runs/robust-mpc-ghent-v1/report.json` records per-route decisions, predictor
diagnostics, hashes, and latency. Keep the algorithm settings fixed; the next
useful comparison needs harder independent routes where policies actually make
different choices, plus media payloads for visual-quality scoring.

## Decision rules for future studies

- If DQN and the rule select the same actions, there is no learned-policy
  difference to promote; move to a stronger ABR reference or a new task
  hypothesis, not a larger network.
- If actions differ but measured visual quality and playback do not improve
  within the fixed safety settings, keep the rule.
- Treat a win on two Ghent routes as a screening result. Before claiming
  generalization, freeze a broader set of independent routes and compare on the
  same action ladder with measured media quality.

This study can validate the existing controller comparison. It cannot validate
the original architecture inspiration by itself. A later architecture study
must compare dense/GRU/Transformer models under the same objective, data,
training budget, and fixed policies.
