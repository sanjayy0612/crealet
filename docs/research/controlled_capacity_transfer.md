# Controlled-capacity transfer study

## Decision

Controlled-scenario training changed the learned policies, but it did not give a
consistent gain across seeds and content. Seed 7 gained VMAF on both videos;
seed 11 gained on Envivio and was nearly flat on Sintel; seed 13 lost heavily on
both. Keep the new checkpoints as research artifacts. Do not promote the DQN or
increase training time again yet.

The useful finding is that training coverage can raise selected bitrate and
quality, but the result is seed-sensitive. The next model change should address
that instability and align selection with visual quality under a stated stall
budget. Compare the reward and checkpoint-selection rule on training and
validation only before another frozen test.

## Experiment sequence

### V1 pilot: transition variety, but insufficient stress

[`controlled_capacity_v1.json`](../../experiments/scenarios/controlled_capacity_v1.json)
defines step drops, repeated capacity changes, and short zero-capacity outages.
The frozen comparison in
[`controlled_capacity_transfer_v1.json`](../../experiments/configs/controlled_capacity_transfer_v1.json)
used 12 generated profiles on each video. Every policy completed all 24
content/profile combinations with no rebuffering. The DQN was more conservative
than highest-allowed and RobustMPC. V1 therefore served as a quality-transfer
pilot and did not test outage resilience.

### V2 stress study: train on severe changes, test on fresh draws

The v1 result motivated longer outages and degraded-recovery periods in
[`controlled_capacity_stress_v2.json`](../../experiments/scenarios/controlled_capacity_stress_v2.json).
This is an engineering stress test. Its severity was chosen after the v1 pilot,
so it is not an untouched natural-network benchmark. The 36 Envivio training
profiles and 12 validation profiles are separate from the 12 fresh test
profiles. Checkpoint selection uses validation only. Sintel is not used for
training or checkpoint selection.

The training protocol is
[`controlled_capacity_training_v2.json`](../../experiments/configs/controlled_capacity_training_v2.json).
It keeps the DQN architecture, log-bitrate reward, optimizer, safety filter,
recovery behavior, and 120-episode budget fixed. A 1 ms tolerance is recorded
for per-trace stall-limit comparisons because one validation replay exceeded
the 8 second limit by 0.8 ms. Raw stall values are retained.

The final comparison protocol,
[`controlled_capacity_stress_transfer_v2.json`](../../experiments/configs/controlled_capacity_stress_transfer_v2.json),
was frozen after all checkpoints and dataset fingerprints were recorded. It
compares the original DQNs and scenario-trained DQNs against highest-allowed and
five-chunk RobustMPC on both Envivio and Sintel. The primary quality scores use
frame-aligned VMAF on completed segments.

## V2 results

All six DQNs completed all 12 profiles on both videos with zero rebuffering.
Highest-allowed and RobustMPC each had one Envivio stall event, with 6.10
seconds rebuffering on that case (0.51 seconds averaged over 12 cases). Both
baselines completed all sessions. No Sintel policy stalled.

Mean per-session VMAF (each session score is weighted by played media duration):

| Content | Policy | Mean VMAF | Mean bitrate (kbps) | Mean reward proxy |
| --- | --- | ---: | ---: | ---: |
| Envivio | Original DQN seed 7 | 58.17 | 946.6 | 676.1 |
| Envivio | Scenario-trained seed 7 | 63.00 | 1,243.0 | 720.1 |
| Envivio | Original DQN seed 11 | 61.37 | 1,118.7 | 703.9 |
| Envivio | Scenario-trained seed 11 | 64.71 | 1,717.3 | 754.9 |
| Envivio | Original DQN seed 13 | 58.10 | 922.6 | 675.8 |
| Envivio | Scenario-trained seed 13 | 43.76 | 436.9 | 575.8 |
| Envivio | Highest-allowed | 64.34 | 1,847.4 | 749.9 |
| Envivio | RobustMPC | 62.80 | 1,834.3 | 728.7 |
| Sintel | Original DQN seed 7 | 62.26 | 695.2 | 155.6 |
| Sintel | Scenario-trained seed 7 | 66.30 | 1,195.8 | 175.3 |
| Sintel | Original DQN seed 11 | 66.16 | 1,141.9 | 173.4 |
| Sintel | Scenario-trained seed 11 | 65.79 | 1,700.0 | 186.9 |
| Sintel | Original DQN seed 13 | 63.05 | 810.5 | 158.6 |
| Sintel | Scenario-trained seed 13 | 58.07 | 428.0 | 140.9 |
| Sintel | Highest-allowed | 65.80 | 1,866.7 | 189.4 |
| Sintel | RobustMPC | 65.80 | 1,866.7 | 189.4 |

The scenario-trained seed 7 improved mean VMAF by 4.83 on Envivio and 4.05 on
Sintel over its original checkpoint. Seed 11 improved Envivio by 3.34 and was
0.37 lower on Sintel. Seed 13 fell by 14.34 on Envivio and 4.98 on Sintel.
The sign of the training effect therefore depends on the seed. Across the
three seeds, training did not produce a reliable quality gain.

The action records help explain the seed 13 regression. Across all 12 test
profiles on each video, that checkpoint never selected above action index 2
(750 kbps). Seeds 7 and 11 selected up to index 5 on Envivio, and up to index 5
and 6 on Sintel. This points to a conservative policy learned in one seed, not
an inability of the ladder or safety mask to offer higher rates.

## Validation-only checkpoint-selection comparison

I saved all six eligible checkpoints per seed in a deterministic rerun and
replayed them on the 12 v2 Envivio validation profiles. I scored their
completed-segment decisions with frame-aligned Envivio VMAF. The existing
completion, startup, and per-trace stall gates stayed fixed. The v2 test split
was not replayed or used for this comparison.

| Seed | Reward-selected episode | VMAF-selected episode | Reward-selected validation VMAF | VMAF-selected validation VMAF | Highest-allowed validation VMAF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 20 | 20 | 67.623 | 67.623 | 69.819 |
| 11 | 20 | 20 | 69.429 | 69.429 | 69.819 |
| 13 | 60 | 80 | 48.116 | 50.308 | 69.819 |

VMAF-based checkpoint selection changed only seed 13. It gained 2.192 VMAF
points on validation while lowering mean reward from 623.46 to 620.14. The
selected checkpoint still trailed highest-allowed by 19.511 validation VMAF
points. This is too small to explain or fix the seed 13 collapse. Reward-based
selection is not the main issue for seeds 7 and 11; seed and optimization
instability remain stronger explanations.

Do not adopt VMAF checkpoint selection from this result alone. It was evaluated
on the same validation profiles used to choose a checkpoint, so its selected
score is optimistic. A new frozen test would be required before any quality
claim. The diagnostic supports investigating training stability and a
VMAF-aligned reward on training/validation data before another test run.

## Final reward experiment and stopping point

The next experiment replaced the Envivio training reward's log-bitrate utility
with a per-segment, per-rung VMAF utility. The measured VMAF values were mapped
linearly to the existing log-bitrate utility range. Startup, stall, switching,
recovery, safety, architecture, optimizer, training data, and 120-episode
budget stayed fixed. The utility matrix and its encoder/VMAF inputs are frozen
in [`envivio_vmaf_utility_v1.json`](../../experiments/rewards/envivio_vmaf_utility_v1.json).

Checkpoints were selected on the same v2 validation profiles. The evaluation
used an independent v3 parameter draw, frozen with the
[`vmaf_reward_stress_transfer_v1.json`](../../experiments/configs/vmaf_reward_stress_transfer_v1.json)
protocol. It compared matched log-reward and VMAF-reward seeds, highest-allowed,
and RobustMPC on both videos.

| Content | Log-reward DQN mean VMAF across seeds | VMAF-reward DQN mean VMAF across seeds | Highest-allowed | RobustMPC |
| --- | ---: | ---: | ---: | ---: |
| Envivio | 57.038 | 60.316 | 64.992 | 63.571 |
| Sintel | 61.982 | 63.924 | 65.034 | 65.034 |

The new reward helped seed 13 by 13.87 Envivio VMAF and 7.14 Sintel VMAF. It
lowered seed 11 by 5.61 and 0.92, respectively. Seed 7 gained 1.57 on Envivio
and lost 0.40 on Sintel. The change therefore reduced the severe seed 13
collapse, but it did not produce a consistent gain across seeds or outperform
the simple rule. Every policy completed all cases. Every policy had the same
two Envivio rebuffer events (1.506 seconds averaged over 12 profiles); no
Sintel case rebuffered.

## Stop decision

Stop training this Dueling Double DQN baseline. We compared longer training,
controlled-scenario training, VMAF-based checkpoint selection, and a
VMAF-aligned training reward. None produced a repeatable quality or stall
advantage over highest-allowed across seeds and contents. More training time is
not the current bottleneck.

Keep highest-allowed as the simple reference controller and RobustMPC as the
lookahead comparator. Highest-allowed had about 0.009 ms median decision time
here, compared with about 0.33 ms for DQN and 32–112 ms for RobustMPC. This
decision applies to this VOD prototype and these trace-driven cases. The
controlled profiles are synthetic and do not establish performance on natural
or current networks. Continue the project only with a new research hypothesis
that changes the available information or decision problem, not another
training-budget increase on this baseline.

The next hypothesis, a supervised throughput predictor feeding RobustMPC, is
recorded in the [throughput-predictor study](throughput_predictor_study.md).
It reduced forecast error on held-out synthetic and historical traces, but did
not create a repeatable playback gain over highest-allowed. That work is also
paused pending new measured traces with lower service and recorded RTT.

The new models also did not beat the baselines consistently. Highest-allowed
scored 64.34 on Envivio and 65.80 on Sintel, with one Envivio stall case.
Scenario-trained seed 11 slightly exceeded highest-allowed on Envivio without
that stall, while seed 7 slightly exceeded it on Sintel. Seed 13 regressed on
both videos. Treat these small and seed-specific margins as diagnostic, not as
evidence of a promoted controller.

## Reproduction artifacts

- Generate V1 data: `.venv/bin/python -m scripts.prepare_controlled_capacity`
- Generate V2 data: `.venv/bin/python -m scripts.prepare_controlled_capacity --spec experiments/scenarios/controlled_capacity_stress_v2.json --output data/controlled-capacity-stress-v2`
- Train each V2 seed with `abr.startup_training --protocol experiments/configs/controlled_capacity_training_v2.json --output-root runs/controlled-capacity-training-v2`
- Frozen V1 decisions: `runs/controlled-capacity-transfer-v1/results.json`
- Frozen V2 decisions: `runs/controlled-capacity-transfer-v2/results.json`
- Frame-aligned quality rows: `runs/single-task-vod-v1/vmaf_transfer.json`
- Validation checkpoint decisions: `runs/controlled-capacity-selection-v1/validation_decisions.json`
- Validation-only checkpoint VMAF: `runs/controlled-capacity-selection-v1/vmaf/vmaf_transfer.json`
- Frozen v3 reward comparison: `runs/vmaf-reward-stress-transfer-v1/results.json`
- V3 frame-aligned quality scores: `runs/vmaf-reward-stress-transfer-v1/vmaf/vmaf_transfer.json`

These data are synthetic application-goodput profiles, not measured radio
traces or packet-level simulations. The test draws use the same transition
families and parameter ranges as training. The videos share the same seven
nominal bitrate levels but have different segment sizes and durations. VMAF
measures encoded-frame quality, not viewer acceptance.
