# DQN training-budget comparison

## Decision

Do not spend more training time on the current setup. Doubling the run budget
from 120 to 240 episodes produced no improvement because validation selected the
same checkpoint episode for every seed in both runs. The selected 240-episode
models have the same learned weights as their 120-episode references. Their
decisions and measured results are identical on the frozen test sessions.

The recommended follow-up was to add controlled bandwidth changes and a second
video ladder, then compare with RobustMPC and BOLA. That work is now recorded
in the [controlled-capacity study](controlled_capacity_transfer.md) and the
[single-task VOD study](single_task_vod_study.md). A later supervised
throughput-predictor experiment also compared a learned estimate with
RobustMPC's harmonic estimate on synthetic and historical measured traces; see
the [predictor study](throughput_predictor_study.md). Those studies did not
establish a repeatable playback gain, so do not increase the training budget.

## Frozen protocol and data

The machine-readable protocol is
[`dqn_training_budget_v1.json`](../../experiments/configs/dqn_training_budget_v1.json).
The test set contains 32 sessions from the Ghent 4G archive. It excludes the
eight filenames already present in the earlier Ghent pilot. The set was frozen
before this matched training run, and it was evaluated after all six selected
checkpoints had fixed SHA-256 hashes. The selected episode numbers are 100, 80,
and 40 for seeds 7, 11, and 13 in both arms.

The independent collection is useful, but its source metadata groups traces by
transport mode and does not prove that every session represents a distinct
geographic route. Its throughput is historical observed HTTP goodput. Replay
assumes 80 ms RTT. This is not a current-network or packet-level benchmark.

## Results

All policies completed all 32 sessions with zero rebuffering. Mean startup was
0.112 seconds for every policy. Therefore, these sessions did not expose a
completion or stall trade-off.

| Policy | Mean reward proxy | Mean bitrate (kbps) | Mean case VMAF | Decision p50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| DQN 120, seed 7 | 869.75 | 1,057.1 | 74.328 | 0.325 |
| DQN 120, seed 11 | 952.66 | 1,664.9 | 83.242 | 0.324 |
| DQN 120, seed 13 | 925.34 | 1,468.9 | 80.388 | 0.325 |
| DQN 240, each matching seed | Identical to its 120 run | Identical | Identical | 0.323–0.324 |
| Highest allowed | 1,101.23 | 3,867.2 | 88.518 | 0.008 |
| RobustMPC, 5-chunk horizon | 1,098.68 | 3,862.8 | 88.313 | 128.905 |

The VMAF column is the mean of per-session scores. Each session score weights
completed segment VMAF by segment duration. The actual Envivio encodes and
frame-aligned reference video were used. VMAF measures encoded-frame quality;
it does not measure viewer acceptance.

The selected 120 and 240 checkpoints have different archive hashes because
checkpoint metadata records a different training protocol hash. For each seed,
the loaded model weights are identical, and all 32 action sequences match. The
longer runs continued training after the best validation checkpoints but did
not select any later checkpoint. More episodes therefore did not improve this
training setup.

RobustMPC nearly matched highest-allowed quality on these high-service traces,
at much higher computation cost. The three DQN seeds chose more conservative
representations, reducing VMAF by 5.3 to 14.2 points versus highest-allowed with
no measured reduction in rebuffering here. This is evidence about this frozen
trace set only. It does not establish that highest-allowed is safer on hard
outages.

## Artifacts

- Training configuration: [`startup_ladder_240.json`](../../experiments/configs/startup_ladder_240.json)
- Frozen test replay: `runs/startup-ladder-budget-240/holdout.json`
- VMAF results: `runs/single-task-vod-v1/vmaf_transfer.json`
- Human-readable VMAF summary: `runs/single-task-vod-v1/report.md`
- 240-episode seed metrics: `runs/startup-ladder-budget-240/seed-{7,11,13}/metrics.json`

## Limits and next experiment

The holdout uses the same Envivio source content and seven-rung encoding ladder
as training. The test is new at the session level, but it does not test content
generalization. All traces completed cleanly under the chosen 60-second buffer
cap, so the comparison cannot reveal behavior during severe outages. The model
uses the existing log-bitrate objective, which is only a quality proxy.

The controlled-capacity pilots, checkpoint-selection comparison, and
VMAF-aligned reward study are complete; see the
[controlled-capacity transfer report](controlled_capacity_transfer.md). The
stress-trained DQN improved some seed/content pairs but regressed on others.
The follow-up learned throughput predictor also failed to provide a repeatable
playback gain; see the [predictor report](throughput_predictor_study.md). Do
not increase the training budget or continue model training on the current
data.
