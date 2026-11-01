# Learned throughput forecast with RobustMPC

## Question and design

The earlier DQN work asked one model to choose a representation directly. This
study tests a smaller responsibility: forecast near-future throughput, then let
the existing RobustMPC planner choose the representation.

```text
recent request history ──> p20 throughput forecast ──> RobustMPC quality plan
                                                └────> existing safety filter
```

The predictor receives eight completed-request history rows, current playback
state, and the upcoming segment duration. It predicts the 20th percentile of
the time-weighted mean controlled capacity during that next segment, after the
trace RTT. The target is capacity in Mbps. It does not see future trace values
at inference time. RobustMPC still scores quality, switching, and rebuffering;
the established action safety filter is unchanged.

The predictor is a small two-hidden-layer feed-forward network trained with
quantile loss. Samples come from highest-allowed playback on Envivio v4 training
profiles. Epoch selection uses v4 validation profiles. The final evaluation
uses all 12 reserved test profiles on Envivio and Sintel. The protocol and
scenario generator were frozen before that evaluation.

## Results

### Forecasts on the Envivio test observations

Both forecasts were scored on the same 588 decisions from highest-allowed
playback. The target is future segment-window mean capacity, not the exact
goodput of the selected representation.

| Forecast | MAE (Mbps) | RMSE (Mbps) | Underprediction | Overprediction |
| --- | ---: | ---: | ---: | ---: |
| Learned p20 | 0.223 | 0.746 | 82.0% | 18.0% |
| Robust harmonic-mean estimate | 0.263 | 1.010 | 44.2% | 55.8% |

The learned forecast has lower error on average and its test coverage is 18.0%,
near the intended p20 calibration of 20%. This aggregate hides family failures:
the learned forecast had lower MAE on repeated changes, slightly higher MAE on
step drops, and higher MAE on outage/degraded-recovery profiles. It is not a
uniformly better forecast for every transition type.

### Playback on the frozen test profiles

All 12 sessions completed for each policy on each video. None rebuffered. This
means the selected buffer, safety rule, and stress cases did not create a
meaningful stall trade-off in this comparison.

| Content | Policy | Mean reward | Mean quality utility / s | Mean bitrate (kbps) | Rebuffer (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Envivio | Highest-allowed | 893.448 | 4.675 | 2132.1 | 0 |
| Envivio | Harmonic RobustMPC | 878.971 | 4.596 | 2119.7 | 0 |
| Envivio | Learned forecast + RobustMPC | 893.534 | 4.675 | 2131.1 | 0 |
| Sintel | Highest-allowed | 226.959 | 4.460 | 1924.0 | 0 |
| Sintel | Harmonic RobustMPC | 226.959 | 4.460 | 1924.0 | 0 |
| Sintel | Learned forecast + RobustMPC | 226.959 | 4.460 | 1924.0 | 0 |

On Envivio, learned RobustMPC exceeded harmonic RobustMPC reward on 6 of 12
profiles and tied on 6; its mean reward was 14.56 higher per session. But it
essentially matched highest-allowed: mean reward was only 0.09 higher. All
Sintel decisions and scores matched across the three policies. The quality
column is the project's segment-VMAF-derived utility, not raw VMAF points.

Median decision time for the learned and harmonic planners was about 109 ms on
Envivio and 31 ms on Sintel. Highest-allowed took about 0.009 ms. The predictor
does not remove the substantial computation cost of RobustMPC's sequence
search.

## Decision

The predictor learned a useful forecast signal on this controlled dataset, but
the downstream benefit was small and content-dependent. It did not beat the
simple highest-allowed rule in a meaningful way, and it added a learned model
to a planner that is much slower than the rule. Do not train this model for
more epochs or promote it as the controller.

Keep the implementation as a completed architecture experiment. The next
evidence needed is a replay on a genuinely separate set of measured network
traces, with a buffer or stall regime that can expose a safety trade-off. If
that data is not available, stop this predictor line; the current synthetic
profiles are not enough to claim network generalization.

## Reproduction artifacts

- Protocol: [`throughput_predictor_v1.json`](../../experiments/configs/throughput_predictor_v1.json)
- Scenarios: [`controlled_capacity_predictor_v4.json`](../../experiments/scenarios/controlled_capacity_predictor_v4.json)
- Dataset provenance: `data/controlled-capacity-predictor-v4/provenance.json`
- Training report and checkpoint: `runs/throughput-predictor-v1/training.json` and `predictor.pt`
- Final comparison: `runs/throughput-predictor-v1/final-results.json`
- Build data: `.venv/bin/python -m scripts.prepare_controlled_capacity --spec experiments/scenarios/controlled_capacity_predictor_v4.json --output data/controlled-capacity-predictor-v4`
- Train: `.venv/bin/python -m scripts.train_throughput_predictor`
- Evaluate: `.venv/bin/python -m scripts.evaluate_throughput_predictor`

These are synthetic controlled application-goodput traces, not measured radio
or packet-level network traces. The test profiles use independent parameter
draws from the same known scenario families as training. The study measures
transfer across those draws only.

## Measured Ghent trace diagnostic

After freezing the synthetic-trained predictor, I evaluated it on all 32 test
sessions in `data/ghent-holdout/dataset.json`. The source contains historical
measured HTTP goodput from a Ghent 4G/LTE archive. Its RTT is not recorded, so
the replay assumes 80 ms. The model was not retrained or calibrated on these
sessions. These traces were used in earlier project analyses, so this is
measured transfer evidence for this predictor, not a newly untouched project
benchmark. The source archive, file hashes, interval conversion, and session
selection are recorded in that dataset's provenance.

### Forecast result

On 1,568 decisions from highest-allowed playback, the learned forecast had
10.54 Mbps MAE and 13.83 Mbps RMSE. The harmonic-mean estimate had 13.88 Mbps
MAE and 17.72 Mbps RMSE. The learned forecast's p20 coverage was 14.3%, versus
the intended 20%; it was conservative on 85.7% of the samples. Its error was
lower in each of the six transport-mode groups in this collection. The absolute
errors remain large because the measured capacity often exceeds this video's
4.3 Mbps top representation by a wide margin.

### Playback result

| Policy | Completed | Mean reward | Mean quality utility / s | Mean bitrate (kbps) | Rebuffer (s) | Median decision time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Highest-allowed | 32/32 | 1074.50 | 5.583 | 3867.2 | 0 | 0.008 ms |
| Harmonic RobustMPC | 32/32 | 1071.99 | 5.570 | 3862.8 | 0 | 129.64 ms |
| Learned forecast + RobustMPC | 32/32 | 1074.26 | 5.582 | 3866.9 | 0 | 128.52 ms |

The learned controller tied with highest-allowed on 31 sessions and scored
lower on the remaining bicycle session. It beat harmonic RobustMPC on that
same bicycle session and tied on the other 31. No policy stalled or failed to
complete. The result is consistent with the synthetic study: a better forecast
can change one difficult decision sequence, but it does not provide a
repeatable playback gain over the simple rule here. RobustMPC also remains
orders of magnitude slower than highest-allowed.

### Updated decision

Do not continue training or promote the learned predictor. Keep it as an
architecture experiment and keep highest-allowed as the practical reference.
The current data show that the predictor transfers better than the harmonic
estimate by forecast error, but the action ladder is saturated on nearly every
measured session. A useful follow-up needs newly collected measured traces that
include sustained service near and below the video ladder, plus recorded RTT,
so forecast changes can affect completion or stalls. Historical traces cannot
answer whether the policy works on today's networks.

The frozen diagnostic is defined in
[`throughput_predictor_ghent_diagnostic_v1.json`](../../experiments/configs/throughput_predictor_ghent_diagnostic_v1.json)
and its full output is `runs/throughput-predictor-v1/ghent-measured-results.json`.
Reproduce it with `.venv/bin/python -m scripts.evaluate_measured_throughput_predictor`.
