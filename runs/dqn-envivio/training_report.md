# Dueling Double DQN: Envivio DASH replay

## Inputs and method

- One 193.68 second Envivio DASH VOD, 49 segments, six representations:
  300, 750, 1,200, 1,850, 2,850 and 4,300 kbps. The simulator uses exact
  segment byte sizes from the public media repository's Git tree.
- Eight measured Ghent 4G sessions: four train, two validation, two held-out
  test. These logs were collected in Belgium in 2015–2016. The importer assumes
  80 ms RTT because the logs do not include RTT.
- Dueling Double DQN, seed 7, 400 episodes, 19,600 environment steps and 4,776
  gradient updates.
- Stall ceiling: each validation trace must have no more rebuffering than its
  throughput baseline. Tolerance is 0 seconds. A checkpoint must also complete
  both validation traces. The held-out test was evaluated after selection.

## Results

| Split / policy | Completion | Mean stalls | Mean played bitrate | Mean reward |
| --- | ---: | ---: | ---: | ---: |
| Validation / DQN | 100% | 0.00 s | 4,300 kbps | 514.04 |
| Validation / throughput | 100% | 0.00 s | — | 502.14 |
| Validation / highest | 100% | 0.00 s | — | 514.04 |
| Test / DQN | 100% | 0.00 s | 4,300 kbps | 514.56 |
| Test / throughput | 100% | 0.00 s | — | 502.20 |
| Test / highest | 100% | 0.00 s | — | 514.56 |

The selected policy chose 4,300 kbps for all 49 segments on both validation
traces and both held-out traces. It met the 0 second validation stall ceiling.
Its test result exactly matches the fixed highest-bitrate baseline. This means
the run shows the policy can complete these specific replays without stalls;
it does not show that DQN learned better adaptation than a simple fixed choice.

These network sessions had high measured throughput relative to this video's
4,300 kbps top rung. This small sample is too easy to test bitrate adaptation.
The next useful experiment is to add independently collected, lower-throughput
and more variable sessions, then repeat the same hard stall comparison before
connecting the policy to a player. Do not treat this result as current-network
or real-player evidence.

Full per-segment selections, baseline episode metrics, validation limits,
checkpoint metadata and training history are in `training_metrics.json`.
The accepted validation-feasible model is `dueling_double_dqn_feasible.npz`.
