# Bounded request-recovery replay

Frozen DQN weights; no new training. Settings were saved before replay.

## Two previously failed sessions

Each row is one complete or explicitly partial playback. Wasted bytes are discarded bodies.

| Policy | Recovery | Session | Complete | Played % | Stall s | Startup s | Played kbps | Wasted MB | Retries |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| lowest | none | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 21.50 | 0.65 | 300.0 | 0.000 | 0 |
| lowest | none | fresh-ferry-2010-09-21_1735CEST | no (partial) | 66.0 | 6.44 | 1.00 | 300.0 | 0.094 | 0 |
| throughput | none | fresh-ferry-2010-09-21_1622CEST | no (partial) | 76.3 | 7.68 | 0.65 | 1579.7 | 0.157 | 0 |
| throughput | none | fresh-ferry-2010-09-21_1735CEST | no (partial) | 63.9 | 10.46 | 1.00 | 445.2 | 0.093 | 0 |
| highest_allowed | none | fresh-ferry-2010-09-21_1622CEST | no (partial) | 76.3 | 1.78 | 0.65 | 981.1 | 0.133 | 0 |
| highest_allowed | none | fresh-ferry-2010-09-21_1735CEST | no (partial) | 66.0 | 6.44 | 1.00 | 398.4 | 0.094 | 0 |
| dqn | none | fresh-ferry-2010-09-21_1622CEST | no (partial) | 76.3 | 1.78 | 0.65 | 981.1 | 0.133 | 0 |
| dqn | none | fresh-ferry-2010-09-21_1735CEST | no (partial) | 66.0 | 6.44 | 1.00 | 398.4 | 0.094 | 0 |
| lowest | timeout_only | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 21.50 | 0.65 | 300.0 | 0.000 | 0 |
| lowest | timeout_only | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 9.93 | 1.00 | 300.0 | 0.094 | 1 |
| throughput | timeout_only | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 25.81 | 0.65 | 1276.3 | 0.157 | 1 |
| throughput | timeout_only | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 14.06 | 1.00 | 550.5 | 0.093 | 1 |
| highest_allowed | timeout_only | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 25.42 | 0.65 | 819.6 | 0.133 | 1 |
| highest_allowed | timeout_only | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 9.93 | 1.00 | 434.5 | 0.094 | 1 |
| dqn | timeout_only | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 25.42 | 0.65 | 819.6 | 0.133 | 1 |
| dqn | timeout_only | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 9.93 | 1.00 | 434.5 | 0.094 | 1 |
| lowest | progress | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 21.50 | 0.65 | 300.0 | 0.000 | 0 |
| lowest | progress | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 9.93 | 1.00 | 300.0 | 0.094 | 1 |
| throughput | progress | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 21.72 | 0.65 | 1276.3 | 0.009 | 1 |
| throughput | progress | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 10.37 | 1.00 | 522.7 | 0.202 | 2 |
| highest_allowed | progress | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 21.58 | 0.65 | 819.6 | 0.007 | 1 |
| highest_allowed | progress | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 9.93 | 1.00 | 434.5 | 0.094 | 1 |
| dqn | progress | fresh-ferry-2010-09-21_1622CEST | yes | 100.0 | 21.58 | 0.65 | 819.6 | 0.007 | 1 |
| dqn | progress | fresh-ferry-2010-09-21_1735CEST | yes | 100.0 | 9.93 | 1.00 | 434.5 | 0.094 | 1 |

## Historical controls

| Dataset | Policy | Recovery | Complete | Timing targets passed | Max stall s |
|---|---|---|---:|---:|---:|
| historical_validation | lowest | none | 3/3 | 3/3 | 6.49 |
| historical_validation | throughput | none | 3/3 | 2/3 | 13.00 |
| historical_validation | highest_allowed | none | 3/3 | 3/3 | 6.49 |
| historical_validation | dqn | none | 3/3 | 3/3 | 6.49 |
| historical_validation | lowest | timeout_only | 3/3 | 3/3 | 6.49 |
| historical_validation | throughput | timeout_only | 3/3 | 2/3 | 13.00 |
| historical_validation | highest_allowed | timeout_only | 3/3 | 3/3 | 6.49 |
| historical_validation | dqn | timeout_only | 3/3 | 3/3 | 6.49 |
| historical_validation | lowest | progress | 3/3 | 3/3 | 6.49 |
| historical_validation | throughput | progress | 3/3 | 2/3 | 12.08 |
| historical_validation | highest_allowed | progress | 3/3 | 3/3 | 6.49 |
| historical_validation | dqn | progress | 3/3 | 3/3 | 6.49 |
| historical_test_already_observed | lowest | none | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | throughput | none | 3/3 | 3/3 | 3.48 |
| historical_test_already_observed | highest_allowed | none | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | dqn | none | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | lowest | timeout_only | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | throughput | timeout_only | 3/3 | 3/3 | 3.48 |
| historical_test_already_observed | highest_allowed | timeout_only | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | dqn | timeout_only | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | lowest | progress | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | throughput | progress | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | highest_allowed | progress | 3/3 | 3/3 | 0.00 |
| historical_test_already_observed | dqn | progress | 3/3 | 3/3 | 0.00 |

## Decision

Progress-based recovery with frozen DQN completed 2/2 failure sessions; 0/2 met completion, startup <=3 s and stalls <=8 s together.

DQN and the highest-allowed rule chose identical actions on both recovery sessions: True. This run provides no evidence of a learned-policy advantage.

Compared with timeout-only retries, progress recovery reduced total DQN stalls from 35.35 s to 31.51 s, and discarded bytes from 0.227 MB to 0.101 MB. Both comparisons cover two complete videos.

Completion alone does not justify promotion. Partial-playback stalls cannot be compared as if they covered a full video. These sessions are development cases, not untouched tests.

Progress recovery can cancel a higher-quality body after 2 seconds if finishing it risks draining the buffer, or after 3 seconds without progress. It avoids abandoning near-complete bodies and uses nominal size for a proposed lowest-quality restart. All retries use the lowest quality. Each retry has a 30-second deadline; the limit is four retries per segment. Timeout/no-progress retries wait 2 seconds. Backoff, RTT, downloaded bytes and playback stalls are all charged.

Next experiment: measure the tradeoff between startup delay and a larger startup/resume buffer on separate network sessions, against the same simple rule. Compare full playback and keep the stall limit. This run does not prove that extra training can remove the stalls.

Limitations: Simulator only; native browser recovery not implemented; Known sessions; no untouched test or new video; Finite observed-goodput replay, not a causal network model; Failed bodies restart in full; no HTTP Range resumption; Policy history includes received bytes from failed attempts; All settings fixed before replay; no selection using these results.

Reproduce: `.venv/bin/python -m scripts.evaluate_request_recovery`
