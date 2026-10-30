# Remaining long-stall diagnosis

## Finding

The dominant cause is a service deficit under the simulator buffer/ladder contract. Lowest-quality requests themselves arrive too slowly. No playback accounting defect was found. This is a diagnosis, not a runtime fix or a model promotion.

The frozen DQN completes the first failure session with 21.017259 s of stalls, startup threshold 4 s, resume threshold 1 s, and buffer cap 30 s.

## Reproduction and reduction

`.venv/bin/python -m scripts.diagnose_stalls --check` exits **1** on the retained 8-second stall-limit failure. Repeated original replays produced the same verdict.

`.venv/bin/python -m scripts.diagnose_stalls --check-minimal` also exits **1**. It uses `minimal_case.json`: one already-buffered unit, one lowest-quality request, and clipped measured trace intervals. No scores, earlier requests, retries or cancellation are needed. The player is restored to its already-started state.

The reduced request takes 14.105869 s with 3.993422 s buffered. Their difference is 10.112446 s, matching the counted stalls to numerical precision. Unused train/validation entries only satisfy the Dataset schema.

## Stall locations

Indices are zero-based. Service is integrated from the original body start to buffer exhaustion, assuming no bytes are wasted. These are same-state diagnostics, not global bounds.

| Segment | Request start s | Buffer s | Requested/effective kbps | Added stalls s | Minimum bytes | Service before buffer empty, bytes |
|---:|---:|---:|---|---:|---:|---:|
| 37 | 120.944 | 28.046 | 750/300 | 0.991 | 124011 | 125448 |
| 38 | 149.981 | 3.993 | 300/300 | 10.112 | 144254 | 21548 |
| 39 | 164.087 | 3.993 | 300/300 | 9.913 | 149991 | 34001 |

## One-factor counterfactuals

All rows cover completed videos. Zero RTT changes only RTT relative to lowest; buffer60 changes only the cap relative to the corresponding policy. Startup/resume thresholds remain 4/1 s. Total delay includes both startup and stalls.

| Case | Startup s | Stalls s | Total delay s | Played kbps | Discarded bytes | Peak buffer s | Targets met |
|---|---:|---:|---:|---:|---:|---:|---|
| dqn | 1.233 | 21.017 | 22.250 | 819.6 | 6717 | 28.661 | False |
| highest_allowed | 1.233 | 21.017 | 22.250 | 819.6 | 6717 | 28.661 | False |
| lowest | 1.233 | 20.933 | 22.166 | 300.0 | 0 | 29.656 | False |
| lowest_zero_rtt | 1.064 | 21.071 | 22.135 | 300.0 | 0 | 29.656 | False |
| lowest_buffer60 | 1.233 | 0.000 | 1.233 | 300.0 | 150142 | 59.619 | True |
| dqn_buffer60 | 1.233 | 0.000 | 1.233 | 985.6 | 200678 | 58.629 | True |
| dqn_no_backoff | 1.233 | 21.017 | 22.250 | 819.6 | 6717 | 28.661 | False |
| dqn_force_lowest_at37 | 1.233 | 20.933 | 22.166 | 819.6 | 0 | 28.661 | False |

Forcing lowest only at segment 37 removes the cancellation and saves just 0.084274 s of stalls. Removing backoff changes nothing here. Using lowest quality for the whole video still fails the stall target. Removing RTT starts playback earlier and slightly reduces total delay, but does not remove the later service deficit.

## Necessary waiting bound

Let C_i be media duration through segment i, P_i its media start, and B the buffer cap. The simulator starts a body only when buffer + duration_i <= B. Played media cannot exceed elapsed wall time at unit rate, so body i cannot start before max(0, C_i - B). Real startup/stalls only delay those buffer-space release times.

At these optimistic releases, transfer the smallest actual file per segment serially, with zero RTT, no retries, no timeouts and no discarded bytes. Denote the resulting earliest arrivals by a_i. No real schedule can deliver the same ordered segments earlier. To play segment i, elapsed time must be at least a_i but played time is only P_i; therefore total startup+stalls >= max_i(a_i - P_i). This is a necessary lower bound; the relaxed schedule need not itself be feasible.

For B=30 s, startup+stalls >= **22.105846 s** at segment 39. With startup <=3 s, stalls must be >= **19.105846 s**. The 8 s stall target cannot be met by any bitrate policy under this exact simulator contract.

## PPO answer

PPO learns a policy distribution from sampled rollouts and clipped policy updates. DQN learns action values from replay and a target network. Either can learn different choices; neither changes the service, cap, media files or necessary waiting bound. PPO under the same contract cannot meet the joint 3 s startup / 8 s stall target on this session. Its exact reward, bitrate and stall total remain unmeasured.

The local audit found 12 saved PPO checkpoints, all trained on the synthetic fixture with incompatible source hashes. They are not a matched real-input baseline and were not silently migrated or replayed. The current PPO trainer also lacks the DQN training-time filter and stall gates. A fair comparison requires matching data, reward, constraints, interaction budget and multiple seeds; these runs do not establish PPO versus DQN superiority.

Source: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## Next change to evaluate

Increasing the prefetch cap from 30 to 60 s removes stalls in both the DQN and lowest-quality diagnostic replays here, without changing startup threshold. This differs from increasing startup/resume thresholds: it permits fetching more media before the long slow period. The larger-cap runs also discard more bytes during retries. This is one inspected session, not validation of a new default.

Evaluate the larger prefetch cap on separate sessions, retaining startup/stall limits and recording downloaded/wasted bytes and memory cost. Alternatively, a lower bitrate rung needs real encoded segment files, not invented sizes. No additional training or production configuration change is made in this diagnosis.

Simulator-specific; same already-inspected failure session and trace start at zero; Observed goodput replay is not a causal radio/packet bound; Serial whole-segment VOD, pre-body backpressure, fixed ladder and unit playback rate; Future trace and exact candidate sizes are offline diagnostics only; Larger buffer changes DQN context outside its training distribution; No matched PPO was trained or measured on these real inputs.

Regenerate evidence: `.venv/bin/python -m scripts.diagnose_stalls`.
