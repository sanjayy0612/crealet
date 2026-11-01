# Emergency-quality safety-gate diagnosis

## Decision

The 12-second buffer threshold is unnecessarily restrictive in some of the
examined simulator states. Reducing it to 4 seconds is a useful experimental
candidate, but does not solve prolonged emergency quality. Keep production
defaults and model weights unchanged until native playback confirms the change.
Adding neural layers is not supported by this diagnosis.

## Method

The [protocol](protocol.json) was written before scoring these interventions.
The original frozen checkpoint, 60-second buffer cap, 0.65 throughput factor,
recovery behavior and reward weights were held fixed. Only the safety rule's
minimum buffer changed: 12 seconds (baseline), 4 seconds, or 0 seconds.

Each setting ran the frozen DQN and the highest-allowed rule over three Envivio
validation cases, five Envivio development cases, and five Sintel development
cases: **78 rollouts across 13 video/session combinations**. These are known
cases, including one synthetic controlled validation trace, not independent
final tests. Reused traces across videos are not independent network samples.
All runs are finite-trace simulator replays, not new browser captures.

For each baseline DQN decision the audit records raw candidate scores, preferred
action before the safety filter, allowed actions, buffer, throughput estimate,
and delivered action after request recovery. At every emergency decision, a
deep copy of the same environment tries the actual 300 kbps segment, with the
same request timeout and recovery. The original rollout is not advanced by that
branch. A probe succeeds only if the requested quality is delivered, no stall is
added, and startup remains within 3 seconds if playback has not yet begun.

These branches use future recorded network service **only offline**. A successful
single request does not establish that all later requests will remain safe, or
provide an implementable oracle policy. No future-service values enter the
model or either threshold-ablation policy.

## Root-cause evidence

Across 140 baseline emergency decisions:

| Observation | Decisions |
| --- | ---: |
| Raw model preferred a higher quality | 140 |
| Normal floor blocked by buffer threshold alone | 28 |
| Normal floor blocked by throughput estimate, possibly also by buffer | 112 |
| Offline 300 kbps request delivered without added stall/startup violation | 108 |
| Successful probes among the buffer-only blocks | 28 of 28 |

Thus the model was not requesting emergency quality in these states; the safety
filter constrained it. This does **not** prove that the model's preferred higher
action would itself be safe. The probes test 300 kbps, not arbitrary raw choices.

The buffer threshold explains only part of the behavior. Another 80 successful
single-request probes were blocked by the throughput estimate. The estimate uses
0.65 times the harmonic mean of up to eight observed body-transfer rates. This
is evidence of conservative decisions on these states, not proof that removing
the throughput margin or shortening history is safe. Future service may change.

## Controlled replay results

DQN and the highest-allowed rule have the same aggregate results below. Counts
and exposure totals are **per policy**, not added across the two policies.

| Minimum buffer | Completed | Startup/stall targets met | Draft quality gate met | Total emergency media | Total stalls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 s | 13/13 | 13/13 | 0/13 | 563.56 s | 0 s |
| 4 s | 13/13 | 13/13 | 6/13 | 495.56 s | 0 s |
| 0 s | 13/13 | 13/13 | 6/13 | 495.56 s | 0 s |

The 4-second setting reduces emergency exposure by **68 seconds (12.1%)** per
policy across the suite. Eight of thirteen cases improve. Each paired rollout
completes with no added stalls and no worse startup. Removing the buffer gate
entirely provides no additional benefit here. Prefer testing the less aggressive
4-second candidate before considering 0 seconds.

Seven cases still fail the draft quality limits: normal quality at least
300 kbps, emergency intervals at most 12 seconds, and emergency exposure at most
10% of media duration. These are the existing proposed product limits, not
viewer-accepted standards. A second content does not demonstrate generalization
to a broad media library. The separate unused-session startup failure was not
part of this diagnostic suite and remains unresolved.

## Verification and next action

- All sixteen Envivio baseline rollouts match the archived frozen simulator
  startup, stalls, completion, reward, bitrate and wasted-byte measurements.
- 31 focused tests passed, covering safety selection, recovery accounting, and
  offline-branch isolation. A recovery fallback cannot count as delivery of
  normal quality even when it avoids stalls.
- Checkpoints, architecture, player defaults and production approval are unchanged.

Reproduce with:

```sh
.venv/bin/python -m scripts.diagnose_quality_gate
.venv/bin/python -m pytest tests/test_quality_gate_diagnosis.py tests/test_action_safety.py tests/test_recovery.py -q
```

Next test the 4-second candidate in native Chrome against the original 12-second
setting, with explicit experiment metadata and unchanged weights. Measure startup,
stalls, emergency exposure and wasted bytes. Separately diagnose throughput
history freshness before any new rate-estimation change. Do not weaken both
buffer and throughput rules at once or promote an architectural change based on
these results.

Raw evidence: [all decisions and counterfactuals](results.json),
[summary](summary.json), [frozen inputs and interventions](protocol.json).
