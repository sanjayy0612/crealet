# Buffer-aware recovery: complete experiment, candidate rejected

**Do not promote this candidate.** Implementation, fixed simulator comparison,
native Chrome confirmation, regression tests and capture audits are complete.
The candidate improves some recoveries but fails both a simulator transfer case
and a native controlled case. Model weights, original neural sources and player
defaults remain unchanged. The adapter is available only through an explicit
experimental flag for reproduction.

## Implemented check

The [adapter](../../abr/buffer_safety.py) preserves every action permitted by
the existing eight-sample harmonic filter. It may additionally permit only the
300 kbps normal floor when all these conditions hold:

- At least three valid recent measurements exist, and each supports 300 kbps
  after the unchanged 0.65 throughput margin.
- Buffer is at least 12 seconds and also meets the configured original gate.
- A nominal segment-size estimate, inflated by 25%, plus recent measured
  header/RTT overhead fits a conservative download budget.
- The budget leaves one segment for possible buffer-cap backpressure, keeps
  four seconds of playback reserve, and is capped at eight seconds.

In the existing units the calculation is:

```text
rate_Mbps = 0.65 × min(last_three_observed_rates_Mbps)
download_s = 1.25 × nominal_segment_MB × 8 / rate_Mbps + observed_header_s
budget_s = min(8, buffer_s - segment_duration_s - 4)
```

The model still sees all eight history rows and uses its original scores. The
adapter does not access future trace service or unrequested actual segment
sizes. These constants are experimental engineering choices, not accepted
product requirements or formal safety bounds. Nominal size error, fresh init
requests, future outages and browser timing can invalidate the estimate.

An initial implementation inside the model package was rejected by the strict
checkpoint-source guard before any rollout. The implementation was moved to
an external adapter; checkpoint validation stayed strict. The initial unscored
[draft protocol](protocol-initial-load-blocked.json) is retained separately.
All scored outcomes use the final [simulator protocol](protocol.json), written
before scoring. No checkpoint was migrated, regenerated or retrained.

## Simulator comparison

Both settings use the same frozen checkpoint, a 4-second base buffer gate,
60-second cap, 0.65 margin and bounded request recovery. Compare the original
filter against the additional buffer-aware check, for DQN and highest-allowed
selection on both videos. Inputs include validation, development and previously
used transfer sessions: **96 rollouts across 24 video/session combinations**.
These are known inputs, not new independent acceptance data.

The table gives totals **per policy**. DQN and highest-allowed produced the same
aggregate values; totals are not added across the two policies.

| Setting | Complete | Startup/stall targets met | Draft quality passes | Emergency media | Stalls | Wasted bytes |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Original filter | 24/24 | 22/24 | 11/24 | 915.32 s | 0 s | 669,137.06 |
| Buffer-aware candidate | 24/24 | 22/24 | 11/24 | 847.32 s | 0 s | 669,137.06 |

The candidate removes 68 seconds of emergency media in aggregate, with no added
stalls, startup delay or wasted bytes in any paired simulator rollout. The two
existing startup failures per policy remain failures, not relaxed targets.

The controlled Envivio validation case remains at 12 seconds of emergency media
in the simulator, avoiding the earlier short-history regression to 76 seconds.
However, on Envivio `transfer-ferry-2011-02-01_1539CET`, emergency media increases
from **64 to 80 seconds**, and mean played bitrate falls from about **698 to
613 kbps**. Both policies reproduce this failure. Thus 46 of 48 paired policy
comparisons pass, and **the simulator promotion gate fails**. Aggregate gains
do not override this negative result.

Adding allowed actions does not guarantee better full-session playback: larger
early downloads shift the time of later requests. Those later requests encounter
different recorded service and produce different histories.

## Native Chrome confirmation

The separate [native protocol](native-protocol.json) fixed the comparison before
capture: slow-first test trace 0 and controlled validation trace 2 on Envivio and
Sintel, original filter versus candidate. This directly covers the intended
recovery and the earlier short-history failure. Native confirmation was planned
when a measured development recovery improved without added stalls; it was not
used to override the failed transfer gate.

Chrome 156.0.8078.74 completed **all eight native playbacks**, all with zero
stalls, startup below three seconds, and no dropped frames. Envivio had 4,842
frames per playback; Sintel had 1,305.

| Content / case | Emergency media: baseline → candidate | Mean bitrate: baseline → candidate | Candidate quality pass | Paired gate |
| --- | --- | --- | --- | --- |
| Envivio / slow-first | 112 → 96 s | 614 → 629 kbps | Fail | Pass |
| Envivio / controlled | 12 → 24 s | 1,119 → 1,055 kbps | Fail; baseline passed | **Fail** |
| Sintel / slow-first | 52.2 → 52.2 s | 8 → 8 kbps | Fail | Pass |
| Sintel / controlled | 8 → 8 s | 1,368 → 1,368 kbps | Fail | Pass |

The draft quality gate requires at most 12 continuous seconds and 10% of media
below 300 kbps. Native controlled Envivio grows from a longest interval of eight
seconds to sixteen, and emergency exposure grows from 6.20% to 12.39%. This
loses an existing quality pass. **No candidate capture meets the draft quality
gate.** The paired gate also checks completion, no added stalls, no additional
discarded media, and startup at most 0.25 seconds above baseline while still
below three seconds. Three of four pairs pass; the controlled case fails.

Controlled Envivio required one retry under each setting. Its baseline discarded
131,072 media bytes; the candidate discarded none. All other captures had no
retries or discarded media. The wasted-byte improvement does not excuse the
quality regression. Sintel's pre-recovery low-quality prefetch remains unresolved.

The native controlled result differs from the simulator's unchanged 12 seconds.
Real request, initialization and recovery timing matter. Do not claim exact
simulator/browser equivalence or that simulator success proves quality safety.

Four isolated Chrome processes ran concurrently per content on one host. This
is a paired behavioral check with shared resource contention, one run per
setting/case, not a statistical latency study or a single-player memory approval.
RSS samples are retained as diagnostics; no memory soak was run. The runner
stopped its own servers and browsers after capture.

## Verification and reproduction

- **176 tests passed**, with two existing Gymnasium infinite-Box warnings.
- The original neural source hash still matches the frozen checkpoint; weights
  are unchanged. The adapter does not bypass checkpoint eligibility or required
  request-recovery configuration.
- All 26 comparable baseline rollouts match the earlier 4-second experiment's
  completion, startup, stalls, mean bitrate, reward, emergency media and waste.
- All eight native captures pass identity, media-byte, visibility, full-duration,
  complete-sequence, choice/filter, event-derived stall and transport checks.
  Every candidate mask is independently recomputed from the actual recorded
  history and state, including header timing, using the recorded adapter config.

```sh
.venv/bin/python -m scripts.evaluate_buffer_recovery
.venv/bin/python -m scripts.validate_buffer_recovery --report-only
.venv/bin/python -m pytest -q
```

Full capture uses the native command without `--report-only`; it launches local
servers and Chrome and refuses to overwrite completed capture directories.
Generated media caches are required. All original defaults remain unchanged.
The explicit `--experimental-buffer-aware-safety` player flag is a research
reproduction option, not an approved deployment setting.

The experiment is complete and the candidate is rejected. A further design
would first need to explain the two regression paths, then pass the same frozen
negative cases and native checks. Additional layers or training have not been
shown to fix either failure.

Evidence: [simulator outcomes](simulator.json), [native outcomes and audit](native.json),
[verification](verification.json), [preceding throughput diagnosis](../throughput-diagnosis/report.md).
