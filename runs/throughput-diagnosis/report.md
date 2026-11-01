# Why quality remains low after a slow period

## Finding

There are two distinct causes in the captured slow-first cases:

1. **Envivio:** old slow measurements dominate the eight-sample harmonic mean
   after network recovery. A nearly full buffer slows replacement of those
   measurements. This adds about 20 seconds of observed permission delay compared
   with the last three measurements on the same captured trajectory.
2. **Sintel:** the entire 52.2-second video downloads before the recorded network
   recovers. There are no later requests to produce fast measurements. This is
   early low-quality prefetch, not stale recovery history in those decisions.

Envivio and Sintel are sample video contents, not models. The model selects a
quality version of their next segment. Envivio's test media lasts 193.68 seconds.

Simply shortening history is not a complete fix: the controlled replay exposed
quality regressions. Production defaults, model weights and architecture remain
unchanged. This work completes the diagnosis; it does not claim the symptom fixed.

## Reproduction and ranked hypotheses

The deterministic captured-state check was run before the ablation:

```sh
.venv/bin/python -m scripts.diagnose_throughput --check-stale
```

It printed `captured_8_vs_3_floor_lockouts: 10`, reconstructed all eight native
captures' actual safety masks, and exited **1**. That is the expected diagnostic
failure signal: the recorded lockouts still exist. It is not a production
acceptance gate or a test that shorter history must always be safe.

The ranked hypotheses were stated before testing:

| Hypothesis | Prediction | Evidence / outcome |
| --- | --- | --- |
| Old slow samples dominate the harmonic mean | Shorter history permits earlier recovery | Confirmed in Envivio captured states and several simulator sessions |
| Full buffer slows replacement of old samples | Near-cap requests become paced by media consumption | Confirmed: buffer about 59.9 s and roughly 4 s between requests in the recovery interval |
| Short video finishes before recovery | No fast measurements are collected before the last decision | Confirmed for Sintel slow-first |
| Retry or small-body accounting distorts rates | Observed rates conflict with actual low service or retries explain the low measurements | No retries or discarded bodies in these native captures; early measured rates match the recorded low service. Later tiny-body timing remains a measurement limitation |

The minimal reproduced pattern contains one 29 kbps observation followed by
three 2,000 kbps observations. With the unchanged 0.65 margin, their harmonic
estimate is about 72 kbps, below the 300 kbps normal floor. Using only the three
fast observations yields 1,300 kbps. This demonstrates the stale-sample mechanism;
it does not prove that a faster estimate predicts future service safely.

## Native timeline evidence

The [native audit](native-audit.json) reconstructs the valid rate history from
each completed segment: total attempt bytes × 8 / body seconds, converted to
Mbps and float32, exactly as the browser does. It combines those rates with the
server's recorded buffer state and calls the actual `allowed_actions` function.
Every original allowed-action mask matched. No model scores were reconstructed
from incomplete historical inputs.

In both Envivio slow-first captures, the last-three estimate first permitted
300 kbps at segment index **23**. The actual eight-sample filter did not permit
the first upgrade until index **28**. Indices are zero-based. These five blocked
decisions per capture account for the ten reproduced lockouts; they are not ten
independent sessions.

At index 23 in the 4-second capture:

| Measurement | Value |
| --- | ---: |
| Buffered media | 59.88 s |
| Eight-sample estimate after margin | 29.08 kbps |
| Last-three estimate after margin | 320.75 kbps |
| Latest-only estimate after margin | 1,130.08 kbps |

The old samples disappear only when more requests complete. With four-second
segments and the buffer near its 60-second cap, backpressure leaves about four
seconds between new completed measurements. Five extra replacements therefore
take about 20 seconds. Completion-timestamp proxies give **20.010 s** in the
4-second capture and **19.977 s** in the 12-second capture. Exact decision
timestamps are not archived; these are completion-to-completion timing proxies,
not precise decision-latency measurements.

The trace's first intervals deliver about 21.43 and 29.07 kbps until trace time
**28.972 s**. Early native body-rate measurements were about 21.4 and 28.9 kbps:
the initial low estimate is justified, not a kbps/Mbps unit error. The 0.65
margin also means a steady 300 kbps measurement would permit only 195 kbps;
about 461.54 kbps observed harmonic throughput is needed to allow a nominal
300 kbps candidate. This conservatism is intentional, not evidence to remove
the margin without a safety comparison.

Sintel's last fragment arrived at capture time **18.896 s** with the 4-second
setting and **19.060 s** with the 12-second setting. Both are before the slow
trace interval ends near 28.972 s. Capture and session clocks have a small
launch offset; the roughly ten-second ordering gap is much larger. Its full
52.2-second duration fits inside the 60-second buffer cap, so the remaining
playback consumes already-downloaded low-quality media. A shorter throughput
history cannot observe a recovery that happened after downloading finished.

See [derived timelines and source hashes](timelines.json). These are analyses of
existing captures; this task did not run new Chrome sessions.

## Controlled full-session replay

The [protocol](protocol.json) was saved before scoring. Compare safety histories
of **8, 3 and 1 measurements**, leaving model history at eight rows, buffer gate
at **4 seconds**, cap at 60 seconds, throughput margin at 0.65, recovery behavior,
checkpoint and reward weights unchanged. History restriction applies only to
the filter's copied observation; it does not alter the model input or original
environment history.

The frozen DQN and highest-allowed rule each replayed thirteen known video/trace
combinations: three Envivio validation, five Envivio development, and five Sintel
development cases. This is **78 simulator rollouts**. The suite includes a
synthetic controlled validation trace; reused traces across videos are not
independent network samples. No unused-session result selected these variants.

DQN and the rule have the same aggregate values below. Totals are per policy,
not sums across both policies.

| Safety history | Completed / playback targets | Draft quality gate | Emergency media | Stalls | Wasted bytes |
| --- | --- | --- | ---: | ---: | ---: |
| 8 samples | 13/13 | 6/13 | 495.56 s | 0 s | 319,684.81 |
| 3 samples | 13/13 | 5/13 | 483.56 s | 0 s | 293,155.25 |
| 1 sample | 13/13 | 6/13 | 443.56 s | 0 s | 309,119.75 |

All changed rollouts completed without added stall seconds or worse startup.
That is a playback finding, not a claim of no quality regressions. The
three-sample variant adds **64 seconds** of emergency media on the controlled
validation case, changing 12 seconds to 76 seconds, and loses its quality gate.
It also adds four seconds in another validation case. The one-sample variant
also adds 64 seconds on the controlled case. Those failures are retained even
though aggregate emergency exposure improves.

Thus neither shortcut uniformly improves quality. In particular, a lower total
does not justify promoting latest-only history, which is sensitive to individual
sample timing and has not been checked in native playback here. Existing
single-request 300 kbps probes are stored only as offline deep-copy diagnostics;
future recorded service never influences policy action selection.

Raw [replay outcomes and paired comparisons](replays.json) retain reward, bitrate,
wasted bytes and negative results. The 26 eight-sample baseline rollouts match
the preceding 4-second experiment's startup, stalls, completion, bitrate, reward,
wasted bytes and emergency exposure.

## Verification and next action

11 focused tests passed. They cover the minimal stale-sample pattern, the filter
still respecting a low buffer, copied history isolation, empty history, original
action safety and independent counterfactual requests. See [verification](verification.json).

Reproduce the successful audit and ablation with:

```sh
.venv/bin/python -m scripts.diagnose_throughput --replay
.venv/bin/python -m pytest tests/test_throughput_diagnosis.py tests/test_quality_gate_diagnosis.py tests/test_action_safety.py -q
```

The `--check-stale` command intentionally remains red against immutable captures;
it should not be made green by changing historical evidence. No runtime fix or
new estimator has been promoted.

Next design and test a **buffer-aware recovery check**: require sustained recent
service and compare conservative estimated next-segment download time against
usable buffer. Evaluate recovery and falling-speed cases separately, including
the controlled case that regressed. Any future probing rule must account for
request size, measurement reliability, timeout/recovery and wasted bytes, without
using future trace service. Short-video low-quality prefetch needs a separate
quality-versus-prefetch decision; a smaller history alone cannot solve it.
