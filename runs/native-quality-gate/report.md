# Native Chrome confirmation of the 4-second buffer gate

## Decision

The 4-second experimental threshold improves the selected upgrade case on both
videos, with no observed stalls or discarded downloads. It does not fix the
slow-network case. **Keep the 12-second default and frozen model unchanged.**
Only one of four candidate playbacks meets the proposed quality limits. These
results support further safety-rule diagnosis, not production approval or adding
neural layers.

## Fixed comparison

The [protocol](protocol.json) was saved before playback. Chrome
156.0.8078.74 decoded actual Envivio (193.68 s) and Sintel (52.2 s) media. Each
video ran DQN on two known development traces at both 12-second and 4-second
thresholds: eight complete native MSE playbacks. The upgrade trace was selected
because simulator ablation improved it; slow-first was selected as a case that
did not improve. This is a diagnostic confirmation, not independent selection.

The checkpoint hash, 60-second buffer cap, 0.65 throughput factor, retry/recovery
behavior, media and per-case trace were held fixed. The only policy-setting
change was the minimum buffer threshold. Capture context records both the
applied safety filter and the original checkpoint filter. The 4-second override
is explicit and opt-in; defaults and weights are unchanged.

The paired playback gate required completion, startup at most 3 seconds, stalls
at most 8 seconds, no additional stall seconds, and candidate startup at most
0.25 seconds above its paired baseline. That startup tolerance was recorded
before playback to allow browser timing variation; it is not a relaxed absolute
startup limit. Discarded downloads were measured and retained as outcomes.

## Results

All eight playbacks completed with **zero stalls, zero discarded media bytes,
and zero retries**. All four paired playback gates passed.

| Content / case | Startup: 12 s → 4 s | Emergency media: 12 s → 4 s | Mean bitrate: 12 s → 4 s | Candidate quality gate |
| --- | --- | --- | --- | --- |
| Envivio / slow-first | 2.774 → 2.612 s | 112 → 112 s | 637 → 605 kbps | Fail |
| Envivio / upgrade | 0.302 → 0.527 s | 16 → 8 s | 1,540 → 1,644 kbps | Pass |
| Sintel / slow-first | 2.755 → 2.647 s | 52.2 → 52.2 s | 8 → 8 kbps | Fail |
| Sintel / upgrade | 0.288 → 0.270 s | 16 → 8 s | 837 → 1,119 kbps | Fail |

Emergency exposure is encoded media duration below the proposed 300 kbps normal
floor. The draft limits allow at most 12 continuous seconds and 10% of media
duration below that floor. Sintel's improved 8 seconds still occupies **15.33%**
of its short duration, so it fails. The slow-first cases remain at 57.83% for
Envivio and 100% for Sintel. Viewer acceptance has not been established.

The 4-second setting removes 16 seconds of emergency media across four paired
video/session cases, from 196.2 to 180.2 seconds total (8.15%). The selected
upgrade case improves on each content, but the slow-first case is unchanged in
emergency exposure. Envivio slow-first's mean bitrate is slightly lower, so the
candidate is not uniformly better across all quality measures.

Native improvement was 8 seconds per upgrade case, compared with 12 seconds in
the simulator. Browser buffering and request timing differ from the simulator.
These results must not be reported as exact simulator/browser equivalence.
The Envivio upgrade candidate also started 0.225 seconds later than baseline;
this passed the declared tolerance but is not an improvement in startup.

## Integrity and checks

All eight captures passed the automated checks in [results.json](results.json):
checkpoint/dataset/trace identity, actual and original safety thresholds,
unchanged throughput margin and cap, override provenance, visible playback,
segment byte sizes, policy choices and allowed actions, complete segment order,
full played duration, event-derived stall accounting and finished transport.
Each Envivio playback reported 4,842 frames and each Sintel playback 1,305 frames.
Two baseline runs each reported one dropped frame; candidates reported none.
That is distinct from a playback stall.

18 focused tests passed for the explicit override, action safety, checkpoint
recovery contract, and native cancellation/recovery. The first sandboxed attempt
could not bind a local test socket; the permitted rerun passed. The override
retains the checkpoint's throughput factor, preserves the default when omitted,
and rejects invalid thresholds or checkpoints without a safety filter.

Four isolated Chrome processes ran concurrently per content on the same host,
with separate baseline and candidate servers. This controls some host conditions
but includes shared resource contention. One capture per setting/case is not a
statistical latency result. Browser RSS samples are retained as diagnostics, not
single-player memory approval, a combined worker measurement, or a memory soak.
The runner stopped its own Chrome processes and local servers after capture.

## Reproduce and next step

Review the archived captures and regenerate the comparison from this checkout:

```sh
.venv/bin/python -m scripts.validate_native_quality_gate --report-only
```

The capture command (without `--report-only`) launches local servers and Chrome
and refuses to overwrite a completed capture directory. Generated media caches
must be prepared as described in [startup-ladder documentation](../../docs/startup-ladder.md).

To inspect the candidate manually:

```sh
.venv/bin/python -m abr.player_server --dataset data/hard_envivio/dataset.json --checkpoint runs/learning-ablation/baseline-seed-7/accepted.npz --media-dir artifacts/startup-ladder-v1 --replay-dataset data/startup-ladder/development.json --allow-experimental-media --experimental-max-buffer-s 60 --enable-request-recovery --experimental-min-buffer-s 4
```

Next diagnose how the eight-sample harmonic throughput estimate recovers after
a slow period. Keep the buffer threshold and frozen weights fixed for that
comparison. Determine whether stale measurements, actual insufficient network
service, or recovery behavior explains the remaining long emergency intervals.
Do not reduce the throughput margin blindly or retrain before that evidence.

Evidence: [protocol](protocol.json), [paired audit](results.json),
[Envivio summary](envivio/summary.json), [Sintel summary](sintel/summary.json),
and [preceding simulator diagnosis](../quality-gate-diagnosis/report.md).
