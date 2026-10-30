# Native request recovery and startup feasibility

The local MSE player now has opt-in bounded recovery. The frozen model and
default 30-second buffer cap are unchanged. The research comparison explicitly
sets a 60-second cap on both recovery-on and recovery-off servers.

```sh
.venv/bin/python -m abr.player_server --port 8850 --replay-dataset data/prefetch-sessions/dataset.json --experimental-max-buffer-s 60
.venv/bin/python -m abr.player_server --port 8851 --replay-dataset data/prefetch-sessions/dataset.json --experimental-max-buffer-s 60 --enable-request-recovery
node scripts/capture_policy_player.mjs --port=8850 --cases=runs/native-recovery/cases.json --memory --output=runs/native-recovery/captures
.venv/bin/python -m scripts.report_native_recovery
```

The committed cases/protocol fix six recovery-on trials (three known sessions,
DQN and the same highest-allowed rule) and two recovery-off hard controls.
Every trial has a 240-second wall budget and an independent finite trace clock.
These are previously observed development cases, not new generalization evidence.

The browser module `request-recovery.mjs` reads each media response with a
ReadableStream. A watchdog can check progress even when `reader.read()` is
blocked. It ports the Python controller's buffer-risk/no-progress rules: wait
at least 2 seconds before early abandonment, do not abandon a nearly finished
body, and estimate a lower-quality restart with nominal size. It does not see
future trace service or unrequested exact sizes.

Failures restart the same segment at the lowest representation, with at most
four retries. Initialization and media requests each have a 30-second deadline.
Each failed attempt consumes the retry budget, including initialization failures.
Timeout, network and no-progress failures wait 2 seconds; buffer-risk cancellation
does not. Playback abort stops recovery. HTTP 410/404 and other permanent media
errors do not repeat indefinitely. Only a complete body is appended to MSE.
Failed chunks are released before retrying; HTTP Range resumption is not used.

The server's cancellation event interrupts RTT/body pacing. The browser waits for
an acknowledgement that the session lock is free, then starts the next request.
This prevents an abandoned long-running response from blocking retries with 409.
The cancellation wait is bounded; failure to acknowledge ends the recovery.

Capture logs keep original policy action, effective appended action, request IDs,
every attempt, byte counts, cause and elapsed time. The next observation uses
the effective bitrate and includes received media bytes from failed attempts in
measured goodput. Completed initialization payloads are recorded separately;
failed initialization byte receipt is not measured. Server-sent bytes and client-
received bytes remain distinct. Partial-run playable backlog is not called waste
from a failed body or compared with full-video stall totals.

The UI shows first-download progress and logs `startup_target_exceeded` after
3 seconds without playback. It keeps the original clock running. This makes the
failure visible; it does not claim to make the first segment arrive sooner.

For the two slow-start sessions, the smallest first segment is 181,801 bytes.
Recorded service provides only about 8,038 and 8,912 bytes in the first 3 seconds.
Even with zero RTT and no initialization traffic, minimum first-segment arrival
is 29.18 and 8.13 seconds. The 3-second startup target is infeasible for the
current whole-segment media on these sessions. A smaller independently playable
startup unit/representation needs actual encoded data and decoder validation;
changing the model or hiding the wait is not a solution.

Tests cover Python/JS decision parity, partial-body discard, stalled-read
watchdogs, retry exhaustion, and real HTTP cancellation during zero service with
successful next-request delivery. The HTTP test needs loopback permission.
Browser trials additionally check actual decoded frames, segment sizes, effective
actions, received-byte accounting, retry bounds and matching native event times.

See [native results](../runs/native-recovery/report.md),
[attempt/capture audit](../runs/native-recovery/summary.json), and
[offline startup bounds](../runs/native-recovery/startup-analysis.json).
