# VOD request recovery

The simulator can now restart a stalled segment request without ending the video
at the first timeout. Recovery is optional and separate from bitrate selection.
Existing environments keep their original behavior unless explicitly enabled.
The browser player has not received this change.

```python
from abr.env import StreamingEnv
from abr.recovery import RecoveryConfig

env = StreamingEnv(dataset, split='validation', recovery=RecoveryConfig())
```

The controller observes the active request's downloaded bytes, elapsed time,
Content-Length and remaining playback buffer. It never uses future bandwidth.
It estimates the cost of a lowest-quality restart from the nominal bitrate and
segment duration, rather than reading an unrequested representation's exact size.
It avoids discarding a nearly finished body. When finishing a large request risks
emptying the buffer, it can cancel the body and restart the same segment at the
lowest quality. It can also retry the lowest quality after a timeout.

There are at most four retries per segment. A retry has its own request deadline
and incurs another RTT. Timeout and no-progress retries wait two seconds. All
waiting advances playback and can cause stalls. An abandoned body is discarded
in full; received bytes cannot be reused without an HTTP Range protocol, which
this experiment does not implement. Playback quality and the next observation
reflect the representation actually appended, not the policy's original choice.

The history's measured goodput includes all received body bytes and body-transfer
time, including failed attempts. Request duration includes retries and backoff.
Reports retain the original policy action, effective action, each retry's outcome,
discarded bytes and retry count. If a request fails before a body starts, the
existing request failure behavior applies. Recovery supports VOD only.

## Measured outcome

Run `.venv/bin/python -m scripts.evaluate_request_recovery` from the repo root.
It saves fixed settings before replay, then compares no recovery, timeout-only
retries, and progress-based recovery across four fixed policies and eight sessions
(96 total episodes). The checkpoint is frozen; there is no new training.

On the two previously failed sessions, DQN with progress recovery completed both
videos. It had previously timed out on both. Full-playback stalls were 21.58 s
and 9.93 s, so neither passed the predeclared 8 s stall limit. Against timeout-only
retries, progress recovery reduced stalls on the first session from 25.42 s to
21.58 s and reduced discarded bytes; the second session was unchanged. DQN made
the same choices as the highest-allowed rule. Historical DQN control sessions
still completed with unchanged stall totals.

These sessions were already inspected and are failure-development cases. They
are not untouched test evidence. Lower partial-playback stall totals from the
original failed runs do not describe a full video. The replay uses measured
application goodput, not a causal radio or packet model.

The next useful experiment is a startup/resume-buffer tradeoff on separate
network sessions, keeping explicit startup and stall limits. These results do not
support promoting the model or claiming that further training alone fixes outages.

See [the complete comparison](../runs/request-recovery/report.md),
[fixed protocol](../runs/request-recovery/protocol.json), and
[machine-readable results](../runs/request-recovery/results.json).

Verification: the full suite passed 129 tests during this change, including
analytical outage, early downgrade, byte accounting, retry limits, trace-end,
RTT/backoff, and zero-retry parity checks. The two existing Gymnasium warnings
about infinite observation-space bounds remain.
