# Frozen 60-second cap validation

The checkpoint remains unchanged. This experiment compares only the 30/60-second
prefetch cap, with startup/resume thresholds fixed at the trained 1/1 seconds.
The protocol names four additional sessions before downloading or inspecting
their values. A fifth, already-inspected hard case is a labelled control.
Policies are frozen DQN, the same highest-allowed filter rule, and lowest quality.

The simulator records full-playback completion, startup, stalls, received bytes,
discarded body bytes, and an estimated compressed-media backlog. That estimate
counts queued encoded segments; it is not measured browser RAM or decoder memory.
The meter does not allocate synthetic media or change playback behavior.

The native comparison runs DQN and highest-allowed at both caps on the first new
session and the known hard control. Each case uses an isolated headless Chrome
instance and its own trace clock. Every two seconds the harness sums resident
memory (RSS) over that Chrome process tree and samples JS heap separately. RSS
can count shared pages more than once and includes the browser, runtime and codecs.
It excludes the Python inference server. Peaks are sampled, not guaranteed maxima.
No device memory budget was supplied, so a measured number is not a deployment gate.

The native player has no recovery. Request failures remain incomplete runs, even
when buffered video could have continued playing. The audit records server bytes
sent, discarded/unappended response bytes, and a separate estimate of appended
but unplayed media. Receipt of failed-response bytes by the browser is unverified.

Explicit experimental server flags keep training identity separate from replay
identity. The default dataset fingerprint check is retained. Replay media must
exactly match the training bitrate ladder, durations and segment-size matrix.
The checkpoint's trained configuration remains recorded alongside the cap override.
No default configuration or checkpoint is rewritten.

```sh
.venv/bin/python -m scripts.evaluate_prefetch --prepare
.venv/bin/python -m scripts.evaluate_prefetch
```

For the native comparison, start these in separate terminals:

```sh
.venv/bin/python -m abr.player_server --port 8840 --replay-dataset data/prefetch-sessions/dataset.json --experimental-max-buffer-s 30
.venv/bin/python -m abr.player_server --port 8841 --replay-dataset data/prefetch-sessions/dataset.json --experimental-max-buffer-s 60
```

With Chrome installed and the verified media cache present:

```sh
node scripts/capture_policy_player.mjs --port=8840 --cases=runs/prefetch-validation/native-cases.json --memory --output=runs/prefetch-validation/native
.venv/bin/python -m scripts.report_prefetch
```

The capture command saves every outcome and exits nonzero if a required playback
does not complete. This is expected for a failed playback, rather than a missing
capture. `report_prefetch` separately audits identities, bytes, choices, visibility,
frame decoding, event-derived stalls, memory samples and transport completion.

Two original stall counters differed from their event logs by about 40 ms because
the handler read the clock more than once. The actual handler code reproduced
this in a failing Node test. It now shares one timestamp per event, and finish
uses a shared end time. Raw captures remain untouched; the report reconstructs
timing from logged playing/waiting events and exposes both values. It does not
widen the consistency tolerance. Source hashes at capture time are preserved
in `native/capture-source.json`; a short post-fix native decode check is saved
under `post-fix-smoke/`.

The conditional training plan is fixed at seeds 7/11/13, 120 episodes each, and the
same safety rule. It runs only after all proposed-cap simulator/native playback
gates pass; memory suitability still needs a target-device budget. A frozen
30-second checkpoint evaluated at 60 seconds is a screening experiment, not a
fair final comparison of a controller trained for 60 seconds.

The 30 simulator episodes completed, with no stalls on the four new sessions at
either cap. Two new sessions had startup delays of 29.18 s and 8.14 s at both
caps, so the simulator gate failed. Six of eight native trials completed. At
60 s, DQN completed the hard control without stalls, while the rule trial timed
out; their independently measured request timings differ, so this is not proof
of a learned-policy advantage. The first new native session had about 29–30 s
startup. The native gate also failed. Training was not started.

Browser process-tree sampled peaks ranged from roughly 914 to 1070 MiB. These
include substantial browser/runtime overhead; cap effects cannot be isolated
from a single trial per case. The simulator's encoded backlog estimates are
reported separately. A target-device budget and native recovery validation are
still needed before considering a new deployment setting.

See [the measured report](../runs/prefetch-validation/report.md),
[protocol](../runs/prefetch-validation/protocol.json), and
[audited summary](../runs/prefetch-validation/summary.json).
