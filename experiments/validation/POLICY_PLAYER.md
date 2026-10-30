# Local policy playback

`policy-player.html` is a finite VOD player built on the browser's native Media
Source Extensions API. It requests the original Envivio fragments and appends
them for real H.264 decoding. It is separate from the existing fixed-lowest
calibration page, whose contract is unchanged. This is a local research player,
not a production dash.js integration or a complete DASH client.

```bash
PYTHONPATH=. .venv/bin/python scripts/download_player_media.py
PYTHONPATH=. .venv/bin/python -m abr.player_server
```

Open `http://127.0.0.1:8787`. Choose a policy and a connection and start playback.
Reload to start a new session. Save measurements after playback. Video is muted
and has no audio track. Seeking and user pauses are excluded from the experiment.

Bounded native recovery is now available with `--enable-request-recovery`.
It streams progress, cancels slow bodies, retries the same segment at lowest
quality, and appends only complete responses. Recovery is opt-in; weights and
the default cap remain unchanged. See [native recovery and measured startup
limits](../../docs/native-recovery.md) for the frozen 60-second experiment.

The default checkpoint is the current-source baseline reproduction under
`runs/learning-ablation/baseline-seed-7/`, whose weights exactly match the
accepted `runs/dqn-safe/` controller. Its recorded
observed-throughput filter applies to DQN choices and is identified in the page
and capture metadata. For the latest two-case verification, use `--port 8788`
on the server, then run
`node scripts/capture_policy_player.mjs --port=8788 --output=runs/player-safe --policies=dqn`.
See `runs/dqn-safe/training_report.md` for the matching non-neural baseline.

The server loads the frozen validation-accepted checkpoint once, checks the
dataset fingerprint, and binds only to loopback. The browser sends its measured
buffer, previous bitrate, remaining fraction, and eight download measurements.
The server builds nominal-size candidates using the same input contract as the
trainer. The model receives no future trace information. A checkpoint's recorded
action filter constrains DQN choices during playback. The throughput baseline uses the same harmonic
mean and 0.85 safety factor as `abr.benchmark`.

The actual media payloads stay in the ignored `artifacts/player-media/` cache.
The downloader checks every fragment's byte size against the training dataset
and verifies the Git blob hash against a pinned upstream commit. The cache is
about 275 MB; it is not committed.

Each player session has an independent finite trace clock. The server paces
16 KiB HTTP body chunks by integrating that trace with `abr.transport.Replay`.
It charges the assumed RTT for initialization and media requests. It never
repeats a trace. This creates controlled application throughput, not packet,
TCP, loss, or radio emulation. Initialization, native decoding, browser scheduling,
and representation switching create timing differences from the simulator.

For repeatable real-time browser measurements, with Google Chrome installed:

```bash
node scripts/capture_policy_player.mjs --smoke
node scripts/capture_policy_player.mjs
```

The runner starts a separate headless Chrome instance and profile for each case
and uses the Chrome DevTools Protocol. Each page must remain visible: Chromium
can defer MediaSource initialization in a hidden tab. The runner checks this.
Independent sessions run concurrently without accelerating the media clock.
The decode check verifies actual media progress and decoded frames. The full
run compares DQN and throughput on one held-out measured ferry connection and
one held-out controlled outage. A highest-bitrate reference is also run on the
outage case. Every case has the same declared 210 second wall-time budget for
the 193.68 second video; exceeding it stops playback and marks the run incomplete.
All final captures, including incomplete playback, are retained. An initial
multi-tab harness failed to initialize hidden pages; its diagnostic artifacts
are excluded from the corrected comparison.
The runner exits nonzero if DQN or the throughput reference fails to complete;
it still saves every capture and the summary. A reference timeout is an outcome
to inspect, not a complete-video result. Run
`PYTHONPATH=. .venv/bin/python scripts/audit_player_captures.py` to check raw
capture integrity and reconstruct stalls independently of policy acceptance.

Captures under `runs/player-hard/` include native `playing`, `waiting` and
`ended` events, 100 ms buffer samples, per-segment choices and inference time,
played duration, decoded/dropped frames, and server request/decision audits.
Startup is the wall time to first `playing`; rebuffering counts `waiting` after
playback starts. Mean played bitrate weights selected representations by the
media duration actually played. An incomplete run is explicitly marked and
must not be compared as if it played the full video.

This experiment uses one silent video, old mobile traces, and a local headless
browser. It does not establish production playback quality or broad network
generalization. See the training and playback reports for observed results.
