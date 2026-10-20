# ABR Decision Engine

Independent research project for small adaptive bitrate (ABR) decision models.
The objective is to reduce measured inference cost while preserving streaming
quality and controlling stalls under unseen network, video, ladder and live-timing
conditions. This is a research hypothesis, not a demonstrated result.

## Current status

- Finite VOD and low-latency live application-level environment.
- Measured trace imports, reproducible diagnostic replay and player-capture tools.
- Draft evaluation protocol; no model training or novel performance result yet.
- Real-player validation still pending packaged media and a controlled network.

This repository contains no legacy PPO model, Flask inference server, historical
checkpoints or legacy training scripts. All future work belongs here.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m abr.benchmark --help
python -m abr.validate_player --help
```

Runtime dependencies are NumPy and Gymnasium. Tests require pytest; JavaScript
recorder checks also use Node.js when available. No Torch or Stable-Baselines3 is
required for this environment foundation. Choose model dependencies when model
implementation starts.

```python
from abr import StreamingEnv, StreamConfig, QoEWeights
```

## Layout

- `abr/env.py`, `transport.py`, `media.py`, `playback.py`: simulation components.
- `abr/trace_import.py`: auditable trace conversions.
- `abr/benchmark.py`: diagnostic baseline evaluation, with no training.
- `abr/validate_player.py`: simulator/player timing comparison.
- `experiments/configs/evaluation_v2.json`: draft evaluation protocol.
- `experiments/validation/`: independent browser capture page.
- `tests/`: environment, import, and telemetry checks.

The environment and observation contracts retain version 2. Renaming the files
changes source hashes; regenerate reports and cases in this repository. Dataset
schemas 1 and 2 remain readable. Raw data, media and generated reports are not
bundled. `python -m abr.benchmark` is the replacement for the previous
`python -m abr.benchmark_v2` command.

## Origin

Extracted from the author's `DASH_RL` research work at commit
`970c67ec015cfc27968b78bb09db243fb95da08a`, from branch
`abr/trace-driven-rebuild`. This is a fresh repository, not a worktree or submodule.
The original checkout is left intact. Only the V2 environment and evaluation
foundation were brought over. The software design references below are citations,
not statements that their code was copied or their results reproduced.

## Environment design and limits

### What the literature changes in our design

| Primary source | Environment pattern or finding | V2 design response |
| --- | --- | --- |
| [Pensieve, SIGCOMM 2017](https://web.mit.edu/pensieve/) and [Sabre implementation](https://github.com/UMass-LIDS/sabre) | Fast replay combines network traces, segment sizes, and player state. Sabre accounts for played quality. | Keep replay fast, but separate transport, media timing, and playback. Use finite traces and count quality when played. |
| [DASH-IF low-latency design](https://dashif.org/dash.js/pages/usage/low-latency.html) | CMAF parts arrive inside one open segment request. Producer idle time distorts whole-request throughput estimates. Playback rate helps control live delay. | Require explicit part tables, wait for source availability, charge one request RTT, and separate transfer, producer wait, and buffer wait. |
| [Fugu, NSDI 2020](https://www.usenix.org/conference/nsdi20/presentation/yan) | Real deployment rankings can differ from simulation. Fugu uses deployment data and transmission-time prediction. | Keep simulation results separate from browser/network validation. V2 does not implement Fugu. |
| [CausalSim, NSDI 2023](https://www.usenix.org/conference/nsdi23/presentation/alomar) | Traces collected under one algorithm can bias replay of another algorithm. | Preserve collection source and session groups. Mark observed goodput as observed goodput. This metadata does not remove causal bias. |
| [CellReplay, NSDI 2025](https://www.usenix.org/conference/nsdi25/presentation/sentosa) | Cellular behavior depends on workload; accurate packet replay needs more than a throughput series. | Keep a distinct future Linux emulation validation stage. V2 does not recreate CellReplay's packet model. |
| [SIGCOMM 2021 5G artifact](https://github.com/SIGCOMM21-5G/artifact/tree/main/Video-Streaming) | Published ABR experiments use measured 4G/5G throughput traces and a browser testbed. | Import timestamp/Mbps traces with explicit interval alignment. Do not treat old measured traces as coverage of all current networks. |

The new implementation is original project code. These are design references,
not claims that we reproduced the papers or matched their results.

### Components and execution

```text
Trace + collection metadata -> transport.Replay ----+
Media segment/part tables --> source availability -+-> env.StreamingEnv
                                                    |       |
                                              playback.Player
                                                    |
                                     observations, reward, raw metrics
```

- `abr/transport.py`: integrates interval goodput, including outages. A request
  has one sampled RTT cost. This is a request-delay approximation, without
  separate forward/reverse propagation of each part. RTT can vary by interval. Trace exhaustion and
  request timeout stop an incomplete episode. Traces never repeat silently.
- `abr/media.py`: segment sizes plus optional per-part durations and sizes for
  every representation. Parts must sum to their segment duration and bytes.
  Live mode rejects missing part tables. It never silently divides segment
  sizes into equal parts. Synthetic tables must be labeled as synthetic.
- `abr/playback.py`: playable-unit queue, separate startup/resume thresholds,
  stall duration and event count, actual played quality, and final buffer drain.
- `abr/env.py`: one action selects one segment representation. VOD makes all
  segments available at reset. Live releases each part at
  `media_end + encoder_delay - source_age_at_reset`. No part is playable until
  its transfer completes. A finite event stops producing at the video's end.
- Live rate control is an optional bounded proportional rule, with a default
  range 0.95–1.05. It uses latency error and avoids speeding up a depleted
  buffer. This is a simple test controller, not a reproduction of LoL+.
- Buffer waits model application backpressure before the next playable unit.
  They are not a TCP receive-window simulation. The buffer must hold one unit
  plus the larger startup/resume threshold, preventing a paused-player deadlock.

Default observations contain only past measurements and public candidate
metadata. They do not contain future network values or true future live sizes.
`history[H,5]` columns are body goodput in Mbps, request time / 10 seconds,
producer wait / 10 seconds, request RTT / 10 seconds, and a valid-sample flag.
`state[5]` is buffer seconds, live latency seconds (zero for VOD), playback rate,
remaining segment fraction, and previous selected Mbps. `candidates[K,3]` is
nominal Mbps, estimated segment MB, and segment duration seconds. Default size
estimates are nominal bitrate times duration. `known_vod_sizes=True` permits
actual VOD sizes only when those sizes are available to the deployed controller.

Body goodput excludes producer idle time, RTT, and buffer waits. Network outage
time remains in the body duration. This is simulator instrumentation, not a
validated browser throughput estimator. A real player needs a separately tested
estimator from receive timestamps.

### Metrics and reward

Reports retain startup delay, stall time/count/ratio, played media duration,
played bitrate, switching, received bytes (including incomplete transfers),
peak buffer, live latency, and playback-rate deviation. Mean live latency is
weighted by wall time after startup, including stalls. The live edge is source
media time; camera capture, display, and decode delays are not measured. No
percentile latency is reported from sparse segment endpoints.

The reward is played-duration-weighted log bitrate minus separate startup,
stall, played-quality switching, excess live-latency, and rate-deviation terms.
Weights are explicit `QoEWeights`. Log bitrate is a proxy, not VMAF or measured
human opinion. Raw metrics remain separate so a reward cannot hide a tradeoff.
Step logs separate request time, total elapsed time, and playback-tail time.
The final step includes the playback tail. Reward timing differs from V1 and
can attribute playback of an earlier selected segment to a later action.

Decisions have unequal elapsed times. A later RL design must explicitly choose
per-decision discounting or elapsed-time discounting (a semi-Markov model).
V2 does not make that training choice yet.

### Measured inputs and reproducible diagnostics

`Dataset.save()` writes schema 2; `Dataset.load()` also accepts schema 1. Trace
metadata adds `source_group`, `source`, `semantics`, and optional
`rtt_series_s`. Use a shared group for every window of a collection session;
if session identity is unknown, group conservatively by collection. This check
cannot find correlation when group labels are missing or wrong. Schema 2 and
changed source hashes invalidate earlier experiment provenance; do not silently
reuse old checkpoints.

`abr.trace_import` supports:

- `load_ghent4g`: the [original six-column interval-byte logs](https://users.ugent.be/~jvdrhoof/dataset-4g/).
  Rate in kbps = bytes times 8 divided by interval milliseconds. Zero bytes
  remain outages. These 2015–2016 LTE logs are useful import checks, not a
  modern-network benchmark. RTT must be an explicit assumption.
- `load_timestamp_mbps`: timestamp-seconds/Mbps pairs, including the Lumos5G
  artifact format. The caller must specify `alignment="start"` or `"end"` from
  the collection/testbed semantics. The importer drops the unbounded endpoint;
  it does not invent the duration of the last sample.
- `load_interval_csv`: exact header `duration_s,goodput_kbps,rtt_s` for other
  audited datasets or controlled experiments.

All importers record the source and raw file checksum. They do not download
anything. Keep raw data and conversion settings with experiment outputs. Do
not convert Puffer per-request streaming logs or CellReplay's packet-delay
artifacts into capacity traces without a separately validated method.

Media JSON uses schema 1 with `video` (the same three video arrays as Dataset),
`part_durations_s` (`[segment][part]`), `part_sizes_bytes`
(`[segment][part][representation]`), and `provenance`. Use `Media.save/load` to
write/read this schema. Part metadata should come from the packaged media.
The loader validates timing and byte conservation; it does not verify codec,
random-access, audio/video alignment, or actual CMAF box structure.

```bash
python -m pytest -q
python -m abr.benchmark --dataset data/audited.json --mode vod --baseline throughput --output runs/v2-vod.json
python -m abr.benchmark --dataset data/audited.json --media data/packaged-media.json --mode live --baseline lowest --output runs/v2-live.json
```

Use `--config path/to/config.json` for `StreamConfig` overrides. Diagnostic
baselines are lowest, highest, and a harmonic-throughput heuristic. They are
not claimed to be state-of-the-art controllers. Reports contain raw steps,
completion/truncation reason, source/config/data/media hashes, and versions.
Reports do not compare partial and complete episodes as if they were equal.

V2 is a tested application-level foundation. It does not model TCP/QUIC,
congestion control, packet loss, radio scheduling, CDN behavior, initialization
fetches, audio, decode,
request abandonment/retry, live seeking, or an unbounded sliding live window.
Its live mode models a finite recorded event beginning at media time zero;
it does not yet implement joining an ongoing stream at an arbitrary segment.
No real-browser equivalence or decision-engine improvement is established.
Before model selection, validate packaged media and playback metrics against
an actual DASH player under controlled network conditions, then include
multiple independently collected network and video groups in the benchmark.


## Evaluation protocol and player validation

The research objective and experiment rules are in
`experiments/configs/evaluation_v2.json`. They specify VOD and finite live tasks,
five training seeds, source-group splits, held-out transfer cases, required
baselines, raw playback/quality/compute metrics, and publication limits. The
protocol is **draft**: target hardware, measured VMAF tables, independently
collected benchmark groups, baseline implementations, and primary acceptance
margins are still missing. Do not call it preregistered or use final test results
to choose these values. The draft's playback tolerances are initial calibration
thresholds, not proven equivalence bounds.

Use the independent capture page at `experiments/validation/player.html` for
initial timing checks. It pins dash.js 4.7.4, holds video quality at index zero,
disables live catchup, records native playback events, and never calls an
inference server or loops playback. It exports
startup/stall events and 100ms buffer, media-clock, rate, and live-latency samples.
Use this dedicated capture page for the comparison.

1. Package a finite, video-only DASH asset and extract its actual segment/part
   durations and byte sizes into Dataset/Media JSON. Confirm representation index
   zero matches the simulator's lowest bitrate. Audio, codecs, container overhead,
   random-access points, and CMAF packaging require independent inspection.
2. Configure the external network testbed and record the configuration, start
   alignment, trace conversion convention, and run ID. This page does not impose
   network conditions. Browser throttling is not automatically equivalent to
   interval-goodput replay. Start with constant-rate VOD, then rate steps and
   outages; verify finite live availability separately.
3. Run `abr.benchmark` with `--baseline lowest --split validation`. For live
   timing checks, pass a config with `catchup: false`; the browser's catchup
   controller differs from V2's proportional rule. Do not use the held-out final
   test set to calibrate the simulator.
4. Export a case from a complete diagnostic report:

   ```bash
   python -m abr.validate_player prepare --report runs/v2-lowest.json --episode 0 --output runs/player-case.json
   python -m http.server 8080 --bind 127.0.0.1
   ```

5. Open `http://127.0.0.1:8080/experiments/validation/player.html`. Load the case
   JSON, enter the packaged manifest URL and external network evidence reference,
   and start capture. Keep the tab visible. The page uses the pinned CDN script;
   it needs internet access for that script and CORS permission for remote media.
   For reproducible offline experiments, host that exact library locally and
   record its checksum. Capture automatically stops at playback end; manual stop
   exports an incomplete run. The prepared case contains expected measurements,
   not the media or network emulator.
6. Compare the exported raw capture:

   ```bash
   python -m abr.validate_player compare --case runs/player-case.json --capture runs/player-capture.json --output runs/player-comparison.json
   ```

The comparison rejects different case identities, missing network evidence,
pauses, seeks, incomplete playback, non-finite/out-of-order timestamps, and missing
or sparse live-latency samples. It reports absolute errors for startup, stalls,
played duration, wall time, stall count, and live latency. Exit 0 means within the
configured tolerances; exit 1 means a numerical mismatch; exit 2 means invalid
input. Keep mismatches and explain them; do not widen tolerances using final
validation/test outcomes.

Native `waiting` after first `playing` starts a stall; a later `playing` ends it.
Repeated waiting events count as one stall. Initial waiting belongs to startup.
The browser's event semantics and timing precision differ from an ideal simulator.
Browser startup includes manifest, initialization, and decode work, which V2 does
not model. Live latency is integrated over sampled wall time after startup,
including stalls, and uses the player's estimated source clock. Missing latency
stays missing. The initial live comparison requires a finite event starting at
media time zero: normal late joining or automatic live seeks are not supported by
V2 and must not be hidden by subtracting an arbitrary playback offset.

A matching JSON identity is a declared experiment condition, not independent proof
that the browser used the claimed bytes or network. Verify the packaging,
representation mapping, and network artifacts separately. One passing comparison
does not establish environment equivalence. The tool always keeps that broader
claim false. Its recorder tests use analytical media-event fixtures; they are not
actual DASH playback validation. No real-browser playback result or V2 trained
model is claimed in this update.
