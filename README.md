# ABR Decision Engine

This project aims to build a **small, streaming-specific decision model**. Like
the decision interface used by [Laya](https://github.com/NandhaKishorM/laya),
it takes a state and a set of allowed choices and returns one choice in a single
forward pass. Here the state is numeric network and player history, and the
choices are the available video representations. The model is intended to run
at each adaptive bitrate (ABR) decision point in a video player. It is not a
port of Laya's language encoder or its weights.

The research question is whether this specialized model can make good bitrate
decisions with low measured inference cost across unseen network, video, ladder,
and live-timing conditions. Playback quality, stalls, and live delay must be
measured separately from model size and speed. No such result has been shown yet.

## Current status

- Finite VOD and low-latency live application-level environment.
- Measured trace imports, reproducible diagnostic replay and player-capture tools.
- Dense, GRU, and ABR-CandidateTransformer v1 architectures implemented; weights are untrained.
- No training loop, trained streaming policy, calibrated decision probabilities,
  or real-player policy integration yet.
- Draft evaluation protocol; no model training or novel performance result yet.
- Real-player validation still pending packaged media and a controlled network.

This repository contains no legacy PPO model, Flask inference server, historical
checkpoints or legacy training scripts. All future work belongs here.

## Project goal and prior work

The intended product is a bounded decision engine: **recent measurements +
player state + candidate representations -> one representation**. The same
interface can score a different number of candidates without changing the
output layer. A streaming player would gather the inputs before each segment
request, run the model, then use the selected representation. A single forward
pass means the model does not generate text or search through outputs. It does
not imply accurate choices, low latency on every device, or calibrated
probabilities; those need training and measurement.

Laya is useful as an interface reference for typed, single-pass choices, but
its pretrained text encoder is unnecessary for the current numeric observations.
[Pensieve](https://web.mit.edu/pensieve/content/pensieve-sigcomm17.pdf) already
trained a neural ABR policy, so a learned bitrate selector alone is not a novel
claim. The proposed contribution must be established by a specific, measured
advantage over strong streaming baselines. The [research note](docs/research/decision_model_and_training_environment.md)
compares the relevant systems and explains the training-data limits.

The current dense and GRU models are small candidate scorers, not miniature
copies of Laya or Pensieve. Their outputs are raw, untrained scores. Training
must define what those scores mean (for example, policy logits or estimated
action value). Applying softmax would not by itself make them reliable
probabilities that a choice is safe or optimal.

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
required for this environment foundation. Install `requirements-model.txt` to
construct and test the optional PyTorch models.

```python
from abr import StreamingEnv, StreamConfig, QoEWeights
```

## Layout

- `abr/env.py`, `transport.py`, `media.py`, `playback.py`: simulation components.
- `abr/trace_import.py`: auditable trace conversions.
- `abr/benchmark.py`: diagnostic baseline evaluation, with no training.
- `abr/validate_player.py`: simulator/player timing comparison.
- `abr/models/`: small candidate scorers, input adapter, checkpoints and inference.
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

### Is this environment suitable for training?

**Yes, for an initial segment-level policy in a finite, trace-driven task.**
`reset()` returns the first observation; `step(representation_index)` advances
the network, source, and player clocks and returns the next observation, reward,
termination flags, and measurements. Repeated calls create an action-dependent
sequence of decisions. The replay runs in simulated time and can run faster than
wall time. Live mode uses explicit part *tables* at simulated production
times, but the action is still chosen once per segment. This is online-style
interaction with a simulator, not a feed of current internet conditions.

The environment should not be rewritten merely to make it "real-time." First
train and evaluate a segment-level policy against measured, provenance-tracked
network traces and packaged video metadata. Controlled-capacity traces are the
cleaner replay input when available; observed throughput from another player's
requests may depend on that player's actions and cannot automatically be treated
as the capacity every candidate would have received. Keep training, validation,
and test sessions separate. The included synthetic dataset is an engineering
fixture, not evidence of real-world generalization.

Before trusting learned-policy results, close these specific gaps:

1. Collect or audit trace, RTT, and packaged video/part-size data. Record the
   collection process and split by independent network sessions and source
   videos. Supply a range of ladders and live timing conditions.
2. Compare simulator startup, stalls, timing, and live latency with an actual
   player under controlled network conditions. The current browser page captures
   playback telemetry for this check; it does not export a complete per-request
   training stream or impose network conditions.
3. Choose an explicit training target and algorithm, record transitions with
   action and elapsed time, and retain incomplete/timeout episodes. A simple
   supervised imitation run can check training plumbing, but only learning from
   a teacher does not establish an improvement over that teacher. An RL method
   can optimize the specified QoE reward through simulator interaction, subject
   to simulator bias.
4. Benchmark learned policies against implemented BOLA, robust MPC, a documented
   live controller, and a reproduced learned reference. Report playback metrics
   and batch-one inference cost on fixed hardware. Test on held-out groups.
5. For a claim about deployment, add a separate player adapter that measures
   requests and playback, forms the same observation contract, invokes the
   policy before each request, and logs actions and outcomes. Validate the
   measurement mapping and run controlled player trials before using field data
   for training or claiming real-time performance.

If the intended controller must react *within* a segment to each live part, or
must handle late joins, request cancellation, or continuously arriving media,
the action and episode contracts need a separately versioned extension. The
current segment-boundary environment cannot test those behaviors.

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


## First decision-model architectures (untrained)

The dense/GRU models below remain comparison baselines. The new
[ABR-CandidateTransformer v1](abr/models/README.md) jointly processes numeric
history, player state/configuration, and candidates using two 64-wide transformer
layers with four attention heads. Its shared policy head returns candidate logits;
its separate value head reads the contextualized state token for future
actor-critic training. The default model has **77,826 trainable parameters**.
There is no ABR training loop yet, and neither head has learned streaming behavior.

```bash
python -m abr.models init --kind candidate_transformer --seed 0 --output runs/transformer-initial.npz
python -m abr.models inspect --checkpoint runs/transformer-initial.npz
```

GRU-specific recurrent code lives in `abr/models/gru/`; dense/GRU shared scoring
remains in `abr/models/network.py`. Their public construction APIs and parameter
counts are preserved. Model source hashing now covers nested Python modules;
initialization artifacts from earlier source versions must be regenerated.

The two original baseline models implement the same decision interface. They do
not use Laya weights, language tokenizers, pretrained encoders, or PPO. Both score
the available bitrate candidates in one forward pass. Neither has been trained
on streaming outcomes. A score is not a probability, calibrated confidence, or
an established estimate of future return.

```text
History [8,5] -> dense network OR masked GRUCell -> 48 features --+
Player state [5] + public streaming context [10] -> 32 features -+-> fusion -> 48
Candidate-set mean/min/max [9] --------------------------------+       |
                                                          concatenate each
                                                          candidate [3]
                                                               |
                                                    shared 51 -> 32 -> 1 scorer
                                                               |
                                                    raw scores -> argmax
```

| Component | Dense baseline | GRU variant |
| --- | --- | --- |
| History encoder | Flatten 8x5 -> 48 -> 48, SiLU | GRUCell: 4 inputs -> 48 hidden, mask-controlled updates |
| Player encoder | 15 -> 32, SiLU | Same |
| Fusion | 89 -> 48, SiLU | Same |
| Candidate scorer | 51 -> 32 -> 1, SiLU between linear layers | Same |
| Trainable parameters | 10,849 | 14,305 |

The GRU reads oldest to newest and skips padded history rows. It re-encodes the
finite window for each decision; it does not maintain a hidden state between
calls. This prevents episode leakage and gives both variants the same finite
history. The dense variant sees the validity flag directly. ModelConfig can
change widths and history length, which changes the parameter count.

The public context vector is streaming mode, target live latency, buffer limit,
startup threshold, resume threshold, catchup enabled, minimum playback rate,
maximum playback rate, catchup gain, and latency deadband. In VOD, irrelevant live
fields use fixed neutral values. Context comes from StreamConfig, not future
network data. Current observations still use the environment v2 schema.

Each candidate supplies nominal Mbps, estimated MB, and duration seconds. The
shared scorer supports a variable number of candidates. Mean/min/max pooling is
independent of candidate order and lets scoring depend on the available ladder.
Batching pads missing candidates and masks them out of pooling and action
selection. This interface support does not prove transfer to unseen ladders.

`make_batch` validates observation values and shapes. Scaling uses fixed units
and log1p for nonnegative magnitudes; nothing is fitted on validation/test data.
The low-level tensor forward expects the validated nonnegative feature contract.
The model has no access to traces, true future live sizes or environment internals.
Future training must specify its reward objective, discounting, and data protocol;
raw model outputs are not yet defined as trained action values.

```bash
python -m pip install -r requirements-model.txt
python -m abr.models init --kind dense --seed 0 --output runs/dense-initial.npz
python -m abr.models init --kind gru --seed 0 --output runs/gru-initial.npz
python -m abr.models inspect --checkpoint runs/gru-initial.npz
python -m pytest -q
```

Artifacts use NPZ numerical arrays plus JSON metadata, loaded with pickle disabled.
They record architecture, observation/feature versions, source hash, parameter
count, initialization seed and PyTorch version. Loading rejects incompatible
source or feature contracts and changed fixed scaling constants. This construction
milestone only saves and loads artifacts labeled `training_status: untrained`.
Regenerate initialization artifacts after model-source changes. Model files belong
under ignored `runs/`; do not confuse them with trained checkpoints.

An explicit untrained rollout can check plumbing on training/validation fixtures:

```bash
python -m abr.models rollout --checkpoint runs/gru-initial.npz --dataset data/audited.json --media data/packaged-media.json --mode live --allow-untrained --output runs/gru-integration.json
```

Without `--allow-untrained`, the command refuses to evaluate initial random weights.
The command excludes the final test split. Reports include raw outcomes and the
artifact's untrained status, model-source hash and file checksum. Successful
playback on an easy fixture only checks integration. It does not demonstrate
quality, generalization, lower computation, or a research contribution.

Model tests are skipped when optional PyTorch is absent; install both developer
and model requirements for the full suite. The tests check candidate permutation
and padding behavior, temporal order, absence of persistent hidden state, gradient
flow, serialization, and VOD/live rollouts. The synthetic regression target used
in one gradient test is not an ABR training algorithm. Next work is to select and
implement training and establish measured baselines; real-player validation and
packaged-media quality measurements remain pending.
