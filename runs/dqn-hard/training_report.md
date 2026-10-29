# Hard VOD training and native browser results

## Outcome

All three requested stages were executed: harder inputs, constrained DQN
training/comparison, and a local browser player connection with real playback
measurements. The pipeline works, but the accepted DQN policy is too
conservative. It selects 300 kbps throughout. It has not demonstrated useful
bitrate adaptation or an advantage over a fixed-lowest policy.

## 1. Harder input

The video remains the original Envivio DASH VOD: six nominal bitrates,
300–4,300 kbps, 49 fragments, 193.68 seconds in the manifest, with exact
fragment sizes. Actual browser playback ended at media time 193.76 seconds.

The input adds fourteen original [Norway HSDPA mobile logs](https://datasets.simula.no/hsdpa/)
collected in 2010–2011. Ten bus/car/tram sessions train the model, two train
sessions provide validation, and two ferry sessions are held out for testing.
Complete routes remain isolated across splits. Their mean goodput ranges from
about 660 to 2,583 kbps. Two training profiles, one validation profile, and one
test profile add explicitly synthetic capacity drops, outages, and recovery.
See `../../data/hard_envivio/README.md` for conversion details and assumptions.

The source's received bytes are divided by differences between measured
monotonic timestamps. The first row is discarded; reported interval discrepancies
are counted. No measured bandwidth is scaled, no trace is looped, and no
unmeasured tail is added. RTT is an explicit 80 ms assumption.

## 2. Dueling Double DQN

The single planned run used seed 7, 1,600 episodes, 74,164 environment steps,
18,417 gradient updates, and a reward penalty of 12 per rebuffer second.
Every baseline uses the same environment and reward weights.

The acceptance rule was fixed before training: complete every validation
episode, startup at most 3 seconds, and rebuffer no greater than the smaller
of 8 seconds and the throughput baseline on that validation trace. The limits
were 8.000, 0.463 and 5.827 seconds. The 8 second ceiling was chosen after the
fixed-lowest baseline already stalled for 6.493 seconds on one validation
journey; it is an experimental ceiling, not an ideal viewer experience target.

Only the episode-20 checkpoint met every rule. Later checkpoints picked higher
bitrates but failed completion or stall constraints. That early checkpoint
was frozen before evaluation on the held-out traces. It selected 300 kbps for
every segment on all three validation and all three test traces.

| Simulator split / policy | Completion | Mean stalls | Mean played bitrate | Mean reward |
| --- | ---: | ---: | ---: | ---: |
| Validation / accepted DQN | 3/3 | 2.164 s | 300 kbps | -26.70 |
| Validation / throughput | 3/3 | 6.430 s | 1,031 kbps | 107.09 |
| Held-out test / accepted DQN | 3/3 | 0.000 s | 300 kbps | -0.83 |
| Held-out test / throughput | 3/3 | 1.159 s | 1,039 kbps | 184.86 |

The fixed-lowest baseline exactly matches DQN on these replays. The fixed-highest
baseline failed to complete all three test episodes. Its partial playback
results are retained in `training_metrics.json`; they are not full-video scores.
The throughput rule has a higher reward and higher quality. DQN passes the
chosen safety gate by giving up quality, not by learning better adaptation.

## 3. Native browser connection

The new local MSE player sends actual buffer and download history to the frozen
model and requests the selected original H.264 fragments. It uses nominal-size
candidate estimates, just like training. No safety override, teacher, future
network input, or hidden fallback changes the model's choice. The media cache
contains 300 verified files (274,635,022 bytes) and stays outside Git.

Each case runs in its own visible headless Chrome instance, with an independent
finite HTTP trace pacer. All cases have the same 210 second wall-time budget
and a 30 second request timeout. The pacer is application-level HTTP replay,
not TCP or radio emulation. It also charges initialization requests; the browser
adds native decoding and scheduling costs absent from the simulator.

| Browser case / policy | Complete | Startup | Stalls | Mean played bitrate |
| --- | --- | ---: | ---: | ---: |
| Measured ferry / DQN | Yes | 1.05 s | 0.00 s | 300 kbps |
| Measured ferry / throughput | No: request timeout | 3.91 s | 4.23 s until stop | 1,319 kbps, partial |
| Controlled outage / DQN | Yes | 4.38 s | 0.00 s | 300 kbps |
| Controlled outage / throughput | Yes | 2.05 s | 0.80 s | 1,173 kbps |
| Controlled outage / highest | No: wall-time budget | 8.40 s | 125.68 s until stop | 4,296 kbps, partial |

DQN decoded all 49 fragments and 4,842 frames in each completed browser run.
It selected the lowest bitrate throughout both cases. Its outage startup of
4.38 seconds **missed the three-second startup target**, so simulation acceptance
did not carry over to every browser timing target. The throughput rule's
measured-case segment 48 timed out: only 300,443 of 550,906 bytes arrived before
the 30 second deadline. That run stopped at media time 187.88 seconds.
The highest reference stopped at media time 75.95 seconds when the wall budget
expired. These partial results must not be treated as completed-video metrics.

Raw captures and screenshots are in `../player-hard/`. Native waiting/playing
events sometimes share a timestamp; raw waiting-event counts include those
zero-duration transitions. `verification.json` independently counts positive
stall durations and uses the existing native-event validator for completed
runs. All five captures passed the data-integrity audit: source and checkpoint
identity, visible-page samples, decoded frames, exact received segment bytes,
server/segment action agreement, and stall duration reconstruction.

The initial multi-tab harness deferred MediaSource startup in hidden tabs and
was excluded from comparison. The corrected runner isolates each case and
checks visibility. Its full run exited nonzero because the throughput reference
did not complete; this is a retained playback outcome, not a successful
baseline test. The repository test suite passed: **97 tests**, with two existing
Gymnasium infinite-bound warnings. Native decode smoke and capture audits also
passed. No production-player or broad simulator-equivalence claim is made.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_hard_vod.py
PYTHONPATH=. .venv/bin/python -m abr.models.train_dueling_dqn --dataset data/hard_envivio/dataset.json --episodes 1600 --seed 7 --evaluate-every 20 --rebuffer-penalty 12 --max-rebuffer-s 8 --max-startup-s 3 --output runs/dqn-hard
PYTHONPATH=. .venv/bin/python scripts/download_player_media.py
PYTHONPATH=. .venv/bin/python -m abr.player_server
# In another terminal:
node scripts/capture_policy_player.mjs --smoke
node scripts/capture_policy_player.mjs
PYTHONPATH=. .venv/bin/python scripts/audit_player_captures.py
```

The next model milestone is to raise quality while retaining the stall and
startup bounds. This run does not justify replacing a strong ABR controller
with DQN in a production player. One silent video, historical routes, one seed,
and two native replay scenarios provide only limited evidence. The source-checked
new checkpoint loads with the current model code; older pilot artifacts require
their original recorded source version.
