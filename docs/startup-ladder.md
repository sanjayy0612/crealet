# Real startup media and matched learning experiment

This completes the next desktop engineering/training screen. It does not prove
production readiness or a neural advantage. Defaults and deployed weights remain
unchanged because the fixed training runs did not outperform the safety rule on
the unused-session tests.

The original lowest first fragment was 181,801 bytes. The derived ladder adds
an 8 kbps, 96x54 survival representation: its first fragment is 5,478 bytes plus
792 bytes of initialization. All seven representations are real H.264 encodes
with closed four-second GOPs. Original source timestamps contain irregularities;
the pipeline normalizes decoded-frame PTS and measures each generated fragment's
sample duration and bytes. Every full generated stream passed strict decoding.
The original cache, datasets and checkpoints are preserved.

This is a major quality tradeoff. The low rung is deliberately blurry and may
remain selected for a whole short clip on a poor connection. Bitrate alone is not
a visual-quality measurement, and no human-quality or VMAF acceptance is claimed.

The first content is the existing Envivio source, not a new video. The second is
the [Sintel trailer](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4),
copyright Blender Foundation / Durian Open Movie Team,
[CC BY 3.0](https://durian.blender.org/about/). It is re-encoded video-only, resized
and normalized from 24 to 25 fps. Its generated duration is 52.2 seconds. Sintel
is used only for content-transfer testing, not training or checkpoint selection.

Install optional media tooling, verify/download the existing Envivio cache if
needed, then build the derived media:

```sh
.venv/bin/python -m pip install -r requirements-media.txt
.venv/bin/python -m scripts.download_player_media
.venv/bin/python -m scripts.prepare_startup_ladder
curl --fail --location --output artifacts/sintel-trailer-480p.mp4 https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4
.venv/bin/python -m scripts.prepare_startup_ladder --source artifacts/sintel-trailer-480p.mp4 --input-fps 24 --output artifacts/sintel-ladder-v1 --data-dir data/sintel-ladder --source-url https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 --label 'Sintel Trailer (Blender Foundation / Durian Open Movie Team)'
.venv/bin/python -m scripts.evaluate_startup_ladder
.venv/bin/python -m scripts.prepare_transfer_sessions
```

Encoding recipes, source/encoder hashes, exact timings and strict-decode results
are tracked under `data/startup-ladder/` and `data/sintel-ladder/`. Large encoded
files remain under ignored `artifacts/`. Byte-identical rebuilding requires the
recorded FFmpeg binary/platform; a different encoder may produce different sizes.
Do not silently reuse the recorded media fingerprint if that happens. The tracked
numeric datasets/checkpoints permit simulator evaluation without rebuilding media.

For the frozen-model native screen, start the original checkpoint with an
explicit new-media opt-in. This records that the checkpoint was not trained on
this ladder. Default media matching remains strict.

```sh
.venv/bin/python -m abr.player_server --port 8861 --replay-dataset data/startup-ladder/development.json --media-dir artifacts/startup-ladder-v1 --experimental-max-buffer-s 60 --enable-request-recovery --allow-experimental-media
node scripts/capture_policy_player.mjs --port=8861 --cases=runs/startup-ladder/cases.json --memory --output=runs/startup-ladder/native
.venv/bin/python -m scripts.audit_startup_ladder
```

All six cases passed. The known slow-start cases started in 2.42–2.80 s with zero
stalls; the hard control also completed without stalls. For the second content:

```sh
.venv/bin/python -m abr.player_server --port 8862 --replay-dataset data/sintel-ladder/development.json --media-dir artifacts/sintel-ladder-v1 --experimental-max-buffer-s 60 --enable-request-recovery --allow-experimental-media
node scripts/capture_policy_player.mjs --port=8862 --cases=runs/startup-ladder/sintel-cases.json --memory --output=runs/startup-ladder/sintel-native
```

Both Sintel cases completed, but stayed at 8 kbps for the entire clip. This is
evidence of playable low-bandwidth fallback, not acceptable viewing quality.

The conditional matched training gate passed before the three fixed runs:

```sh
.venv/bin/python -m abr.startup_training --seed 7
.venv/bin/python -m abr.startup_training --seed 11
.venv/bin/python -m abr.startup_training --seed 13
.venv/bin/python -m scripts.evaluate_startup_transfer
.venv/bin/python -m scripts.report_startup_work
```

Each seed uses 120 episodes, identical data/configuration/recovery/filter and
update budgets. Validation selects an eligible checkpoint within each seed. All
three selected hashes are saved before scoring four unused sessions on both
videos. Two seeds beat validation reward, but every trained seed had lower
transfer reward than the rule. These results do not justify model promotion.

All transfer episodes finished without stalls. One new session still takes
about 10.35 s to start under the new assets; it provides only about 1.1 KB by
3 seconds. That failure is retained, and the startup limit is not relaxed.

New checkpoints record required request recovery. The player rejects loading
them without that configuration. To inspect a selected experimental checkpoint,
use its matching training dataset and media cache explicitly; it does not replace
the original default checkpoint:

```sh
.venv/bin/python -m abr.player_server --checkpoint runs/startup-ladder/seed-7/accepted.npz --dataset data/startup-ladder/training.json --media-dir artifacts/startup-ladder-v1 --replay-dataset data/startup-ladder/development.json --enable-request-recovery
```

Remaining limits are concrete: desktop Chrome only, no specified device-memory
budget, no accepted minimum visual quality, one same-route recording family, a
short second video, and no learned-policy advantage. Mobile/production approval
requires an explicit target and those acceptance criteria. See
[measured results](../runs/startup-ladder/report.md) and
[frozen selection/transfer evidence](../runs/startup-ladder/transfer.json).
