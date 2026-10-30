# Frozen 30/60-second prefetch validation

Do not retrain or promote: the predeclared playback gates did not pass. Memory measurements are descriptive; no device budget was supplied.

Settings, four new session filenames, native cases and candidate training budgets were saved before downloading/replaying the new traces. Both caps keep startup/resume thresholds at 1 s. Frozen DQN and highest-allowed use the same observed-goodput safety filter.

## Simulator: four new sessions and one known control

Each row covers a full video. Backlog is an estimate of compressed segment bytes that remain in the playable queue, counting the whole partially played head segment. It is not allocated browser RAM and excludes decoded frames, source-buffer overhead and in-flight requests.

| Cap s | Policy | Session | Startup s | Stalls s | Downloaded MiB | Discarded KiB | Peak encoded backlog MiB | Targets met |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 30 | lowest | prefetch-ferry-2011-01-29_1800CET | 29.181 | 0.000 | 7.061 | 0.000 | 1.206 | False |
| 30 | lowest | prefetch-ferry-2011-01-31_1830CET | 0.684 | 0.000 | 7.061 | 0.000 | 1.206 | True |
| 30 | lowest | prefetch-ferry-2011-02-01_0629CET | 0.732 | 0.000 | 7.061 | 0.000 | 1.206 | True |
| 30 | lowest | prefetch-ferry-2011-02-01_0740CET | 8.135 | 0.000 | 7.061 | 0.000 | 1.206 | False |
| 30 | lowest | fresh-ferry-2010-09-21_1622CEST | 0.652 | 21.498 | 7.061 | 0.000 | 1.206 | False |
| 30 | highest_allowed | prefetch-ferry-2011-01-29_1800CET | 29.181 | 0.000 | 24.186 | 8.111 | 7.183 | False |
| 30 | highest_allowed | prefetch-ferry-2011-01-31_1830CET | 0.684 | 0.000 | 34.023 | 0.000 | 7.101 | True |
| 30 | highest_allowed | prefetch-ferry-2011-02-01_0629CET | 0.732 | 0.000 | 37.132 | 0.000 | 7.108 | True |
| 30 | highest_allowed | prefetch-ferry-2011-02-01_0740CET | 8.135 | 0.000 | 26.076 | 0.000 | 6.086 | False |
| 30 | highest_allowed | fresh-ferry-2010-09-21_1622CEST | 0.652 | 21.583 | 19.088 | 6.559 | 4.702 | False |
| 30 | dqn | prefetch-ferry-2011-01-29_1800CET | 29.181 | 0.000 | 24.186 | 8.111 | 7.183 | False |
| 30 | dqn | prefetch-ferry-2011-01-31_1830CET | 0.684 | 0.000 | 34.023 | 0.000 | 7.101 | True |
| 30 | dqn | prefetch-ferry-2011-02-01_0629CET | 0.732 | 0.000 | 37.132 | 0.000 | 7.108 | True |
| 30 | dqn | prefetch-ferry-2011-02-01_0740CET | 8.135 | 0.000 | 26.076 | 0.000 | 6.086 | False |
| 30 | dqn | fresh-ferry-2010-09-21_1622CEST | 0.652 | 21.583 | 19.088 | 6.559 | 4.702 | False |
| 60 | lowest | prefetch-ferry-2011-01-29_1800CET | 29.181 | 0.000 | 7.061 | 0.000 | 2.265 | False |
| 60 | lowest | prefetch-ferry-2011-01-31_1830CET | 0.684 | 0.000 | 7.061 | 0.000 | 2.265 | True |
| 60 | lowest | prefetch-ferry-2011-02-01_0629CET | 0.732 | 0.000 | 7.061 | 0.000 | 2.265 | True |
| 60 | lowest | prefetch-ferry-2011-02-01_0740CET | 8.135 | 0.000 | 7.061 | 0.000 | 2.265 | False |
| 60 | lowest | fresh-ferry-2010-09-21_1622CEST | 0.652 | 0.000 | 7.061 | 0.000 | 2.212 | True |
| 60 | highest_allowed | prefetch-ferry-2011-01-29_1800CET | 29.181 | 0.000 | 30.919 | 0.000 | 13.067 | False |
| 60 | highest_allowed | prefetch-ferry-2011-01-31_1830CET | 0.684 | 0.000 | 28.958 | 7.599 | 13.317 | True |
| 60 | highest_allowed | prefetch-ferry-2011-02-01_0629CET | 0.732 | 0.000 | 37.132 | 0.000 | 13.802 | True |
| 60 | highest_allowed | prefetch-ferry-2011-02-01_0740CET | 8.135 | 0.000 | 26.932 | 0.000 | 11.845 | False |
| 60 | highest_allowed | fresh-ferry-2010-09-21_1622CEST | 0.652 | 0.000 | 23.272 | 247.348 | 9.366 | True |
| 60 | dqn | prefetch-ferry-2011-01-29_1800CET | 29.181 | 0.000 | 30.919 | 0.000 | 13.067 | False |
| 60 | dqn | prefetch-ferry-2011-01-31_1830CET | 0.684 | 0.000 | 28.958 | 7.599 | 13.317 | True |
| 60 | dqn | prefetch-ferry-2011-02-01_0629CET | 0.732 | 0.000 | 37.132 | 0.000 | 13.802 | True |
| 60 | dqn | prefetch-ferry-2011-02-01_0740CET | 8.135 | 0.000 | 26.932 | 0.000 | 11.845 | False |
| 60 | dqn | fresh-ferry-2010-09-21_1622CEST | 0.652 | 0.000 | 23.272 | 247.348 | 9.366 | True |

## Native Chrome: completion, timing and actual process memory

Eight isolated headless Chrome instances ran in the same capture batch. RSS is sampled every 2 s across each browser process tree and can count shared pages multiple times. The empty-browser baseline and sampled peak include browser/runtime/codecs; the difference is not an isolated measurement of buffer allocation. JS heap excludes most native decoder/MSE storage. Python inference-server memory is not included.

| Case | Complete | Startup s | Stalls s | Media sent MiB | Discarded unappended KiB | Baseline RSS MiB | Sampled peak RSS MiB | Peak JS heap MiB | Targets met |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap30-new-first-dqn | True | 30.193 | 0.082 | 28.226 | 0.000 | 912.7 | 1009.1 | 1.52 | False |
| cap30-new-first-highest-allowed | True | 29.381 | 0.020 | 27.321 | 0.000 | 982.2 | 931.9 | 1.34 | False |
| cap30-known-hard-dqn | False | 0.976 | 4.351 | 17.562 | 144.053 | 972.9 | 920.2 | 1.31 | False |
| cap30-known-hard-highest-allowed | True | 1.472 | 23.822 | 19.082 | 0.000 | 962.0 | 942.3 | 1.49 | False |
| cap60-new-first-dqn | True | 29.414 | 0.108 | 30.310 | 0.000 | 940.9 | 925.2 | 1.41 | False |
| cap60-new-first-highest-allowed | True | 29.504 | 0.079 | 30.630 | 0.000 | 892.0 | 914.1 | 1.37 | False |
| cap60-known-hard-dqn | True | 2.513 | 0.000 | 22.136 | 0.000 | 823.0 | 1070.0 | 1.50 | True |
| cap60-known-hard-highest-allowed | False | 2.330 | 0.000 | 22.232 | 142.017 | 849.0 | 1061.1 | 1.47 | False |

Startup/stall timings in this report are reconstructed from the archived native event timestamps. Two original recorder stall counters differed from their event logs by about 40 ms because handlers read the clock separately for logging and counters. Raw captures/counters remain unchanged; discrepancies are explicit in summary.json. A failing handler-clock test reproduced the problem, and the recorder now uses one timestamp per event. The timing acceptance limits were not widened.

Failed native runs remain partial. Their stall totals do not describe full-video performance. Media bytes above are server-sent bytes; receipt of a failed response by the client cannot be proved from this audit. Prefetched but unplayed appended bytes are separately estimated in summary.json. Initialization traffic is excluded.

## Gate and next step

Simulator 60 s gate on new sessions: **False**. Native 60 s gate: **False**. No retraining was started. The fixed seed budget in protocol.json is a conditional plan, not an executed training result.

The native player still fails requests after 30 s and has no recovery. A larger buffer can protect already-buffered playback while a request is slow, but cannot make this player resume after its request loop exits. Port and validate bounded request recovery before using native playback results to justify retraining. Slow first delivery also needs separate startup treatment; increasing the prefetch cap cannot download the first segment sooner.

Independent browser runs can follow different request timing and measurements, even when policies would agree on identical input. The DQN/rule difference on the hard control is not proof of a neural advantage from one trial per case.

Keep default cap and frozen weights unchanged. No memory acceptance claim is made without a target-device budget. These are four more sessions on the same old route and one silent video; there is no new-route/video generalization claim.

Reproduce simulator: `.venv/bin/python -m scripts.evaluate_prefetch --prepare` then `.venv/bin/python -m scripts.evaluate_prefetch`. Native commands and source/config identity are documented in docs/prefetch-validation.md.
