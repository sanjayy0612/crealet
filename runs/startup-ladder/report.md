# Remaining-work execution: startup media, training and broader tests

## Outcome

A real low-bandwidth ladder fixed startup on the known slow cases in native Chrome. Three matched DQN runs were completed, but none beat the same rule on the unused sessions. No new neural checkpoint or default buffer/media setting is promoted.

## Real media and quality tradeoff

The existing highest-quality Envivio source was decoded and re-encoded into seven aligned representations: 8, 300, 750, 1200, 1850, 2850 and 4300 kbps. The 8 kbps rung is only 96x54 pixels. All media sizes are actual encoded bytes; sample durations come from MP4 boxes and every generated stream passed strict FFmpeg decode. Decoded-frame PTS were normalized, avoiding the old source timestamp irregularities. The original inputs and checkpoints remain intact.

The new first fragment is **5478 bytes** with **792 bytes** of initialization. The old smallest first fragment was 181,801 bytes. This is an encoding/ladder improvement, not evidence that DQN learned a better startup decision.

## Native desktop playback

| Envivio case | Startup s | Stalls s | Completed |
|---|---:|---:|---|
| slow-first-dqn | 2.792 | 0.000 | True |
| slow-first-highest-allowed | 2.630 | 0.000 | True |
| slow-start-dqn | 2.800 | 0.000 | True |
| slow-start-highest-allowed | 2.422 | 0.000 | True |
| hard-dqn | 0.346 | 0.000 | True |
| hard-highest-allowed | 0.369 | 0.000 | True |

All six cases passed the predeclared completion, startup <=3 s and stalls <=8 s gates. They use a frozen model trained on the old media, an explicit 60 s cap, and recovery. Every complete Envivio run decoded 4,842 frames. These are known development cases.

The second real content is the [Sintel trailer](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4), copyright Blender Foundation / Durian Open Movie Team, [CC BY 3.0](https://durian.blender.org/about/). It was re-encoded video-only, normalized from 24 to 25 fps, into the same nominal ladder. Its duration is 52.2 s. It was never used for training or checkpoint selection.

| Sintel case | Startup s | Stalls s | Decoded frames | Lowest-rung fraction |
|---|---:|---:|---:|---:|
| sintel-dqn | 2.705 | 0.000 | 1305 | 100.0% |
| sintel-highest-allowed | 2.639 | 0.000 | 1305 | 100.0% |

The slow-session Sintel runs stayed at 8 kbps for the entire clip. Smooth playback does not establish acceptable visual quality. Bitrate is only the reward proxy; no VMAF/MOS or human-quality acceptance is claimed.

Sampled browser process-tree RSS and server decision timings are shown below. RSS includes runtime/codecs and may count shared pages more than once. Decision time includes the shared inference lock under parallel trials; it excludes HTTP transport. Neither metric establishes a mobile-device budget.

| Case | Sampled Chrome RSS MiB | Server decision p95 ms |
|---|---:|---:|
| slow-first-dqn | 1129.4 | 22.442 |
| slow-first-highest-allowed | 1173.1 | 1.786 |
| slow-start-dqn | 1106.6 | 16.459 |
| slow-start-highest-allowed | 1136.3 | 1.955 |
| hard-dqn | 1176.7 | 22.391 |
| hard-highest-allowed | 1240.6 | 1.013 |
| sintel-dqn | 1034.6 | 19.836 |
| sintel-highest-allowed | 1164.9 | 1.367 |

## Matched training and fixed selection

Seeds 7, 11 and 13 each ran 120 episodes on the new real ladder, cap60 and recovery. Data, safety mask, reward, replay/update budgets and validation selection were fixed before scoring unused sessions. Checkpoints had to complete validation, start within 3 s, and cause no more stalls per trace than the rule (also <=8 s). Best original validation reward chose the checkpoint within each seed. This is 360 training episodes, 17,640 environment steps and 4,038 optimizer updates. Final weights were not silently substituted for the selected snapshots.

| Seed | Selected episode | Validation reward | Rule reward | Validation win |
|---:|---:|---:|---:|---|
| 7 | 100 | 676.763 | 646.845 | True |
| 11 | 80 | 662.485 | 646.845 | True |
| 13 | 40 | 595.384 | 646.845 | False |

## Four unused network sessions

All fixed checkpoints were frozen with hashes before scoring these sessions. They are different recordings from the same historical ferry route. All policies completed all videos with zero stalls. One session exceeded the startup gate for every policy on both videos; it remains in the results.

| Content | Policy | Complete | Targets passed | Mean played kbps | Mean stalls s | Mean reward |
|---|---|---:|---:|---:|---:|---:|
| envivio | highest_allowed | 4/4 | 3/4 | 772.1 | 0.000 | 557.456 |
| envivio | lowest | 4/4 | 3/4 | 8.0 | 0.000 | -2.814 |
| envivio | old_frozen | 4/4 | 3/4 | 772.1 | 0.000 | 557.456 |
| envivio | seed_7 | 4/4 | 3/4 | 627.9 | 0.000 | 538.700 |
| envivio | seed_11 | 4/4 | 3/4 | 651.6 | 0.000 | 544.700 |
| envivio | seed_13 | 4/4 | 3/4 | 573.5 | 0.000 | 516.283 |
| sintel | highest_allowed | 4/4 | 3/4 | 300.4 | 0.000 | 84.886 |
| sintel | lowest | 4/4 | 3/4 | 8.0 | 0.000 | -2.806 |
| sintel | old_frozen | 4/4 | 3/4 | 300.4 | 0.000 | 84.886 |
| sintel | seed_7 | 4/4 | 3/4 | 247.0 | 0.000 | 81.023 |
| sintel | seed_11 | 4/4 | 3/4 | 299.6 | 0.000 | 84.839 |
| sintel | seed_13 | 4/4 | 3/4 | 207.5 | 0.000 | 63.892 |

Every newly trained seed had lower transfer reward than the rule on both contents. Two validation wins did not generalize. Reward values are comparable only within the same new ladder/content; the minimum-bitrate reference changed from the old ladder.

## Remaining startup bound

| New session | Service by 3 s, bytes | Smallest first fragment bytes | Optimistic startup s |
|---|---:|---:|---:|
| transfer-ferry-2011-02-01_0840CET | 400425 | 5478 | 0.085 |
| transfer-ferry-2011-02-01_1000CET | 1144 | 5478 | 10.346 |
| transfer-ferry-2011-02-01_1539CET | 154653 | 5478 | 0.563 |
| transfer-ferry-2011-02-01_1639CET | 583317 | 5478 | 0.019 |

The remaining failing session supplies only about 1.1 KB by 3 s; the new first fragment is still about 5.5 KB. Its roughly 10.35 s startup is an asset/service limit under the tested contract, not a DQN-specific failure. No weights can make missing bytes arrive. Further packaging work would need a smaller independently playable unit and a new confirmation set; these outcomes must not be tuned away.

## Delivery and remaining limits

The desktop research player can now load explicitly changed media with measured sizes/hashes. It retains strict default media checks, records original training identity separately, and refuses to drop recovery required by a newly trained checkpoint. The encoding scripts, fixed protocols, selected checkpoints and captures are retained.

The overall project is not production-approved. No target device or memory budget was supplied, no mobile/browser matrix was executed, and low-rung visual quality has not been accepted. The neural advantage remains unproven. The practical tested reference remains the safety rule with bounded recovery and explicit media settings.

Commands, asset preparation and source credit are documented in docs/startup-ladder.md. Large media payloads stay in artifacts/; tracked provenance and exact recipes rebuild them.
