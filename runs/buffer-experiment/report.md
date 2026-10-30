# Startup/resume buffer experiment

Frozen DQN, progress-based request recovery, no training. All nine profiles and four additional sessions were fixed before downloading/replay.

Targets: complete every video, startup <=3 seconds, total stalls <=8 seconds. Buffer thresholds are media seconds, not wall-clock seconds. Segments are indivisible: the first segments last 3.9934 seconds, so thresholds 4 and 8 require two and three initial segments respectively. The buffer cap remains 30 seconds.

## Four additional sessions

Means below cover completed videos only. Completion and timing gates include every session. Total delay is startup plus all playback stalls; a larger resume threshold can lengthen each pause.

| Policy | Startup/resume thresholds | Complete | All targets met | Mean startup s | Mean stalls s | Mean total delay s | Mean played kbps |
|---|---|---:|---:|---:|---:|---:|---:|
| lowest | start1.0-resume1.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 300.00 |
| highest_allowed | start1.0-resume1.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 1024.87 |
| dqn | start1.0-resume1.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 1024.87 |
| lowest | start1.0-resume4.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 300.00 |
| highest_allowed | start1.0-resume4.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 1024.87 |
| dqn | start1.0-resume4.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 1024.87 |
| lowest | start1.0-resume8.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 300.00 |
| highest_allowed | start1.0-resume8.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 1024.87 |
| dqn | start1.0-resume8.0 | 4/4 | 4/4 | 1.05 | 0.00 | 1.05 | 1024.87 |
| lowest | start4.0-resume1.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 300.00 |
| highest_allowed | start4.0-resume1.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 1022.55 |
| dqn | start4.0-resume1.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 1022.55 |
| lowest | start4.0-resume4.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 300.00 |
| highest_allowed | start4.0-resume4.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 1022.55 |
| dqn | start4.0-resume4.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 1022.55 |
| lowest | start4.0-resume8.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 300.00 |
| highest_allowed | start4.0-resume8.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 1022.55 |
| dqn | start4.0-resume8.0 | 4/4 | 4/4 | 1.61 | 0.00 | 1.61 | 1022.55 |
| lowest | start8.0-resume1.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 300.00 |
| highest_allowed | start8.0-resume1.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 1029.51 |
| dqn | start8.0-resume1.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 1029.51 |
| lowest | start8.0-resume4.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 300.00 |
| highest_allowed | start8.0-resume4.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 1029.51 |
| dqn | start8.0-resume4.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 1029.51 |
| lowest | start8.0-resume8.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 300.00 |
| highest_allowed | start8.0-resume8.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 1029.51 |
| dqn | start8.0-resume8.0 | 4/4 | 3/4 | 2.17 | 0.00 | 2.17 | 1029.51 |

## DQN per-session timing

| Sessions | Profile | Trace | Complete | Startup s | Stalls s | Stall count | Targets met |
|---|---|---|---|---:|---:|---:|---|
| additional_sessions | start1.0-resume1.0 | buffer-ferry-2010-09-22_0702CEST | True | 0.58 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume1.0 | buffer-ferry-2010-09-23_1001CEST | True | 0.94 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume1.0 | buffer-ferry-2010-09-28_1003CEST | True | 1.79 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume1.0 | buffer-ferry-2010-09-29_0702CEST | True | 0.87 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume4.0 | buffer-ferry-2010-09-22_0702CEST | True | 0.58 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume4.0 | buffer-ferry-2010-09-23_1001CEST | True | 0.94 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume4.0 | buffer-ferry-2010-09-28_1003CEST | True | 1.79 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume4.0 | buffer-ferry-2010-09-29_0702CEST | True | 0.87 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume8.0 | buffer-ferry-2010-09-22_0702CEST | True | 0.58 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume8.0 | buffer-ferry-2010-09-23_1001CEST | True | 0.94 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume8.0 | buffer-ferry-2010-09-28_1003CEST | True | 1.79 | 0.00 | 0 | True |
| additional_sessions | start1.0-resume8.0 | buffer-ferry-2010-09-29_0702CEST | True | 0.87 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume1.0 | buffer-ferry-2010-09-22_0702CEST | True | 1.13 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume1.0 | buffer-ferry-2010-09-23_1001CEST | True | 1.44 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume1.0 | buffer-ferry-2010-09-28_1003CEST | True | 2.46 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume1.0 | buffer-ferry-2010-09-29_0702CEST | True | 1.40 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume4.0 | buffer-ferry-2010-09-22_0702CEST | True | 1.13 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume4.0 | buffer-ferry-2010-09-23_1001CEST | True | 1.44 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume4.0 | buffer-ferry-2010-09-28_1003CEST | True | 2.46 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume4.0 | buffer-ferry-2010-09-29_0702CEST | True | 1.40 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume8.0 | buffer-ferry-2010-09-22_0702CEST | True | 1.13 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume8.0 | buffer-ferry-2010-09-23_1001CEST | True | 1.44 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume8.0 | buffer-ferry-2010-09-28_1003CEST | True | 2.46 | 0.00 | 0 | True |
| additional_sessions | start4.0-resume8.0 | buffer-ferry-2010-09-29_0702CEST | True | 1.40 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume1.0 | buffer-ferry-2010-09-22_0702CEST | True | 1.75 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume1.0 | buffer-ferry-2010-09-23_1001CEST | True | 1.90 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume1.0 | buffer-ferry-2010-09-28_1003CEST | True | 3.15 | 0.00 | 0 | False |
| additional_sessions | start8.0-resume1.0 | buffer-ferry-2010-09-29_0702CEST | True | 1.88 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume4.0 | buffer-ferry-2010-09-22_0702CEST | True | 1.75 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume4.0 | buffer-ferry-2010-09-23_1001CEST | True | 1.90 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume4.0 | buffer-ferry-2010-09-28_1003CEST | True | 3.15 | 0.00 | 0 | False |
| additional_sessions | start8.0-resume4.0 | buffer-ferry-2010-09-29_0702CEST | True | 1.88 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume8.0 | buffer-ferry-2010-09-22_0702CEST | True | 1.75 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume8.0 | buffer-ferry-2010-09-23_1001CEST | True | 1.90 | 0.00 | 0 | True |
| additional_sessions | start8.0-resume8.0 | buffer-ferry-2010-09-28_1003CEST | True | 3.15 | 0.00 | 0 | False |
| additional_sessions | start8.0-resume8.0 | buffer-ferry-2010-09-29_0702CEST | True | 1.88 | 0.00 | 0 | True |
| previous_failure_controls | start1.0-resume1.0 | fresh-ferry-2010-09-21_1622CEST | True | 0.65 | 21.58 | 3 | False |
| previous_failure_controls | start1.0-resume1.0 | fresh-ferry-2010-09-21_1735CEST | True | 1.00 | 9.93 | 1 | False |
| previous_failure_controls | start1.0-resume4.0 | fresh-ferry-2010-09-21_1622CEST | True | 0.65 | 23.26 | 2 | False |
| previous_failure_controls | start1.0-resume4.0 | fresh-ferry-2010-09-21_1735CEST | True | 1.00 | 11.20 | 1 | False |
| previous_failure_controls | start1.0-resume8.0 | fresh-ferry-2010-09-21_1622CEST | True | 0.65 | 29.57 | 1 | False |
| previous_failure_controls | start1.0-resume8.0 | fresh-ferry-2010-09-21_1735CEST | True | 1.00 | 12.41 | 1 | False |
| previous_failure_controls | start4.0-resume1.0 | fresh-ferry-2010-09-21_1622CEST | True | 1.23 | 21.02 | 3 | False |
| previous_failure_controls | start4.0-resume1.0 | fresh-ferry-2010-09-21_1735CEST | True | 2.03 | 6.56 | 1 | True |
| previous_failure_controls | start4.0-resume4.0 | fresh-ferry-2010-09-21_1622CEST | True | 1.23 | 22.70 | 2 | False |
| previous_failure_controls | start4.0-resume4.0 | fresh-ferry-2010-09-21_1735CEST | True | 2.03 | 8.11 | 1 | False |
| previous_failure_controls | start4.0-resume8.0 | fresh-ferry-2010-09-21_1622CEST | True | 1.23 | 29.00 | 1 | False |
| previous_failure_controls | start4.0-resume8.0 | fresh-ferry-2010-09-21_1735CEST | True | 2.03 | 9.72 | 1 | False |
| previous_failure_controls | start8.0-resume1.0 | fresh-ferry-2010-09-21_1622CEST | True | 1.78 | 23.08 | 2 | False |
| previous_failure_controls | start8.0-resume1.0 | fresh-ferry-2010-09-21_1735CEST | True | 2.95 | 5.90 | 1 | True |
| previous_failure_controls | start8.0-resume4.0 | fresh-ferry-2010-09-21_1622CEST | True | 1.78 | 27.08 | 1 | False |
| previous_failure_controls | start8.0-resume4.0 | fresh-ferry-2010-09-21_1735CEST | True | 2.95 | 7.53 | 1 | True |
| previous_failure_controls | start8.0-resume8.0 | fresh-ferry-2010-09-21_1622CEST | True | 1.78 | 29.10 | 1 | False |
| previous_failure_controls | start8.0-resume8.0 | fresh-ferry-2010-09-21_1735CEST | True | 2.95 | 9.00 | 1 | False |

## Interpretation and limits

DQN and highest-allowed differed in action sequences in 0 of 54 paired profile/session comparisons. Changed startup/resume context was not used to train this checkpoint; weights remain frozen.

The default profile completed 4/4 additional sessions, with maximum stalls 0.00 s. Zero stalls give no opportunity to demonstrate a resume-buffer improvement. Mean startup was 1.05 s, 1.61 s and 2.17 s at thresholds 1, 4 and 8 s. The 8 s threshold passed all targets on 3/4 sessions. Keep the current defaults; these results do not establish a better profile.

On the two previous failure controls, startup=4 s and resume=1 s reduced stalls from 21.58/9.93 s to 21.02/6.56 s, with startup changing from 0.65/1.00 s to 1.23/2.03 s. One session still failed the stall gate. Increasing only the resume threshold to 8 s increased stalls to 29.57/12.41 s even though it reduced the number of separate pauses. A lower pause count does not mean less waiting.

No profile is promoted from this comparison. Choosing a profile after seeing these results requires a separate confirmation set. A mean below a limit does not imply every session passed it. Previous failure sessions are development controls only.

Same VOD and ferry route, four additional sessions only; Frozen DQN was trained with startup/resume thresholds of 1 s; changed context is untrained; Thresholds are media seconds; whole segments arrive together (first segment 3.9934 s); All nine profiles share these sessions; this is exploratory comparison, not independent validation of a selected profile; Observed goodput replay, RTT assumed 80 ms; no packet/radio simulation; Simulator only; browser unchanged.

Reproduce: `.venv/bin/python -m scripts.prepare_buffer_experiment` then `.venv/bin/python -m scripts.evaluate_buffer_experiment`.
