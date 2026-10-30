# Buffer thresholds: measured tradeoff

The experiment changed startup and resume thresholds through 1, 4 and 8 media
seconds, independently. It kept the 30-second buffer cap, frozen DQN checkpoint,
bitrate safety rule, stall penalty and progress-based request recovery unchanged.
The existing simulator already supports these settings; no runtime code changed.

Before downloading/replaying data, the protocol fixed nine profiles and the
first four unused ferry sessions in archive filename order. It compares frozen
DQN, the highest-allowed bitrate rule, and fixed lowest quality. The same nine
profiles also replay the two previously failed sessions as development controls.
This makes 162 episodes in total, with no new training or result-based tuning.

All four additional sessions already completed with zero stalls under the default
1/1-second thresholds. Larger startup thresholds raised mean startup from 1.05 s
to 1.61 s (4-second threshold) and 2.17 s (8-second threshold). One session with
the 8-second threshold exceeded the 3-second startup limit. Resume changes cannot
show a benefit on these four sessions because no stalls occurred.

On the earlier failed sessions, changing startup to 4 seconds while keeping
resume at 1 second reduced full-playback stalls from 21.58/9.93 seconds to
21.02/6.56 seconds. Startup increased from 0.65/1.00 seconds to 1.23/2.03 seconds.
This helped one case pass, but the other still exceeded the 8-second stall limit.
Increasing only the resume threshold to 8 seconds produced fewer separate pauses
but increased total stall duration to 29.57/12.41 seconds.

DQN and the simple highest-allowed rule produced identical actions in all 54
paired profile/session comparisons. The current results show no learned-policy
advantage. Changed buffer settings are also outside this checkpoint's original
training configuration.

Keep the current defaults. This experiment does not establish a buffer profile
that fixes both difficult cases. A larger resume threshold can turn several short
pauses into one longer pause. A larger startup buffer can move waiting to the start
without improving the total delay. Both effects must be measured, not assumed.

The primary sessions remain the same VOD and ferry route. These are additional
sessions, not evidence about a new video or route. The replay begins at trace time
zero and covers the finite video; it does not replay every possible trace offset.
All profiles share the four sessions, so any choice made after seeing the results
needs separate confirmation. The browser player and deployed defaults are unchanged.

See the [complete report](../runs/buffer-experiment/report.md),
[fixed protocol](../runs/buffer-experiment/protocol.json),
[results and paired differences](../runs/buffer-experiment/results.json), and
[raw trace provenance](../data/buffer-sessions/provenance.json).

Reproduce from the repository root:

```sh
.venv/bin/python -m scripts.prepare_buffer_experiment
.venv/bin/python -m scripts.evaluate_buffer_experiment
```

The next useful investigation is the remaining long-stall case: compare its
minimum-quality download demand with the service available before the playback
buffer empties. This can separate a bitrate/recovery timing problem from a period
when even the lowest representation cannot arrive in time. Such an offline
diagnostic must not feed future bandwidth into the policy.
