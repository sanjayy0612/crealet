# Filtered DQN: higher bitrate with the existing stall limits

The combined controller now varies bitrate. Reserved-trace mean bitrate rose
from 300 to 736 kbps with zero stalls. Native browser playback completed at
1,002 kbps on the measured connection and 713 kbps on the outage case, with
zero stalls and startup below three seconds in both.

**The improvement is attributable to the explicit filter.** A simple rule that
chooses the highest permitted bitrate exactly matches filtered DQN on validation
and reserved traces. All 98 native decisions also selected the highest permitted
action. Learned weights have not demonstrated an advantage over that rule.

## Method

The old run explored unrestricted actions and only filtered checkpoints after
training. One of 80 validation checkpoints passed; it stayed at 300 kbps.
The new action filter always permits the lowest bitrate. Higher bitrates are
allowed only with at least 12 seconds buffered and below 65% of the harmonic
mean of up to eight past valid goodput samples. The constants were chosen using
validation diagnostics. No future trace values are used.

The filter restricts random exploration, greedy actions, next-action selection
in the Double DQN target, and deployment. The online network chooses the
permitted next action and the target network scores it, preserving the
[Double Q-learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295)
selection/evaluation separation. Checkpoint metadata records the filter; player
and rollout CLI apply it explicitly. It is a heuristic, not a stall guarantee.

The architecture, dataset and reward are unchanged. The limits remain startup
at most 3 seconds and stalls no greater than the smaller of 8 seconds and the
throughput baseline on each validation trace. Stall penalty remains 12.
One seed-7 run used 600 episodes, 29,400 environment steps and 7,226 gradient
updates. All 30 validation checkpoints passed. The best eligible checkpoint was
episode 20, frozen before reserved-trace evaluation.

## Simulator comparison

| Policy | Validation bitrate | Validation stalls | Reserved bitrate | Reserved stalls |
| --- | ---: | ---: | ---: | ---: |
| Previous accepted DQN | 300 kbps | 2.164 s | 300 kbps | 0.000 s |
| DQN plus filter | 717 kbps | 2.164 s | 736 kbps | 0.000 s |
| Highest allowed, no neural model | 717 kbps | 2.164 s | 736 kbps | 0.000 s |
| Original throughput rule | 1,031 kbps | 6.430 s | 1,039 kbps | 1.159 s |

Every row completed all three validation and all three reserved episodes.
The original throughput rule violates the stricter validation stall limits on
two traces. The new controller stays at 300 kbps on the extremely poor train
journey and increases bitrate on stronger connections.

As an ablation, the new weights were evaluated on validation without the filter:
none of three episodes completed, with mean stalls of 69.34 seconds on partial
playback. **These weights must be used with their recorded filter.**
The reserved sessions are fixed regression data used by the previous experiment;
they never contribute gradients or checkpoint selection, but this is not a
fresh external generalization test.

## Actual browser playback

| Connection | Complete | Mean played bitrate | Startup | Stalls |
| --- | --- | ---: | ---: | ---: |
| Measured ferry | Yes | 1,002 kbps | 0.91 s | 0.00 s |
| Controlled outage | Yes | 713 kbps | 1.11 s | 0.00 s |

Each run decoded 49 fragments and 4,842 frames. Measured-case choices included
300, 750, 1,200 and 1,850 kbps; outage choices included 300, 750 and 1,200 kbps.
Every choice respected the recorded filter and matched the highest allowed
action. The server audit records allowed actions and whether filtering applied.

Both native captures passed independent identity, segment-byte, visibility,
decoded-frame, decision and stall-event checks. Artifacts are in
`../player-safe/`. The same 30 second request timeout and 210 second wall budget
were used. The earlier experiment ran five Chrome instances and this one ran
two, so startup changes cannot be attributed solely to the policy. HTTP pacing
does not reproduce TCP or radio behavior.

The full suite passed **104 tests**, including buffer boundaries, outages,
missing history, candidate permutation, padding and masked bootstrap selection.
Two existing Gymnasium infinite-bound warnings remain. Browser capture and
independent capture-audit processes both completed successfully.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m abr.models.train_dueling_dqn --dataset data/hard_envivio/dataset.json --episodes 600 --seed 7 --evaluate-every 20 --rebuffer-penalty 12 --max-rebuffer-s 8 --max-startup-s 3 --safety-filter --buffer-reserve-s 12 --throughput-factor .65 --output runs/dqn-safe
PYTHONPATH=. .venv/bin/python -m abr.player_server --port 8788 --checkpoint runs/dqn-safe/dueling_double_dqn_feasible.npz
# Another terminal:
node scripts/capture_policy_player.mjs --port=8788 --output=runs/player-safe --policies=dqn
PYTHONPATH=. .venv/bin/python scripts/audit_player_captures.py --directory runs/player-safe --checkpoint runs/dqn-safe/dueling_double_dqn_feasible.npz --cases=dqn-measured,dqn-outage
```

This is one video, one seed, historical mobile logs and two local silent-video
browser cases. The filter cannot foresee an outage or guarantee future stalls
will be zero. A neural deployment advantage requires beating the matching
non-neural rule on additional independent sessions.
