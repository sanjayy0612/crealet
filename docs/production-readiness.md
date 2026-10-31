# Production readiness: proposed desktop profile

**Production is not approved.** The target and limits below are now concrete
proposals. Hardware and archived measurements were checked on 1 November 2026
(India time). A measured result does not establish user acceptance.

The machine-readable [profile](../experiments/configs/production_desktop_v1.json)
and [assessment](../runs/production-readiness/assessment.json) separate observed
facts from proposed requirements. This is a post-hoc diagnostic of existing runs.
Freeze the profile before new independent acceptance trials; do not use this
assessment to promote a checkpoint. No player defaults or model weights changed.

## Reference target

Verified locally: **Apple M2, Mac14,2, 8 GiB RAM, arm64, macOS 26.5.1,
Chrome 156.0.8078.74**. See the [hardware record](../runs/production-readiness/hardware.json).
The archived browser captures independently record the same Chrome version.

Proposed pilot scope: one native MSE video player in an isolated Chrome process
tree, with one local Python/PyTorch inference worker. This is the existing desktop
prototype topology. Other tabs are outside the measurement. Safari, Firefox,
Windows, Linux, Android and iOS remain unverified; 8 GiB is a proposed reference
minimum, not a tested lower hardware boundary.

## Memory budget

These are project proposals, not industry limits. Together they reserve at most
2 GiB, or 25% of the reference machine's RAM, for this single-player setup.

| Component | Proposed limit | Observed evidence |
| --- | --- | --- |
| Isolated Chrome process-tree RSS | 1.5 GiB | Maximum sampled across eight playbacks: 1.211 GiB |
| Python inference worker RSS | 512 MiB | Fresh-worker probe lifetime peak: 224.36 MiB |
| Browser plus worker RSS | 2 GiB | Not measured together during playback |
| JavaScript heap, separate diagnostic | 64 MiB | Maximum sampled: 1.74 MiB |
| Retained RSS growth after warm-up | At most 64 MiB over a 30-minute soak | Not measured |

RSS means resident memory reported by the operating system. Browser values were
sampled every two seconds across the isolated Chrome process tree. They include
browser runtime and decoder memory; they do not isolate the video buffer. Summing
process RSS can count shared pages more than once. Sampling can miss short peaks.
JavaScript heap is already part of process memory; do not add it again.

The [worker probe](../runs/production-readiness/server-memory.json) loaded the
original frozen checkpoint with the experimental playback configuration and ran
100 inference calls with one Torch thread. It did not serve HTTP video traffic.
Its 77,826 parameter tensors occupy 311,304 bytes, but the process needs much more
memory for Python and PyTorch. Small weights do not imply a small runtime.

Adding the separately observed browser and worker peaks gives 1.431 GiB. This is
an arithmetic estimate, **not a simultaneous whole-stack peak**. Current samples
fit the draft limits; full playback plus a 30-minute repeated-playback soak must
still measure browser and worker together. Record the warm baseline, peak RSS,
and retained growth after returning to the same idle state. Do not claim no leak
from these short samples.

Chrome's [memory guidance](https://developer.chrome.com/docs/devtools/memory-problems)
explains that acceptable memory depends on the target device. The
[legacy heap API documentation](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory)
also explains why heap readings do not describe all browser memory. Our heap
samples use Chrome DevTools Protocol; RSS remains the process-budget measure.

## Minimum visual quality

Proposed normal floor: **320×180 at 25 fps, nominal 300 kbps**. Proposed emergency
fallback: **96×54 at 25 fps, nominal 8 kbps**, for at most **12 continuous seconds**
and **10% of total media duration**. These limits are design choices
for review, not measured viewer preferences or a perceptual-quality score.

This preserves the project's priority of avoiding stalls. If the network cannot
support the normal floor, continue emergency playback and count a quality-gate
failure. Do not force an unsafe bitrate increase or deliberately pause playback.
A visible degraded-quality state is proposed future behavior; this audit does
not implement it or enforce the limits at runtime.

Both videos need human review at the intended player size, including motion,
faces and text. Neither 180p acceptability nor an acceptable minimum for all
content has been established. The 8 kbps fallback is too coarse to call normal
quality merely because the decoder can play it.

The following values were recomputed from complete captured playback, using each
segment's actual media duration. DQN and the highest-allowed safety rule produced
the same emergency exposure in each paired case.

| Case, both policies | Time below 300 kbps | Share of video | Longest continuous interval | Draft quality gate |
| --- | --- | --- | --- | --- |
| Envivio, hard | 29.68 s | 15.32% | 16 s | Fail |
| Envivio, slow-first | 112 s | 57.83% | 112 s | Fail |
| Envivio, slow-start | 44 s | 22.72% | 44 s | Fail |
| Sintel | 52.2 s | 100% | 52.2 s | Fail |

**Zero of eight native playbacks meet the proposed quality limits.** These are
media exposure measurements, not a viewer study. The audit uses the nominal
bitrate to identify emergency segments in these two fixed ladders; it does not
measure visual quality from pixels or generalize the bitrate floor to other codecs.

## Readiness decision and remaining checks

The reference device is identified and draft budgets are defined. Approval is
still pending because quality exposure fails, human quality acceptance is absent,
and simultaneous full-stack memory and soak measurements are missing.

The existing playback targets remain completion, startup at most 3 seconds, and
total stalls at most 8 seconds per session. One unused network session still
takes about 10.35 seconds to start. Do not hide it or relax that target after the
fact. Define a service-conditioned startup requirement before broader acceptance.
The [training and transfer report](../runs/startup-ladder/report.md) also shows
that the new learned checkpoints did not beat the safety rule on transfer reward.

Next: review these draft device and quality choices, then run a simultaneous
memory soak and independent playback trials against the frozen profile. A future
training run should only follow a feasible media/network target; extra training
cannot deliver bytes that the network does not supply.

## Reproduce the audit

From the repository root, using the checked-in captures and worker probe:

```sh
.venv/bin/python -m scripts.assess_production
```

To refresh the hardware record on the reference Mac with Chrome installed:

```sh
.venv/bin/python -m scripts.assess_production --inspect-hardware
```

The second command requires permission to read local system hardware. Neither
command retrains, launches a player, repeats the worker memory probe, or completes
the missing soak. The assessment records capture, profile and audit-source hashes.
