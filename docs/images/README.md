# Working diagram

`abr-working.png` illustrates the measured-input, frozen-policy and playback loop.
It is a generated explanatory illustration, not a screenshot or benchmark plot.
This original diagram records the simulator-only milestone.

Generated with the built-in imagegen tool. The complete generation prompt is in
`abr-working.prompt.txt`; the targeted arrow correction is recorded in
`abr-working.edit-prompt.txt`. The final diagram was visually inspected for label
accuracy and the distinction between offline training and frozen inference.

`abr-working-native.png` is the current README version. The built-in imagegen
tool changed only the recovery support label to include the native player.
The exact edit prompt is `abr-working-native.prompt.txt`; the earlier asset
is retained for history. The current image was visually inspected.

## Research comparison figures

The README comparison charts are generated SVGs, not decorative illustrations.
They are rendered from the committed result summaries with
`python scripts/render_readme_figures.py`:

- `vmaf-policy-comparison.svg` compares mean VMAF for highest-allowed,
  RobustMPC, and three-seed DQN results on Envivio and Sintel. DQN whiskers show
  the observed min-to-max seed means, not confidence intervals.
- `bola-vmaf-comparison.svg` compares BOLA-Basic, its safety-filtered variant,
  highest-allowed, and the frozen DQN on the separate four-session VOD screen.
- `predictor-vs-playback.svg` compares throughput forecast error and decision
  latency on 32 historical Ghent sessions, with the reported playback outcome.

Each chart describes a specific experiment. The figures should not be read as a
single head-to-head leaderboard because their datasets and protocols differ.
