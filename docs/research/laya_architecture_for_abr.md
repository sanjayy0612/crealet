# Laya architecture: a starting point for ABR design

Reviewed 27 October 2026 against the public `main` branch. These URLs are mutable;
this note describes the source inspected, not a pinned release or reproduced result.
The existing ABR dense/GRU models are older implementations, not the assumed design
for the next model.

## Verified inference structure

Laya's `build_sequence` constructs one sequence per question:

```text
[CLS] question type + instructions [SEP]
[MASK] option 0 [MASK] option 1 ... [SEP] state [SEP]
```

Dictionary/list states are serialized as JSON text. Options are rendered as text.
The marker positions are saved so the network can later extract one vector per
option. `[MASK]` is an option readout marker here; the decision path does not
generate words to replace it. Input and option budgets can truncate the text.

Source: [`serialize_state`, `render_options`, `build_sequence`, `build_head`](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py).

`DecisionModel` applies:

1. A bidirectional transformer encoder to the sequence.
2. A learned question-type embedding, added to the token representations.
3. An additional transformer encoder head (default two layers, configurable).
4. A gather operation extracting the vector at each option marker.
5. A shared scorer: LayerNorm -> Linear(d,d) -> GELU -> Linear(d,1).

This produces a variable-length set of option logits. Padding markers are masked.
The additional head processes the token sequence before marker extraction; it is
not merely a fixed classifier over the `[CLS]` vector.

Source: [`DecisionModel.__init__`, `DecisionModel.forward`, `build_model`](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py).

The model also produces auxiliary action logits from the first-token vector plus
distribution statistics (top probability, top-two gap, normalized entropy, option
count). These auxiliary logits are distinct from the option-selection logits.

The README identifies ModernBERT-large for the English and typed-decisions
checkpoints (about 421M parameters), and mmBERT-base for multilingual (about 322M).
These are checkpoint descriptions, not ABR size recommendations or independently
verified latency results.

Source: [official README](https://github.com/NandhaKishorM/laya/blob/main/README.md).

## Attention and option order

The default sequential layout permits options and state to interact through
bidirectional attention, subject to the backbone's attention pattern. Sequential
positions can make option order affect scores. The source also implements an
optional parallel layout with shared option positions and restrictions on direct
cross-option attention in the backbone. Its comments state published checkpoints
were trained with the sequential layout; do not assume the optional layout is
enabled for a particular checkpoint.

Source: [`parallel_layout`, `parallel_option_masks`, `uses_parallel_layout`](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py).

## What single-pass means

For one ordinary question, all candidate scores are produced in one model forward
execution. This execution still contains many transformer layers. It does not
generate an answer token by token. Multiple questions can be batched as separate
sequences; this is not evidence that all questions share one encoded state.
Long-document scanning and candidate shortlisting can require additional passes.

Sources: [`build_sequence`, `DecisionModel.forward`, `window_budget`](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py),
[official README](https://github.com/NandhaKishorM/laya/blob/main/README.md).

## Training: what is established here

The README describes RLCD training against proper scoring rules. The inspected
`proper_reward` helper combines a target-distribution log score and spherical
score, plus a ranked-probability penalty for ordinal `score` questions. This
helper alone does not establish the complete optimizer, sampling procedure, or
checkpoint training history; those require a separate inspection of training code
and artifacts. Softmax outputs and concentrated distributions do not by themselves
establish calibrated correctness probabilities.

Sources: [`proper_reward`, `answer_confidence`](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py),
[official README](https://github.com/NandhaKishorM/laya/blob/main/README.md).

## Tentative ABR translation—not a design decision

A mechanism worth investigating is joint contextualization of history, player
state and candidate representations, followed by one shared readout per candidate.
An ABR version could embed numeric observations directly instead of serializing
them as language. That would be a new numeric architecture inspired by Laya, not
the original pretrained Laya model.

Open decisions include whether joint attention is useful for short numeric
histories; how to encode temporal order without imposing arbitrary candidate
order; what each score means (policy logit, expected return, or another target);
and how inference cost and playback outcomes compare on held-out conditions.
Laya's target-distribution reward is not automatically a replacement for ABR's
sequential quality/stall objective. No architecture or training method is selected
by this note.
