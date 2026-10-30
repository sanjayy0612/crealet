# Native VOD recovery validation

Frozen weights and explicit 60-second cap. Previously observed development cases; startup/resume thresholds remain 1/1 s. Six recovery-on trials and two matched recovery-off hard controls were fixed before replay. No model training or default cap change.

## Implemented behavior

The browser reads streamed body chunks and checks progress while a read is blocked. It can abandon a slow higher-quality request and restart the same segment at lowest quality. At most four retries per segment; initialization and media requests each have a 30 s deadline. Timeout/network/no-progress failures back off 2 s; buffer-risk abandonment does not. Initialization failures consume an attempt too. Whole failed media bodies are discarded. Only a complete body is appended to MediaSource, so retries cannot mix partial encodings.

A cancellation endpoint interrupts the HTTP pacer and acknowledges release of the session request lock before the browser retries. Attempt IDs connect client-received bytes with server audit records. History and previous bitrate reflect the effective representation, with received bytes from failed attempts included in goodput. Recovery is opt-in using `--enable-request-recovery`; existing default mode and checkpoint remain unchanged.

## Native results

Timing uses the common event clock. Partial runs remain labelled partial; their metrics do not describe full-video performance. RSS is the sampled whole Chrome process-tree sum, including runtime/codecs and potentially duplicate shared pages, not isolated buffer RAM.

| Case | Complete | Startup s | Stalls s | Retries | Discarded media KiB | Sampled RSS MiB | Targets met |
|---|---|---:|---:|---:|---:|---:|---|
| recovery-on-slow-first-dqn | True | 29.319 | 0.000 | 1 | 496.000 | 1069.5 | False |
| recovery-on-slow-first-highest-allowed | True | 29.262 | 0.000 | 1 | 512.000 | 1145.8 | False |
| recovery-on-slow-start-dqn | True | 8.265 | 0.000 | 0 | 0.000 | 1085.2 | False |
| recovery-on-slow-start-highest-allowed | True | 8.244 | 0.000 | 0 | 0.000 | 1085.8 | False |
| recovery-on-hard-dqn | True | 1.343 | 0.000 | 2 | 144.000 | 1126.2 | True |
| recovery-on-hard-highest-allowed | True | 0.996 | 0.147 | 2 | 144.000 | 1127.1 | True |
| recovery-off-hard-dqn | False | 0.959 | 0.000 | 0 | 128.000 | 1128.6 | False |
| recovery-off-hard-highest-allowed | False | 0.870 | 0.000 | 0 | 144.000 | 1160.7 | False |

Client-received media bytes and discarded failed-body bytes are measured from ReadableStream chunks. Initialization payloads are recorded separately. This differs from the earlier server-sent estimate, which cannot prove failed-byte receipt. Independent native sessions can have different request timing and observations; a single DQN/rule outcome difference does not prove a neural-policy advantage.

## Slow-start finding

Before first playback, the policy already uses the lowest representation. The following is an offline diagnostic with exact smallest segment sizes, no initialization traffic and zero RTT. It is a favorable lower bound, never an input to the policy.

| Session | Smallest first segment bytes | Service available by 3 s, bytes | Minimum startup s | 3 s target impossible for these assets |
|---|---:|---:|---:|---|
| prefetch-ferry-2011-01-29_1800CET | 181801 | 8038 | 29.181 | True |
| prefetch-ferry-2011-01-31_1830CET | 181801 | 692092 | 0.604 | False |
| prefetch-ferry-2011-02-01_0629CET | 181801 | 640594 | 0.652 | False |
| prefetch-ferry-2011-02-01_0740CET | 181801 | 8912 | 8.132 | True |
| fresh-ferry-2010-09-21_1622CEST | 181801 | 936535 | 0.572 | False |

The two slow-start cases cannot reach first playback by 3 s with the current whole-segment media contract, even without request overhead. Recovery can preserve completion but cannot create missing bytes. The UI now shows first-download byte progress and explicitly records when the startup target is exceeded; it does not restart the clock or hide waiting.

Meeting the startup limit needs a genuinely smaller independently playable startup unit/representation, or a different service condition. Re-encoding/packaging must preserve decoder validity and be measured against the ~8–9 KB service budgets; simply labelling a rung lower or changing the reward is insufficient.

## Decision

Keep the model frozen. Recovery and startup feasibility are separate findings. Do not retrain or claim all playback targets are met while startup remains infeasible for these assets. Next evaluate actual smaller startup media, with exact sizes and real decoding, rather than further tuning on these same traces.

Reproduce using commands in docs/native-recovery.md. Raw attempts, events, transport audits, memory samples and screenshots are retained under captures/.
