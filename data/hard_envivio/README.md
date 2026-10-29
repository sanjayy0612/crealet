# Harder VOD input

Video bytes and durations are unchanged from `../realistic_envivio/`: one
193.68 second Envivio DASH video with 49 segments and six representations.

Measured network input comes directly from the original
[Norway HSDPA dataset](https://datasets.simula.no/hsdpa/), collected in 2010–2011.
The raw files are included under `raw/`, with source URLs and SHA-256 checksums
in `dataset.json`. These are historical observed HTTP goodput measurements,
not measurements of modern networks or action-independent link capacity.

The source defines column 5 as bytes received since the previous measurement.
We divide those bytes by the difference between column 2's monotonic millisecond
timestamps. The first row has no preceding timestamp and is discarded. Some
source files' column 6 intervals omit small amounts of elapsed time, so their
discrepancies are counted in `summary.json`; the conversion uses measured
elapsed timestamps consistently. No throughput values are scaled, no intervals
are repeated, and no synthetic tail is added. RTT is an explicit 80 ms assumption.

Entire routes stay within one split:

- Train: four bus sessions, two car sessions, four tram sessions.
- Validation: two Oslo–Vestby train sessions.
- Test: two ferry sessions.

Two training profiles and one profile in each other split add controlled
capacity drops, complete outages, and recovery. They are explicitly labelled
**synthetic controlled stress cases**, separate from the measured logs. Their
durations and capacities are in `summary.json`.

The fixed lowest policy stalls for 6.49 seconds on one validation train session.
For this experiment, an accepted checkpoint must complete every validation
trace, have startup at most 3 seconds, and have stalls no greater than the
smaller of 8 seconds and the throughput baseline's stalls on that trace. The
8 second ceiling is an experimental acceptance limit, not a claim of ideal
viewer experience. Reward charges 12 per stall second; every compared policy
uses the same reward and player settings. Held-out test results are computed
after checkpoint selection.

Rebuild with `PYTHONPATH=. .venv/bin/python scripts/prepare_hard_vod.py`.
The script retrieves original files with the system HTTPS client and retains
the originals for audit. It requires network access only for uncached files.
