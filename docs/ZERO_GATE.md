# Exact-zero input gate protocol

This narrow gate is motivated by the previously published zero-back failure. Its evaluation reuses observed HARTH data and is **exploratory**, even though this protocol is committed before executing the gate evaluation. It is not an independent holdout or evidence of realistic sensor-fault detection.

## Rule fixed before evaluation

`finite-v1` is the existing non-finite rejection policy. `exact-zero-v1` additionally rejects a finite window when **all fifteen summaries for either three-axis sensor equal exactly zero**. Reasons identify back, thigh or both. Missing/non-finite values take precedence. No approximate-zero tolerance, variance threshold or learned detector is used. A single zero axis or zero standard deviation is insufficient for rejection. A nonzero constant gravity vector remains eligible.

Do not refit models or change confidence thresholds. The new gate is explicitly selectable in local inference; `finite-v1` remains the default for reproducibility. Rejection is an input decision, not learned abstention. Rejected windows have null labels and confidence.

## Fixed evaluation

- Load the same hash-checked input and raw/sigmoid artifacts as the selective run. Require matching dependencies and clean accepted/error counts and confusion matrices.
- On all four input roles separately, count rejections on unperturbed windows by participant and class. Give denominators, particularly classes 6, 7, 8 (standing, sitting, lying), as defined by the [UCI source](https://archive.ics.uci.edu/dataset/779/harth). These are observed reference-input rejections, only an **apparent false-rejection proxy**: the recordings are not adjudicated fault-free.
- On the same four test participants, repeat all eleven existing stress scenarios. Add zero thigh, zero both sensors, near-zero back (scale its summaries by 1e-6), and nonzero constant back (three copies of mean=1, std=0, min=max=RMS=1).
- Compare finite and exact-zero policies for both frozen variants on paired windows. Report input rejection, model abstention, answers, errors, all-window coverage, risk among answers, per-class/person counts and participant bootstrap intervals. Zero answers imply undefined risk, never zero error rate.
- Report rejected counts on changed rows as scenario detection fractions. Unchanged scenarios have no defined detection fraction. Detecting deliberately constructed exact zeros is a rule check, not general sensor-fault recall.
- Keep gain, bias, inversion, sensor-swap, near-zero and nonzero-flatline misses visible. Do not tune a tolerance after inspecting results.

The local synthetic regression checks include stationary nonzero gravity, single zero axes, near-zero values, nonzero flatlines, missing-value precedence, input immutability, both policies and preventing rejected rows from reaching prediction. Aggregate research evidence is public; raw/derived vectors and model binaries stay local.

```bash
.venv/bin/python evaluate_gate.py artifacts/harth-input-v1 \
  --models artifacts/selective-v1 --output artifacts/zero-gate-v1
```

Use a new output directory and the pinned ML environment. Success requires exit code 0 and inspection of the generated report and stderr. This does not deploy or establish clinical safety.
