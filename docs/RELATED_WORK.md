# Related work: personalisation and set-valued uncertainty (2026)

This note compares two 2026 papers with the **current, recorded** abstain-har protocol. It is a reading aid for deciding future experiments. Nothing here has been re-implemented or run, and no result below is a claim about this repository.

## Current protocol (for reference)

| Aspect | abstain-har today |
|---|---|
| Data | UCI HARTH v1.2, 22 participants, back and thigh accelerometers at 50 Hz ([protocol](HARTH_PROTOCOL.md)) |
| Windows | Non-overlapping 250 samples (5 s) |
| Features | 30 named summaries (mean, population SD, min, max, RMS for six channels) |
| Model | Random Forest selected by fit-only participant-grouped CV ([baseline](BASELINE.md)) |
| Split | Fixed by hash before reading data: fit 12, calibration 3, threshold 3, test 4 participants |
| Output | One of `input_rejected`, `model_abstained`, `accepted`; rejected and abstained rows have `label: null` ([local inference](LOCAL_INFERENCE.md)) |
| Metrics | Coverage (answered fraction) and selective risk (errors among answers), with participant-level variation ([abstention](ABSTENTION.md)) |

## BayaHAR — few-shot user adaptation

Burzer, Riedel, Beigl, Röddiger. *BayaHAR: Lightweight Bayesian Few-Shot User Adaptation for On-Device Personalized Human Activity Recognition.* arXiv [2606.04798v2](https://arxiv.org/abs/2606.04798v2); ISWC 2026, DOI [10.1145/3830727.3834833](https://doi.org/10.1145/3830727.3834833). Code: [maxbrzr/baya-har](https://github.com/maxbrzr/baya-har).

Why it matters here: abstain-har evaluates on unseen participants and records large per-participant differences. BayaHAR addresses the same unseen-user drop by adapting to a new user with a small amount of that user's labelled data, without gradient updates, and its evaluation includes HARTH.

| Aspect | BayaHAR (as described by the authors) | abstain-har |
|---|---|---|
| Model | Pre-trained deep network turned into a prototype model, updated by Bayesian inference | Random Forest on 30 summaries |
| Windows | 3 s | 5 s |
| Evaluation | Leave-one-subject-out on four datasets, with support/query separation for the adapted user | One fixed participant split with separate fit, calibration, threshold and test roles |
| User data needed | A few seconds of labelled data **per activity** for the new user | None at prediction time |
| Uncertainty | Not a guarantee of prediction-time uncertainty or sensor-fault handling | Explicit abstention and input rejection, without guarantees |

Not directly comparable: it is not a drop-in replacement for the Random Forest, the "3 seconds" is per activity rather than total calibration time, and its weakly supervised variant still needs to know which activities were performed. Any future trial should be a separate experiment that keeps each adapted user's support and query data apart and reports participants whose results get **worse** after adaptation.

## HDUQ-HAR — conformal prediction sets on microcontrollers

Lamaakal, Yahyati, Maleh, El Makkaoui, Ouahbi. *Efficient uncertainty aware human activity recognition on microcontrollers using hyperdimensional computing and conformal prediction.* Scientific Reports 16:26262 (2026). DOI [10.1038/s41598-026-57375-8](https://doi.org/10.1038/s41598-026-57375-8).

Why it matters here: the [robustness evaluation](ROBUSTNESS.md) shows confident wrong answers under changed inputs. Instead of a single label or abstention, the paper returns a **set of candidate labels** with reasons derived from distances, margins and internal vote dispersion, calibrated per class.

Concepts that must stay separate if this is ever explored:

| Concept | Meaning | Present in abstain-har |
|---|---|---|
| Input rejection | Input fails a validity rule before prediction | Yes (`input_rejected`) |
| Model abstention | Valid input, confidence below a fixed threshold | Yes (`model_abstained`) |
| Prediction set | Several candidate labels returned together | No |
| Set coverage | Fraction of cases whose true label is inside the set | No — **different from abstain-har's coverage** (answered fraction) |
| Reason codes | Why a set or abstention was returned | No |

Cautions recorded from reading: not evaluated on HARTH or on these features; conformal guarantees assume exchangeability, which time dependence and distribution shift can break; the paper describes weight tuning and quantile computation on the same calibration split, so finite-sample guarantees need further checking; no public implementation link was found in the article. Adoption is on hold.

## Possible next steps (not started)

1. Keep this comparison as the reference before any re-implementation.
2. If set-valued output is explored, specify the response schema first so that `input_rejected`, `model_abstained`, a prediction set and reason codes cannot be confused, and use separate data for score selection and final calibration.
3. Report set coverage, set size and risk on single-label answers as separate numbers.
