# Fixed-model feature stress protocol

This is an exploratory stress experiment on the same previously viewed HARTH test participants. No new population, real device fault, or independent validation dataset is introduced. Perturbations are applied to **derived window summaries**, not raw sensor recordings. The labels remain the original activity targets; no label determines the perturbation.

## Frozen components

Reuse both local raw and sigmoid model artifacts from the recorded selective-prediction run, only after matching their SHA-256 to the checked-in report. Models, calibration, feature order and thresholds (raw 0.43; sigmoid 0.64) remain fixed. No `.fit()` or threshold search occurs. The input and source hashes must match that report. The clean control must reproduce its selected counts and unfiltered confusion matrices before stress results are produced.

Joblib files are executable Python serialization: use only these locally generated artifacts whose hashes match the recorded evidence. The runner does not download model files. Local hashes are not an authentication system against malicious replacement of both evidence and artifacts.

## Scenarios fixed before execution

| Scenario | Transformation on the same test windows |
| --- | --- |
| clean | Identity control. |
| missing_thigh_10 / 30 / 100 | Replace all 15 thigh summaries by NaN on ceil(N × fraction) rows. A seed-7 permutation defines nested subsets; no labels are used. |
| zero_back | Replace all 15 back summaries by zero on every row: an undetected constant-output scenario. |
| gain_half / gain_double | Multiply all summaries by positive gain 0.5 or 2.0. |
| bias_back_x_small / large | Add 0.25 g or 1 g to back-x mean/min/max; leave std unchanged; transform RMS as sqrt(rms² + 2 × bias × mean + bias²). |
| invert_back_x | Negate mean; new min=-old max and new max=-old min; std and RMS unchanged. |
| swap_sensors | Exchange the back and thigh blocks, preserving the within-block feature order. |

These algebraic gain, offset and inversion rules match the corresponding raw-signal transformations of these five summaries, but the experiment still does not model sensor electronics, temporal faults, arbitrary rotations or realistic missingness mechanisms. Sensor swapping is a column-contract fault, not proof of transfer across placements. Zero-valued summaries may pass the deliberately limited input gate.

## Input gate and accounting

Require a numeric N×30 array. A window with any non-finite feature is **input_rejected**, never sent to the model. Do not impute it. A finite window is either **accepted** or **model_abstained**, using the fixed threshold. This finite-value gate is not a general out-of-distribution detector or a new claim about the raw-data adapter.

Report three mutually exclusive counts and verify their sum equals the original cohort. Overall coverage uses all original windows as denominator, including rejected inputs. Also report coverage among valid inputs separately. Selective risk uses only accepted predictions; zero accepted predictions means null risk. Report unfiltered classifier errors on valid inputs separately. Results include per-subject and per-true-class counts, and 1,000 participant-bootstrap replicates using the prior seed-7 method. Four participants remain insufficient for stable population uncertainty estimates.

## Paired comparison

Each transformed row is paired with its original row. Record the 3×3 transition table from clean status to stressed status, the number of clean accepted/correct rows becoming accepted/wrong, and errors before/after among rows accepted in both conditions. The latter has a fixed shared denominator. Also report overall changes in coverage and risk without implying the accepted populations are identical.

Persist scenario definitions, changed-row counts, mask hashes, reference/model/source/environment hashes and class/participant breakdowns. Do not publish derived feature rows or model binaries. The stress result cannot establish future error control; high-confidence errors are an expected failure mode to measure rather than hide.

## Reproduce

```bash
.venv/bin/python robustness.py artifacts/harth-input-v1 \
  --models artifacts/selective-v1 --output artifacts/robustness-v1
```

Use the pinned ML environment and a new output directory. Prepare the input and selective model artifacts using the preceding protocols first. Success requires exit code 0 and the completed report. Existing CI covers the preceding input and selective-prediction components with synthetic data; it does not exercise this stress runner or download research data or local model artifacts. No model or threshold is changed after inspecting a stress case.
