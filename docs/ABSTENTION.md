# Sigmoid calibration and selective prediction protocol

This is an **exploratory extension on a previously viewed test split**. The baseline test results are already public. The following choices are fixed before running this extension; the test is not represented as a new independent validation sample.

## Four distinct roles

Reuse the pinned HARTH source, windows, 30 features and participant assignments. Refit the already selected Random Forest using the unchanged baseline parameters on `fit` only. The baseline selected it by fitting-participant cross-validation. Do not reload arbitrary external model binaries.

Fit `CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid", ensemble=False)` on the three `calibration` participants only. Require all twelve labels in fitting and calibration data; reject missing-class calibration rather than silently accepting an ineffective class calibrator. Small positive counts remain a limitation even when every label is present. Sigmoid calibration is applied per class and normalised for multiclass predictions. It may change the most probable class. See the [pinned scikit-learn calibration reference](https://scikit-learn.org/1.7/modules/calibration.html).

Keep **both raw and sigmoid variants** in the report. There is no test-based choice of calibration method or winning variant. On the three `threshold` participants, select a separate maximum-probability threshold for each variant. Freeze those decisions before calculating any test prediction in this extension. The original forest is not retrained by calibration.

## Threshold selection

- Grid: 0.00, 0.01, ..., 1.00. Accept when maximum class probability is **greater than or equal** to the threshold.
- Eligible thresholds must retain at least 100 windows, at least 50% coverage, and at most 5% empirical error among accepted windows on the threshold role.
- Among eligible thresholds choose highest coverage, then lowest threshold. These fixed research targets are neither clinical requirements nor statistical risk guarantees.
- If none qualify, record `no_feasible_threshold`, use a null threshold meaning abstain on every input, and report risk as null. Do not loosen the target after viewing results.
- The threshold role lacks class 140. It cannot support any class-140-specific risk claim. Selection pools windows; it does not constrain every participant's error rate.

An abstained result has `label: null`, not an asserted class. A confidence score is not a guarantee of correctness. Input-quality rejection remains an upstream decision, separate from model abstention.

## Scores and curves

For each variant on test, report unfiltered classification metrics and:

- Natural-log loss: mean negative log probability of the true label, clipped at float64 epsilon for numerical evaluation only.
- Multiclass Brier: mean sum over all 12 classes of `(probability - one_hot_target)^2`, without the binary half-scaling; range 0–2.
- Top-label reliability: ten fixed equal-width confidence bins; left-inclusive/right-exclusive except the final bin includes 1. Report count, mean confidence and accuracy for each bin; empty means are null. ECE is the count-weighted absolute confidence–accuracy gap. This binned top-label diagnostic is not classwise calibration certification.
- Risk–coverage points at the fixed grid, plus an explicit reject-all point. Zero accepted predictions imply undefined risk, not zero error rate.
- At the selected operating point, overall, participant and true-class support, accepted count, errors, coverage and selective risk. True-class breakdown is evaluation-only; target labels are never inference features.
- Participant bootstrap: 1,000 draws with replacement, seed 7, summing per-person counts. Report percentile intervals for coverage and risk, and the number of valid risk replicates. A replicate with zero accepted windows has undefined risk and is excluded from the risk quantiles, not treated as zero. Four test people yield unstable intervals, without a population risk guarantee.

Log loss and Brier measure more than calibration alone; do not claim calibration improved solely because either decreased. The threshold dataset's empirical 5% target may fail on test.

## Reproduce

```bash
.venv/bin/python selective.py artifacts/harth-input-v1 --output artifacts/selective-v1
```

Requires the fixed `requirements-ml.txt` environment and prepared input directory from the baseline. A new output directory is required. `policy.json` is written before test predictions; `report.json` and `REPORT.md` indicate the completed experiment. Local model files are ignored, never published. A failed run can leave diagnostics; require exit code 0 and the final report. Input hashes must match the checked-in baseline evidence; source, protocol and environment hashes accompany the result. Hashes do not authenticate maliciously replaced local files.

Full research experiments remain local; CI uses synthetic temporary inputs. No deployment, clinical validation or guaranteed error control follows from this experiment.

Optional aggregate charts (Python 3.12 plotting snapshot):

```bash
.venv/bin/python -m pip install -r requirements-plots.txt
.venv/bin/python scripts/plot_selective.py artifacts/selective-v1/report.json \
  --output artifacts/selective-v1-plots
```

The plot shows top-label reliability and risk–coverage with the selected operating points. Empty reliability bins and undefined-risk points are omitted; their counts remain in JSON. `plot-source.json` ties the figure to its report and renderer hashes. Plotting does not fit or select any model parameters.
