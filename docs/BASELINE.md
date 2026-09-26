# Fixed HARTH baseline protocol

This protocol is fixed before the first model evaluation on the four held-out participants. It uses the [HARTH input protocol](HARTH_PROTOCOL.md), unchanged source manifest, and 30 feature columns. It compares three CPU baselines and makes no clinical or deployment claim.

## Models and selection

| Name | Frozen configuration |
| --- | --- |
| dummy | DummyClassifier, most_frequent. |
| logistic | StandardScaler then LogisticRegression; C=1, lbfgs, balanced class weights, max_iter=2000, tol=0.0001. Other parameters use scikit-learn 1.7.2 defaults. |
| random_forest | 200 trees, min_samples_leaf=2, balanced_subsample weights, seed=7, n_jobs=1. Other parameters use scikit-learn 1.7.2 defaults. |

Use four-fold GroupKFold without shuffling on **fitting participants only**. Each fold fits a fresh model; scaling and class weights are learned using that fold's training rows only. There is no hyperparameter search. Between logistic and random_forest, select the highest mean fold macro-F1 over the fixed 12 labels; alphabetical name breaks an exact tie. The dummy is a reference, not a candidate for subsequent calibration.

The runner writes `selection.json` before any test prediction. Each of the three models is then fitted on all 12 fitting participants and scored on the same four test participants. Calibration and threshold rows are validated on input but not fitted, predicted or scored in this experiment. A convergence warning stops the experiment; it is not suppressed to obtain a score.

The [scikit-learn GroupKFold](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GroupKFold.html) interface maintains group separation. The [LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html) and [RandomForestClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html) references specify the pinned estimator parameters.

## Metrics

- Accuracy: correctly predicted windows / all retained test windows.
- Macro-F1: arithmetic mean of per-class F1 over all twelve declared labels. If both true and predicted support are zero, that class contributes zero. This fixed-vocabulary metric is also used for CV and participant reports.
- Balanced accuracy: mean recall over classes with **true support** in the evaluated subset. The exact included support is reported, so a participant with fewer classes is not mistaken for a full twelve-class evaluation.
- Confusion matrices: rows are true labels, columns are predicted labels, with the full label order saved. Undefined precision/recall is null; absent-class F1 follows the zero convention above.
- Each test participant has a separate report. Also report mean participant accuracy, distinguishing equal-person weighting from pooled window accuracy.
- Approximate 95% percentile intervals: resample test participants, with replacement, four at a time, 1,000 replicates, seed 7. Sum their confusion matrices in each replicate. Use the same resampling sequence for every model. These are cluster bootstrap intervals, not independent-window intervals. Four participants are too few for stable population uncertainty estimates.

Report fit time and one test-batch prediction time separately. These local CPU measurements exclude CSV loading and are not online serving benchmarks. No GPU or cloud is used. Calibration, abstention, Brier scores and risk–coverage curves are outside this comparison.

## Reproduction

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
# First prepare the pinned archive using docs/HARTH_PROTOCOL.md.
.venv/bin/python baseline.py artifacts/harth-input-v1 --output artifacts/baseline-v1
```

Use a new output directory. A failed run may leave diagnostic files; only a completed `report.json` and exit code 0 indicate completion. Inputs are never modified. SHA-256 values connect the source archive, source manifest, window CSV, input report, experiment code, protocol and environment pins. Hashes detect mismatches; they are not signatures proving origin. Local `.joblib` models remain ignored; never load untrusted pickle/joblib files.

Only aggregate reports are checked in. Source recordings, derived feature rows and model binaries stay local. Results may differ slightly across numerical libraries/hardware; timings and binary model hashes need not repeat byte for byte. The pinned environment and single-thread numerical limit reduce variation.

## Evaluation boundary

The threshold role lacks class 140, as documented before training. It must remain visible in subsequent calibration/abstention work. Do not revise the split or feature extraction using held-out model results. Once the test has been viewed, follow-up changes informed by it are exploratory; this baseline is not a fresh test for every later version. Model comparison alone does not prove independent author proficiency, generalisation to new populations, or operational reliability.
