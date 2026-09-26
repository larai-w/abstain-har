# HARTH source and input protocol

## Selected distribution

Use the **22-participant UCI HARTH distribution** for the next research input path. The [dataset page](https://archive.ics.uci.edu/dataset/779/harth) explicitly assigns CC BY 4.0 and describes two accelerometers, lower back and thigh, at 50 Hz with acceleration in g. Cite Logacjov, Kongsvold, Bach, Bårdstu, and Mork, *HARTH*, DOI [10.24432/C5NC90](https://doi.org/10.24432/C5NC90), and the accompanying [research article](https://doi.org/10.3390/s21237853). Preserve attribution and identify transformations when reporting derived results.

The [author README at a fixed revision](https://github.com/ntnu-ai-lab/harth-ml-experiments/blob/dad2cfbe89a26f72f19770419469ac037de200df/README.md) identifies the UCI copy as v1.2 and the newer author-hosted dataset as v2.0 with 31 participants. These are different distributions. Do not substitute the current GitHub folder for this input. The software repository's MIT license is not the basis for assigning the UCI data license.

The [source manifest](../contracts/harth-v1.2-source.json) records the download URL, archive size and SHA-256, member names, CSV headers, and participant assignments. The downloaded archive contains a directory and 22 CSVs; no README or separate license file is bundled. No conflicting archive terms were found. This finding is specific to these bytes, not a blanket conclusion about other distributions.

**Implementation status:** source selection and header inspection are complete. No HARTH sensor rows have been parsed, windows constructed, or models trained here. The existing `har_input.py` accepts UCI HAR's 561-feature layout and cannot read HARTH. The following is the specification for a separate adapter, not an implemented command.

## Input contract

- Identify subjects by CSV filename stem; preserve zero padding. Do not infer their identities.
- Select six named channels: `back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`. Keep `timestamp` and `label` separate from model features.
- The inspected headers contain an extra `index` column in `S015.csv` and `S021.csv`, and an unnamed leading column in `S023.csv`. Permit only these recorded exceptions and exclude them from features. Validate each file against its manifest header; do not silently drop arbitrary extra columns.
- Validate row width, finite sensor values, activity vocabulary, and within-file timestamp ordering. Record gaps and duplicate/backward timestamps explicitly. Never invent UTC offsets or care-event arrival times.
- The activity vocabulary is `1,2,3,4,5,6,7,8,13,14,130,140`; retain all twelve classes. Report per-role support and absent classes after the split is fixed. Never move people or select another seed to improve class balance after inspecting outcomes.
- Keep raw archives and derived arrays in ignored local storage. Publish code and aggregate evidence, not sensor rows or participant recording timestamps.

## Participant separation

This project defines its own split; it is not an official HARTH train/test split. Sort the 22 filename stems by SHA-256 of UTF-8 `7:subject_id`, then assign the first four to test, next three to calibration, next three to threshold selection, and remaining twelve to fit. Exact assignments are in the manifest. No sensor values or labels were read to choose them.

Assign participants **before** windowing. Keep these same roles for every model. Fit-only grouped cross-validation may choose model parameters; calibration and threshold roles have separate purposes. Final test results must not select features, windows, models, calibration methods, or thresholds. Four test participants give limited evidence of generalisation; report participant-level results and uncertainty rather than treating thousands of windows as independent people.

## First window and feature specification

The initial protocol uses non-overlapping 250-sample windows (5 seconds at the nominal rate). Split streams at any non-positive timestamp delta or delta above 30 ms; report counts and retained duration. Do not interpolate over gaps or cross file/subject boundaries. Reject windows containing invalid sensor values and count incomplete trailing samples. Revisit timing assumptions only with documented fit-partition diagnostics before evaluation is frozen.

For each channel, calculate mean, population standard deviation, minimum, maximum, and root mean square: 30 named features. These definitions need no fitted statistics. Any later scaler or imputer must be fitted inside the fitting fold only. Subject IDs, timestamps, export indices, and label-derived statistics are not features.

Assign the window target by the most frequent sample label, breaking ties by ascending numeric label. Preserve mixed-label windows; record their label agreement fraction for evaluation diagnostics without filtering by that fraction. Filtering with ground-truth purity would create an unavailable deployment-time quality gate. Input quality rejection remains separate from model abstention.

## Next executable milestone

Implement a streaming HARTH adapter against this manifest, then produce an input-quality and window-retention report. Required acceptance cases include the three header exceptions, malformed rows, timestamp breaks, participant separation, mixed-label windows, and fit-only transformations. These checks have **not** been run yet. A subsequent baseline can compare DummyClassifier with a simple classifier on the same retained windows, followed by calibration and risk–coverage analysis. No performance targets or clinical claims follow from selecting this dataset.
