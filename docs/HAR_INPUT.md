# UCI HAR text-layout adapter and subject split planning

`har_input.py` implements a separate input path for the release's 561-feature text layout. It does not accept the three-feature synthetic care-summary bundle as a HAR vector. No scaler, imputer, model, calibration function, or abstention threshold is fitted here.

**Verification status:** executable validation and split planning have been tested on generated synthetic files in this layout. The official archive's documentation and feature metadata have been inspected; its feature matrices have not been loaded or used for evaluation in this project. See the [unresolved terms discrepancy](DATASETS.md#uci-har-distribution-inspection-2026-09-26) before research-data use.

## Reproduce the synthetic input example

Requires Python 3.11 or later and the standard library only. Both output paths must be new:

```bash
python3 scripts/make_synthetic_har.py --output data/synthetic-har
python3 har_input.py data/synthetic-har --dataset-kind synthetic \
  --calibration-subjects 1 --threshold-subjects 1 --seed 7 \
  --output artifacts/har-input
```

The generator creates 36 fake training rows and 12 fake test rows, with six classes and deliberately repeated feature names. These values are arbitrary arithmetic expressions, not recordings or realistic sensor simulations. The [checked-in report](../examples/har-input-v1/REPORT.md) is explicitly synthetic.

A `SYNTHETIC.txt` marker is required for synthetic mode and rejected in `uci-har-research` mode. This prevents accidental relabelling of this fixture; it does not authenticate any dataset. Do not add that marker to real research recordings.

## Files and positional schema

Point the adapter at a directory with:

```text
features.txt
activity_labels.txt
train/X_train.txt
train/y_train.txt
train/subject_train.txt
test/X_test.txt
test/y_test.txt
test/subject_test.txt
```

The X, y, and subject files must have the same non-zero number of rows within each partition. Row `i` in each describes the same window. Row IDs are `train:i` or `test:i`, with one-based line numbering; they are not timestamps.

| Input | Validation |
| --- | --- |
| Feature definitions | Exactly 561 consecutive positions, each with a non-empty source name. |
| Feature vectors | Exactly 561 finite numeric values; release range [-1, 1] with 1e-6 numerical tolerance. |
| Labels | One integer in 1–6 per row, with the documented six-class mapping. |
| Subject IDs | One integer in 1–30 per row. |
| Official partitions | No subject may appear in both train and test. |
| Source files | Hash all eight files before and after loading; reject a detected change. |

Feature identifiers are **`f001` … `f561`**, preserving the original order. Source names remain metadata and may repeat. Inspection of the downloaded metadata found 561 positions but only 477 distinct names. Using a dictionary keyed only by those names would lose columns.

The adapter uses the release-provided features as given. It does not reconstruct sensor filtering, windowing, or feature extraction. It invents no arrival timestamps, units for individual derived features, or connection to the care-history timeline. The basic layout check does not authenticate an official release or require its full original row counts; file hashes and source records are necessary for an actual experiment.

## Split policy

The official test subjects and rows stay exclusively in `test`. Only the original training subjects are assigned to the other roles:

1. Rank subject IDs by SHA-256 of `seed:subject_id`.
2. Reserve the requested number for `calibration`.
3. Reserve the requested number for `threshold`.
4. Keep all remaining training subjects in `fit`, requiring at least two for later grouped model selection.

Default reservations are three calibration subjects and three threshold subjects. The smaller synthetic example explicitly requests one each. Requests that leave fewer than two fit subjects fail; subjects are never reused to satisfy a requested count.

Feature values and activity labels do not influence assignment. Class counts and missing classes are reported **after** assignment as diagnostics. Missing classes do not cause the tool to try another seed or move people silently. Small-sample suitability must be assessed before any model comparison. This hash-based rule is reproducible, not a claim of optimal or representative sampling.

Future cross-validation inside `fit` must also keep people disjoint. Any additional preprocessing must be fitted only on the fitting portion of each fold. The fixed release already contains preprocessing done by its authors; this adapter cannot establish that original preprocessing's evaluation properties.

## Outputs and boundaries

`report.json` contains source and adapter hashes, feature metadata, per-partition counts, subjects per role, row IDs, label counts, and missing classes. `REPORT.md` summarises the result. Neither includes raw feature vectors. `load_dataset()` exposes validated rows to future local training code; no training entry point is provided.

Input files are not modified and existing output directories are never overwritten. Success returns 0; malformed or inconsistent input returns 2. The CLI has no downloading, account, or cloud actions.

For a locally held research copy with resolved use conditions, the layout interface is:

```bash
python3 har_input.py 'data/UCI HAR Dataset' --dataset-kind uci-har-research \
  --calibration-subjects 3 --threshold-subjects 3 --seed 7 \
  --output artifacts/uci-har-input
```

This command has **not** been run against the downloaded research matrices in this project. Do not interpret synthetic examples or passing CI as results on UCI HAR. Do not tune the split using test outcomes.
