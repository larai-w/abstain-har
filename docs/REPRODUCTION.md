# Reproduce the recorded pipeline

## Scope

Use a fresh checkout and virtual environment. The recorded reproduction uses Python 3.12.10 on the same macOS host as the original experiments, with a new checkout, empty virtual environment and freshly generated features/models. It reuses only the pinned source archive; package installation may use pip's wheel cache. It is not an independent reviewer, different hardware or Linux research-model reproduction. Linux CI covers synthetic tests separately.

Raw recordings and model binaries are not hosted in this repository. Download the archive from the [dataset source](https://archive.ics.uci.edu/dataset/779/harth), respecting its license and attribution. Confirm the SHA-256 in `contracts/harth-v1.2-source.json`; `harth_input.py` also checks it before parsing. An existing local copy with that hash can be used. Do not copy previous generated windows or models into the fresh checkout when checking full reproduction.

## Environment and synthetic checks

```bash
git clone https://github.com/larai-w/abstain-har.git
cd abstain-har
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

Record the checkout revision with `git rev-parse HEAD`. Full-run evidence identifies the exact revision that was exercised. Optional plotting dependencies are not required. Never load joblib artifacts downloaded from an untrusted party.

## Regenerate features, models and evaluations

Place the pinned archive at `data/harth.zip`. Every output directory below must be new. Inspect both stdout and stderr, and stop on any nonzero exit code; the existence of a partial output directory does not establish success.

```bash
.venv/bin/python harth_input.py data/harth.zip --output artifacts/harth-input-v1
.venv/bin/python baseline.py artifacts/harth-input-v1 --output artifacts/baseline-v1
.venv/bin/python selective.py artifacts/harth-input-v1 --output artifacts/selective-v1
.venv/bin/python robustness.py artifacts/harth-input-v1 \
  --models artifacts/selective-v1 --output artifacts/robustness-v1
.venv/bin/python evaluate_gate.py artifacts/harth-input-v1 \
  --models artifacts/selective-v1 --output artifacts/zero-gate-v1
.venv/bin/python local_predict.py examples/local-inference-v1/request.json \
  --models artifacts/selective-v1 --variant sigmoid --input-policy exact-zero-v1 \
  --output artifacts/zero-gate-v1/local-response.json
```

A calibration warning about the least populated class having one member is expected and must be retained. It documents limited calibration support, not a reason to silently change the split or suppress a failure. A convergence error or other nonzero exit is not equivalent to that recorded warning.

## Compare evidence

```bash
.venv/bin/python scripts/compare_reproduction.py --artifacts artifacts \
  --output artifacts/reproduction-comparison.json
```

The comparator checks six complete aggregate JSON trees: prepared input, baseline, selective prediction, stress, zero gate and local response. Integers, strings, booleans, nulls, list lengths and keys must match exactly. Finite floating-point values use absolute tolerance 1e-12 and zero relative tolerance. That tolerance is fixed in code rather than chosen after seeing a mismatch.

The report lists every excluded path: wall-clock completion timestamps, Git execution revision, fit/prediction timings and Python/platform descriptions. Package versions and source/input hashes are still compared. Model binary hashes are checked separately and remain visible. Exit 0 requires both semantic comparisons and model hash comparisons to succeed; exit 1 means a mismatch, and exit 2 means comparison could not complete. Existing comparison files are not overwritten.

The stress and inference tools require the recorded artifact hashes before loading models. A rebuild on other numerical libraries or hardware might have matching metrics but different model bytes and be refused. Do not overwrite the checked-in reference, loosen the guard, or call this a successful end-to-end reproduction. Record the stage and discrepancy; a separately designed provenance workflow would be needed to admit different artifacts.

## Recorded evidence

[Fresh-environment run record](../examples/reproduction-v1/run.json) · [Aggregate comparison](../examples/reproduction-v1/comparison.json)

The run record distinguishes original pipeline checks from the later comparator tests. Timings are local batch elapsed times, not service latency measurements. No new research split or independent generalisation evidence is introduced by rerunning the same pipeline.
