# abstain-har

Human activity recognition with explicit abstention and subject-disjoint evaluation.

**Status: synthetic input integration implemented; model research pending.** No model has been trained or evaluated in this repository. There are no model benchmark results, deployed endpoints, or clinical validation claims.

## Research question

How does a classifier perform on people it has not seen during training, and how does its error rate change when it can abstain from answering?

The evaluation approach separates participants across model fitting, probability calibration, threshold selection, and final testing. Results should report both the fraction of predictions accepted (coverage) and the error rate among accepted predictions (selective risk), including participant-level variation.

Abstention does not guarantee correctness. A model can be confidently wrong, especially when sensor inputs or populations change. With zero accepted predictions, selective risk is undefined, not zero.

## Try the synthetic input integration

A [versioned integration with Open Care Evidence Toolkit](docs/CARE_INTEGRATION.md) now checks as-of data provenance, input quality decisions, and subject-disjoint input splits:

```bash
python3 prepare_inputs.py examples/integration/bundle.json \
  --splits examples/integration/splits.json --output artifacts/prepared.json
```

The fixed example contains six synthetic samples: four eligible input rows, one quality rejection, and one insufficient-data exclusion. The three-value indicator summaries are not HAR sensor features. No training, prediction, or model abstention takes place.

## Repository contents

- [Synthetic integration](docs/CARE_INTEGRATION.md): contract, limitations, and fresh two-repository reproduction.
- `prepare_inputs.py`: checked feature batches and explicit exclusions.
- `tests/`: subject separation, temporal availability, feature consistency, and CLI regression checks.
- [Dataset notes](docs/DATASETS.md): candidate sources, attribution, and compatibility limits.
- `scripts/check_public_repo.py`: public-content checks for staged or tracked files.
- `.github/workflows/public-content.yml`: the same content check in CI.

There are no training, inference, or model-evaluation commands yet. CI checks repository content, consumer behaviour, and the integration against a pinned producer revision. Passing CI is not evidence of model quality.

## Local content check

Requires Python 3.11 or newer and Git; no third-party Python packages are required.

```sh
python3 scripts/check_public_repo.py --tracked
```

Before committing:

```sh
python3 scripts/check_public_repo.py --staged
git diff --cached --check
```

The content check is a guard against known patterns, not a complete security or privacy audit. Review every file before publication.

## Scope and limitations

This is a research software project, not a medical device or a clinical decision system. Public research datasets must not be described as synthetic data. Any future synthetic examples must be labelled separately from predictions derived from research recordings.

Do not contribute private health records, identifiable personal information, credentials, or course assignment materials. Dataset licenses remain separate from this repository; raw datasets and trained models are not included.
