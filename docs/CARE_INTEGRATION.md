# Synthetic care-summary input integration

This path prepares inputs only. It does not train a classifier, infer activities, calibrate probabilities, select an abstention threshold, or estimate selective risk.

## Run the checked-in example

Requires Python 3.11 or later; no third-party packages or network calls are needed:

```bash
python3 prepare_inputs.py examples/integration/bundle.json \
  --splits examples/integration/splits.json --output artifacts/prepared.json
python3 -m unittest discover -s tests -v
```

The output file must be new. Success exits 0; validation errors exit 2 without creating a result. Input files are never modified.

Expected result: six synthetic samples, four eligible input rows (one per role), and two exclusions. One sample has no known observations at its cutoff; another fails an observation-quality check. These are **input gate outcomes**, not model abstentions. See [prepared.json](../examples/integration/prepared.json) for the complete result.

## What is exchanged

The producer is [Open Care Evidence Toolkit](https://github.com/larai-w/open-care-evidence-toolkit). Read its [integration contract](https://github.com/larai-w/open-care-evidence-toolkit/blob/main/INTEGRATION.md) for the full semantics.

The identical [contract registry](../contracts/care-bundle-v1.json) in both repositories defines version 1 of `synthetic_indicator_summary_v1`. It is a version registry, not a standalone JSON Schema validator. Its byte hash is checked before consumption.

Each eligible feature vector is exactly:

```text
[observed_fraction, known_count, unknown_count]
```

The fraction is observed / (observed + not_occurring), with not_recorded excluded from that denominator. These are synthetic indicator summaries, **not** UCI HAR's features or sensor windows. Sample/subject identifiers, cutoffs, and audit hashes remain separate from the numerical vector. The prepared batches have no activity labels and cannot train a HAR classifier as provided.

## Validation boundary

The consumer rejects:

- unsupported contract/version, dataset kind, feature set, feature order, or contract hash;
- missing or duplicate sample IDs;
- selected observations received after their cutoff, or observed after receipt;
- invalid audited statuses, contradictory lineage, unsupported quality rules, and inconsistent gate decisions;
- non-finite numbers, boolean counts, additional/missing features, or features that disagree with audited counts;
- feature values attached to excluded rows;
- subject overlap within/across roles or incomplete/extra subject assignments.

All four roles (`fit`, `calibration`, `threshold`, `test`) must have declared subjects. Every subject in the bundle appears exactly once in that manifest. If a role has no eligible rows after gating, its zero count remains visible; the output does not claim that a training/evaluation protocol is ready.

These checks validate the consistency of declared data. They cannot authenticate subject identity or source timestamps, detect subject aliases, or reconstruct absent history. The consumer trusts the producer's resolution of the full history and its quality findings, while independently rechecking the exported cutoffs, statuses, lineage shape, and arithmetic. Hashes are provenance aids, not signatures.

Future records may alter a full-history audit hash without changing any historical feature. Do not use full-archive audit metadata as historical model inputs.

## Fresh two-repository integration

Clone both repositories side by side and run from this repository:

```bash
python3 scripts/check_care_integration.py --toolkit ../open-care-evidence-toolkit
```

The script:

1. Compares the two contract files byte-for-byte.
2. Exports a fresh bundle from the toolkit's fixed synthetic request into a temporary directory.
3. Runs this consumer against the checked-in subject split.
4. Compares the complete result to the recorded prepared example.

CI pins the producer commit, so unrelated upstream changes cannot silently alter the example. To update the producer, review changes, regenerate both copies of the example bundle and prepared result, update the pin, and rerun the cross-repository check. Local use of a newer unpinned checkout can intentionally fail the comparison when source hashes change.

## Path to a real HAR experiment

A later adapter needs its own versioned schema for sensor units, sample rate, windows/features, activity labels, and research-data provenance. Its split manifest must be defined at participant level before fitting preprocessing or models. The synthetic contract exercise supplies none of the model-quality evidence required for that experiment.
